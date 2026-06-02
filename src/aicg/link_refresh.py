"""Deterministic ``refresh_links`` handler.

For each URL the audit flagged broken in a single doc, try (in order):

1. **Re-check** — HEAD the URL again. Audit results are often stale
   (transient DNS / 5xx / rate-limit). If the link is now alive, no
   replacement needed.
2. **Follow redirects** — if the URL now returns 3xx, follow it and use
   the final 200 URL as the replacement.
3. **Wayback Machine** — query ``archive.org/wayback/available`` for
   the most recent good snapshot and use that.

If none of those produce a verified replacement, leave the URL alone
and let the next audit re-emit the work item.

When at least one replacement is found, write the doc, commit on a
branch, open a PR — **never auto-merge**. Content changes from this
handler go through human review (the curriculum has public readers
now).

Pure stdlib + ``urllib``; no Claude in the loop. Two fetcher functions
are injectable so the handler is testable without real HTTP.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .state import utc_now

USER_AGENT = "AICG-link-refresh/1.0 (+https://github.com/ai-infra-curriculum)"
HTTP_TIMEOUT = 10.0
MAX_REDIRECTS = 5
WAYBACK_API = "https://archive.org/wayback/available"

# Domains we will never substitute INTO a curriculum doc as a
# replacement for an external citation. VeriSwarm.ai is the
# maintainer-level attribution but practitioner sources, not
# authoritative ones — see source policy.
DISALLOWED_REPLACEMENT_DOMAINS = frozenset(
    {
        "veriswarm.ai",
        "www.veriswarm.ai",
        "localhost",
        "127.0.0.1",
    }
)


@dataclass(frozen=True)
class LinkResolution:
    original: str
    replacement: str | None
    source: str  # "alive" | "redirect" | "wayback" | "unresolved"
    note: str = ""

    @property
    def needs_edit(self) -> bool:
        return (
            self.replacement is not None
            and self.replacement != self.original
            and self.source != "alive"
        )


HttpFetcher = Callable[[str, str], "FetchResult"]  # (url, method) -> FetchResult
WaybackFetcher = Callable[[str], str | None]  # url -> archived_url | None


@dataclass(frozen=True)
class FetchResult:
    status: int | None
    final_url: str
    error: str = ""


def default_http_fetcher(url: str, method: str = "HEAD") -> FetchResult:
    """One HTTP request, no redirect-following (caller follows manually)."""
    # Guard scheme up front. urllib raises ValueError("unknown url type: ...")
    # on scheme-less URLs (e.g. a relative redirect Location header that
    # leaked into the queue), and ValueError is not in the catch list below.
    # Unhandled, it would crash the entire remediate tick.
    if not url.startswith(("http://", "https://")):
        return FetchResult(
            status=None, final_url=url, error=f"unsupported url scheme: {url!r}"
        )
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT}
    )

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            return None  # disable automatic follow

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=HTTP_TIMEOUT) as resp:
            return FetchResult(status=resp.status, final_url=resp.geturl())
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location", "") if exc.headers else ""
        # Resolve relative Location headers against the request URL — some
        # servers return paths only (`/en/blog/...`) which downstream code
        # cannot HEAD without a scheme.
        if location:
            location = urllib.parse.urljoin(url, location)
        return FetchResult(status=exc.code, final_url=location or url)
    except urllib.error.URLError as exc:
        return FetchResult(status=None, final_url=url, error=str(exc.reason))
    except TimeoutError:
        return FetchResult(status=None, final_url=url, error="timeout")
    except (OSError, ValueError) as exc:
        return FetchResult(status=None, final_url=url, error=str(exc))


def default_wayback_fetcher(url: str) -> str | None:
    """Ask archive.org for the most recent good snapshot of ``url``."""
    api_url = f"{WAYBACK_API}?url={urllib.request.quote(url, safe=':/?&=')}"
    request = urllib.request.Request(
        api_url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    snapshot = (
        data.get("archived_snapshots", {}).get("closest") or {}
    )
    if not snapshot.get("available"):
        return None
    if str(snapshot.get("status", "")) != "200":
        return None
    return snapshot.get("url") or None


def resolve_link(
    url: str,
    *,
    http_fetcher: HttpFetcher = default_http_fetcher,
    wayback_fetcher: WaybackFetcher = default_wayback_fetcher,
) -> LinkResolution:
    """Try re-check → redirect chase → wayback."""
    result = http_fetcher(url, "HEAD")
    if result.status and 200 <= result.status < 300:
        return LinkResolution(original=url, replacement=url, source="alive")

    # Follow up to MAX_REDIRECTS redirects, then HEAD-check the
    # final URL to confirm it actually returns 200.
    if result.status and 300 <= result.status < 400 and result.final_url:
        seen = {url}
        current = result.final_url
        for _ in range(MAX_REDIRECTS):
            if current in seen:
                break
            seen.add(current)
            hop = http_fetcher(current, "HEAD")
            if hop.status and 200 <= hop.status < 300:
                if _replacement_allowed(current):
                    return LinkResolution(
                        original=url, replacement=current, source="redirect"
                    )
                break
            if hop.status and 300 <= hop.status < 400 and hop.final_url:
                current = hop.final_url
                continue
            break

    # Wayback fallback.
    archived = wayback_fetcher(url)
    if archived and _replacement_allowed(archived):
        return LinkResolution(
            original=url, replacement=archived, source="wayback"
        )

    return LinkResolution(
        original=url,
        replacement=None,
        source="unresolved",
        note=result.error or (f"status {result.status}" if result.status else "no response"),
    )


def apply_replacements(body: str, resolutions: list[LinkResolution]) -> str:
    """Substitute each resolved URL in the doc body.

    Replace longest first so a URL that is a prefix of another doesn't
    eat the longer one. Replaces the literal URL string — works for
    both ``[text](url)`` markdown links and bare URLs.
    """
    pairs = sorted(
        (r for r in resolutions if r.needs_edit and r.replacement),
        key=lambda r: len(r.original),
        reverse=True,
    )
    new_body = body
    for r in pairs:
        new_body = new_body.replace(r.original, r.replacement)
    return new_body


def handle_refresh_links_item(
    *,
    workspace: Path,
    item: dict[str, Any],
    http_fetcher: HttpFetcher = default_http_fetcher,
    wayback_fetcher: WaybackFetcher = default_wayback_fetcher,
    open_pr: bool = True,
) -> dict[str, Any]:
    """Resolve and apply replacements for one ``refresh_links`` item.

    Returns a result dict describing what happened. Always opens a PR
    when at least one URL was replaced; never modifies main directly.
    """
    repo = item.get("repo", "")
    path_rel = item.get("path", "")
    repo_path = workspace / repo
    target = repo_path / path_rel
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "refresh_links",
        "repo": repo,
        "path": path_rel,
        "status": "started",
    }

    if not target.exists():
        result["status"] = "skipped"
        result["reason"] = "file_not_found"
        return result

    urls = [d.get("url") for d in (item.get("details") or []) if d.get("url")]
    if not urls:
        result["status"] = "skipped"
        result["reason"] = "no_urls"
        return result

    resolutions = [
        resolve_link(
            url, http_fetcher=http_fetcher, wayback_fetcher=wayback_fetcher
        )
        for url in urls
    ]
    result["resolutions"] = [
        {"url": r.original, "replacement": r.replacement, "source": r.source, "note": r.note}
        for r in resolutions
    ]

    edits = [r for r in resolutions if r.needs_edit]
    alive = [r for r in resolutions if r.source == "alive"]
    unresolved = [r for r in resolutions if r.source == "unresolved"]

    result["summary"] = {
        "edited": len(edits),
        "now_alive": len(alive),
        "unresolved": len(unresolved),
    }

    if not edits:
        # Either everything is now alive (audit was stale) or we can't
        # fix anything. Either way: no PR, leave the item for the next
        # audit cycle to re-evaluate.
        if alive and not unresolved:
            result["status"] = "no_action_now_alive"
        else:
            result["status"] = "no_replacements_found"
        return result

    body = target.read_text(encoding="utf-8")
    new_body = apply_replacements(body, resolutions)
    if new_body == body:
        # Defensive — every needs_edit resolution should have produced
        # a substitution. If body is unchanged, surface that instead of
        # opening an empty PR.
        result["status"] = "no_diff_after_apply"
        return result

    target.write_text(new_body, encoding="utf-8")
    result["status"] = "edited"

    if open_pr:
        pr_outcome = _open_link_refresh_pr(repo_path, repo, path_rel, edits, unresolved)
        result["pr"] = pr_outcome
        result["status"] = pr_outcome.get("status", "edited")

    return result


def _replacement_allowed(url: str) -> bool:
    """Source-policy filter for replacements."""
    if not url.startswith(("http://", "https://")):
        return False
    host = re.sub(r"^https?://", "", url, count=1).split("/", 1)[0].lower()
    host = host.split(":", 1)[0]  # strip port
    return host not in DISALLOWED_REPLACEMENT_DOMAINS


def _open_link_refresh_pr(
    repo_path: Path,
    repo: str,
    path_rel: str,
    edits: list[LinkResolution],
    unresolved: list[LinkResolution],
) -> dict[str, Any]:
    today = utc_now()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", path_rel.lower()).strip("-")[:80]
    branch = f"aicg/{today}/refresh-links/{slug or 'doc'}"
    title = f"docs: refresh broken links in {path_rel}"

    lines = [
        "## Refresh broken links",
        "",
        f"`{path_rel}` — replaced {len(edits)} broken link(s) "
        "using deterministic resolution (re-check / redirect chase / "
        "Wayback Machine). No Claude in the loop.",
        "",
        "**Replacements:**",
        "",
        "| Source | Old | New |",
        "| --- | --- | --- |",
    ]
    for r in edits:
        lines.append(f"| {r.source} | {r.original} | {r.replacement} |")
    if unresolved:
        lines += ["", "**Still unresolved (left in place):**", ""]
        for r in unresolved:
            lines.append(f"- {r.original} — {r.note or 'no resolution found'}")
    lines += [
        "",
        "Please eyeball the replacements — Wayback snapshots in "
        "particular may have stale rendering. Merge when satisfied.",
    ]
    body = "\n".join(lines)

    steps = [
        ["git", "-C", str(repo_path), "fetch", "origin", "main"],
        ["git", "-C", str(repo_path), "checkout", "main"],
        ["git", "-C", str(repo_path), "pull", "--ff-only"],
        ["git", "-C", str(repo_path), "checkout", "-B", branch],
        ["git", "-C", str(repo_path), "add", path_rel],
        ["git", "-C", str(repo_path), "commit", "-m", title],
        ["git", "-C", str(repo_path), "push", "-u", "origin", branch],
        ["gh", "pr", "create", "--base", "main", "--title", title, "--body", body],
    ]
    trail: list[dict[str, Any]] = []
    pr_url: str | None = None
    for cmd in steps:
        completed = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, check=False
        )
        trail.append(
            {
                "step": cmd[1] if cmd[0] == "git" else cmd[0],
                "argv": cmd[-1] if cmd[0] == "gh" else " ".join(cmd[-3:]),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-300:],
                "stderr_tail": completed.stderr[-300:],
            }
        )
        if cmd[:3] == ["gh", "pr", "create"] and completed.returncode == 0:
            pr_url = completed.stdout.strip()
        if completed.returncode != 0 and cmd[1] != "commit":
            subprocess.run(
                ["git", "-C", str(repo_path), "checkout", "main"],
                capture_output=True, text=True, check=False,
            )
            return {"status": "pr_failed", "branch": branch, "steps": trail}
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", "main"],
        capture_output=True, text=True, check=False,
    )
    return {"status": "pr_opened", "branch": branch, "pr_url": pr_url, "steps": trail}
