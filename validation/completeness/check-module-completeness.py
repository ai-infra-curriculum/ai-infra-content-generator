#!/usr/bin/env python3
"""
Module Completeness Checker

Validates that module content meets minimum quality standards for production curriculum.

Usage:
    python check-module-completeness.py lecture-notes.md
    python check-module-completeness.py --directory modules/ --verbose
    python check-module-completeness.py --json --output report.json lecture-notes.md

Standards checked:
- Word count (12,000+ words)
- Code examples (10+ complete examples)
- Case studies (3+ with metrics)
- Troubleshooting section (7+ issues)
- Required sections present
- Learning objectives (8+)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class CompletenessChecker:
    """Checks module completeness against quality standards."""

    STANDARDS = {
        'min_words': 12000,
        'min_code_examples': 10,
        'min_case_studies': 3,
        'min_troubleshooting': 7,
        'min_learning_objectives': 8,
        'required_sections': [
            'overview',
            'learning objectives',
            'introduction',
            'case stud',  # Matches "case study" or "case studies"
            'best practices',
            'troubleshooting',
            'summary',
            'resources'
        ]
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}

    def check_file(self, filepath: Path) -> Dict:
        """Check completeness of a single file."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Checking: {filepath}")
            print(f"{'='*70}\n")

        content = filepath.read_text(encoding='utf-8')

        results = {
            'file': str(filepath),
            'passed': True,
            'checks': {}
        }

        # Run all checks
        results['checks']['word_count'] = self._check_word_count(content)
        results['checks']['code_examples'] = self._check_code_examples(content)
        results['checks']['case_studies'] = self._check_case_studies(content)
        results['checks']['troubleshooting'] = self._check_troubleshooting(content)
        results['checks']['required_sections'] = self._check_sections(content)
        results['checks']['learning_objectives'] = self._check_learning_objectives(content)

        # Determine overall pass/fail
        for check in results['checks'].values():
            if not check['passed']:
                results['passed'] = False

        self._print_results(results)
        return results

    def _check_word_count(self, content: str) -> Dict:
        """Check if word count meets minimum."""
        # Remove code blocks to get accurate word count
        no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        # Count words
        words = len(no_code.split())

        target = self.STANDARDS['min_words']
        passed = words >= target
        percentage = (words / target) * 100

        result = {
            'passed': passed,
            'value': words,
            'target': target,
            'percentage': percentage,
            'message': f"Word count: {words:,} / {target:,} ({percentage:.1f}%)"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")

        return result

    def _check_code_examples(self, content: str) -> Dict:
        """Check if minimum code examples present."""
        # Count code blocks
        code_blocks = re.findall(r'```\w+\n.*?```', content, flags=re.DOTALL)

        # Filter out very short blocks (< 3 lines)
        substantial_blocks = [
            block for block in code_blocks
            if len(block.split('\n')) > 3
        ]

        count = len(substantial_blocks)
        target = self.STANDARDS['min_code_examples']
        passed = count >= target

        result = {
            'passed': passed,
            'value': count,
            'target': target,
            'message': f"Code examples: {count} / {target}"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")
            if count < target:
                print(f"   Need {target - count} more code examples")

        return result

    def _check_case_studies(self, content: str) -> Dict:
        """Check if minimum case studies present with metrics."""
        # Look for case study sections
        # Pattern: ## [Company Name] - [Description]
        pattern = r'##+ .+? - .+?\n'
        potential_case_studies = re.findall(pattern, content)

        # Validate each case study has metrics (numbers with units/percentages)
        valid_case_studies = []

        for study_header in potential_case_studies:
            # Get content after this header until next major header
            idx = content.find(study_header)
            if idx == -1:
                continue

            # Find next section
            next_section = content.find('\n## ', idx + len(study_header))
            if next_section == -1:
                section_content = content[idx:]
            else:
                section_content = content[idx:next_section]

            # Check for metrics:
            # - Numbers followed by units (ms, GB, %, etc.)
            # - Dollar amounts
            # - Percentages
            has_metrics = bool(re.search(
                r'(\d+[\.,]?\d*\s*(ms|s|GB|MB|TB|%|\$|users|requests))',
                section_content,
                re.I
            ))

            # Check for Before/After or Results section
            has_results = bool(re.search(
                r'(results?|before|after|improvement|reduction|increase)',
                section_content,
                re.I
            ))

            if has_metrics or has_results:
                valid_case_studies.append(study_header)

        count = len(valid_case_studies)
        target = self.STANDARDS['min_case_studies']
        passed = count >= target

        result = {
            'passed': passed,
            'value': count,
            'target': target,
            'message': f"Case studies: {count} / {target}"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")
            if count < target:
                print(f"   Need {target - count} more case studies with metrics")

        return result

    def _check_troubleshooting(self, content: str) -> Dict:
        """Check if troubleshooting section has minimum issues."""
        # Find troubleshooting section
        trouble_match = re.search(
            r'##+ .*?troubleshooting.*?\n(.*?)(?=\n##+ |\Z)',
            content,
            re.I | re.DOTALL
        )

        if not trouble_match:
            count = 0
        else:
            trouble_section = trouble_match.group(1)

            # Count issues (### headers or numbered issues)
            issue_count = len(re.findall(r'###+ .+?\n', trouble_section))

            if issue_count == 0:
                # Try counting numbered items
                issue_count = len(re.findall(r'^\d+\.\s+\*\*', trouble_section, re.M))

            count = issue_count

        target = self.STANDARDS['min_troubleshooting']
        passed = count >= target

        result = {
            'passed': passed,
            'value': count,
            'target': target,
            'message': f"Troubleshooting issues: {count} / {target}"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")
            if count < target:
                print(f"   Need {target - count} more troubleshooting issues")

        return result

    def _check_sections(self, content: str) -> Dict:
        """Check if all required sections are present."""
        required = self.STANDARDS['required_sections']
        content_lower = content.lower()

        missing = []
        found = []

        for section in required:
            # Look for section as header (## Section Name)
            pattern = r'##+ .*?' + re.escape(section)
            if re.search(pattern, content_lower):
                found.append(section)
            else:
                missing.append(section)

        passed = len(missing) == 0

        result = {
            'passed': passed,
            'value': len(found),
            'target': len(required),
            'missing': missing,
            'message': f"Required sections: {len(found)} / {len(required)}"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")
            if missing:
                print(f"   Missing sections: {', '.join(missing)}")

        return result

    def _check_learning_objectives(self, content: str) -> Dict:
        """Check if minimum learning objectives present."""
        # Look for learning objectives section
        obj_match = re.search(
            r'##+ .*?learning objectives.*?\n(.*?)(?=\n##+ |\Z)',
            content,
            re.I | re.DOTALL
        )

        if not obj_match:
            count = 0
        else:
            obj_section = obj_match.group(1)

            # Count objectives (numbered or bulleted)
            numbered = len(re.findall(r'^\d+\.', obj_section, re.M))
            bulleted = len(re.findall(r'^[-*]', obj_section, re.M))

            count = max(numbered, bulleted)

        target = self.STANDARDS['min_learning_objectives']
        passed = count >= target

        result = {
            'passed': passed,
            'value': count,
            'target': target,
            'message': f"Learning objectives: {count} / {target}"
        }

        if self.verbose:
            status = "✅" if passed else "❌"
            print(f"{status} {result['message']}")
            if count < target:
                print(f"   Need {target - count} more learning objectives")

        return result

    def _print_results(self, results: Dict):
        """Print summary of results."""
        print(f"\n{'='*70}")
        print("COMPLETENESS SUMMARY")
        print(f"{'='*70}\n")

        print(f"File: {results['file']}")

        # Print overall status
        if results['passed']:
            print("\n✅ MODULE PASSED - Meets all quality standards\n")
        else:
            print("\n❌ MODULE NEEDS WORK - Does not meet minimum standards\n")

        # Print individual checks
        for check_name, check_result in results['checks'].items():
            status = "✅" if check_result['passed'] else "❌"
            print(f"{status} {check_result['message']}")

        # Print action items if failed
        if not results['passed']:
            print(f"\n{'='*70}")
            print("ACTION ITEMS")
            print(f"{'='*70}\n")

            for check_name, check_result in results['checks'].items():
                if not check_result['passed']:
                    print(f"• {check_result['message']}")

                    # Add specific guidance
                    if check_name == 'word_count':
                        words_needed = check_result['target'] - check_result['value']
                        print(f"  → Add {words_needed:,} more words")
                        print(f"  → Expand thin sections or add more examples")

                    elif check_name == 'code_examples':
                        needed = check_result['target'] - check_result['value']
                        print(f"  → Add {needed} more code examples")
                        print(f"  → Include examples for each major concept")

                    elif check_name == 'case_studies':
                        needed = check_result['target'] - check_result['value']
                        print(f"  → Add {needed} more case studies")
                        print(f"  → Include company name and specific metrics")

                    elif check_name == 'troubleshooting':
                        needed = check_result['target'] - check_result['value']
                        print(f"  → Add {needed} more troubleshooting issues")
                        print(f"  → Document common errors and solutions")

                    elif check_name == 'required_sections':
                        print(f"  → Add missing sections: {', '.join(check_result['missing'])}")

                    elif check_name == 'learning_objectives':
                        needed = check_result['target'] - check_result['value']
                        print(f"  → Add {needed} more learning objectives")
                        print(f"  → Use action verbs (e.g., 'Implement', 'Configure')")

                    print()

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Check module completeness against quality standards'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to lecture notes file or directory'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file for JSON results'
    )
    parser.add_argument(
        '--directory',
        '-d',
        action='store_true',
        help='Check all files in directory'
    )

    args = parser.parse_args()

    checker = CompletenessChecker(verbose=args.verbose or not args.json)

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    files_to_check = []

    if args.directory or path.is_dir():
        files_to_check = list(path.rglob('*.md'))
        if not args.json:
            print(f"Found {len(files_to_check)} markdown files\n")
    else:
        files_to_check = [path]

    all_results = []
    all_passed = True

    for file in files_to_check:
        results = checker.check_file(file)
        all_results.append(results)

        if not results['passed']:
            all_passed = False

    # Output JSON if requested
    if args.json:
        output_data = {
            'files_checked': len(files_to_check),
            'files_passed': sum(1 for r in all_results if r['passed']),
            'all_passed': all_passed,
            'results': all_results,
            'standards': CompletenessChecker.STANDARDS
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))

    # Print summary if multiple files
    elif len(files_to_check) > 1:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}\n")

        passed_count = sum(1 for r in all_results if r['passed'])
        print(f"Files checked: {len(files_to_check)}")
        print(f"Files passed: {passed_count}")
        print(f"Files need work: {len(files_to_check) - passed_count}")

        if all_passed:
            print("\n✅ ALL MODULES PASSED")
        else:
            print("\n❌ SOME MODULES NEED WORK")

            print("\nFiles needing work:")
            for result in all_results:
                if not result['passed']:
                    print(f"  • {result['file']}")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
