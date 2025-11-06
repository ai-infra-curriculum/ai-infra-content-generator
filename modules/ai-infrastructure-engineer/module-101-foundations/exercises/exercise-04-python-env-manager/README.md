# Exercise 04: Advanced Python Environment Manager

## Learning Objectives

By completing this exercise, you will:
- Build a tool to manage multiple Python versions
- Automate virtual environment creation and management
- Implement dependency conflict resolution
- Create a production-ready CLI tool
- Practice advanced Python packaging concepts

## Overview

In AI/ML infrastructure, managing multiple Python versions and environments is critical. Different projects require different Python versions, dependencies, and configurations. This exercise builds a comprehensive Python environment management tool.

## Prerequisites

- Python 3.11+ installed
- Understanding of virtual environments (venv, virtualenv)
- Basic knowledge of dependency management (pip, requirements.txt)
- Familiarity with subprocess and CLI development

## Problem Statement

You need to build `pyenvman`, a Python environment manager that can:

1. **Detect and manage multiple Python versions**
   - List all Python versions on the system
   - Download and install specific Python versions
   - Set default Python version per project

2. **Virtual environment automation**
   - Create venvs with specific Python versions
   - Activate/deactivate environments
   - Clone existing environments
   - Export environment specifications

3. **Dependency conflict resolution**
   - Detect conflicting package versions
   - Suggest compatible version combinations
   - Validate requirements.txt files
   - Generate lock files for reproducibility

4. **Project scaffolding**
   - Initialize new Python projects with templates
   - Set up pre-commit hooks
   - Configure testing and linting
   - Generate CI/CD configurations

## Requirements

### Functional Requirements

#### FR1: Python Version Management
- List all Python versions installed on system
- Download and install Python versions (3.8 through 3.12+)
- Detect Python versions from common locations
- Set per-project Python version
- Support system Python, pyenv, conda, asdf

#### FR2: Virtual Environment Management
-Create virtual environments with specified Python version
- List all virtual environments
- Activate environments (generate activation scripts)
- Delete environments with confirmation
- Clone environments (copy all packages)
- Export environment to requirements.txt/environment.yml
- Import environment from specification

#### FR3: Dependency Management
- Parse requirements.txt and detect conflicts
- Check package compatibility with Python version
- Suggest compatible version ranges
- Generate pip-compile style lock files
- Update dependencies to latest compatible versions
- Security audit dependencies (check for known vulnerabilities)

#### FR4: Project Initialization
- Create new project from templates (basic, FastAPI, ML, etc.)
- Set up git repository with .gitignore
- Install pre-commit hooks
- Generate README.md template
- Create pyproject.toml with modern configuration
- Set up pytest, mypy, black, ruff configurations

### Non-Functional Requirements

#### NFR1: Performance
- List operations complete in < 1 second
- Installation operations show progress bars
- Support concurrent package downloads

#### NFR2: Usability
- Clear, colorized CLI output
- Progress indicators for long operations
- Helpful error messages with suggestions
- Interactive prompts with sensible defaults
- Support for --quiet and --verbose modes

#### NFR3: Reliability
- Atomic operations (rollback on failure)
- Verify package integrity after download
- Backup before destructive operations
- Handle network failures gracefully

#### NFR4: Compatibility
- Support Linux, macOS, Windows
- Work with system Python, pyenv, conda
- Compatible with pip, poetry, pipenv

## Implementation Tasks

### Task 1: Python Version Detection (4-5 hours)

Build a system to detect all Python installations:

```python
# src/python_detector.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import subprocess

@dataclass
class PythonVersion:
    """Represents a Python installation"""
    version: str  # e.g., "3.11.5"
    path: Path
    manager: str  # "system", "pyenv", "conda", "asdf"
    is_virtual: bool = False

class PythonDetector:
    """Detect Python installations on the system"""

    def detect_all(self) -> List[PythonVersion]:
        """
        Detect all Python installations

        TODO: Implement detection logic:
        1. Search common system paths (/usr/bin, /usr/local/bin)
        2. Check pyenv installations (~/.pyenv/versions)
        3. Check conda environments
        4. Check asdf installations
        5. Deduplicate by version and path
        6. Sort by version (newest first)
        """
        pass

    def detect_system_python(self) -> List[PythonVersion]:
        """TODO: Find system Python installations"""
        pass

    def detect_pyenv(self) -> List[PythonVersion]:
        """TODO: Find pyenv Python installations"""
        pass

    def detect_conda(self) -> List[PythonVersion]:
        """TODO: Find conda Python environments"""
        pass

    def get_version_info(self, python_path: Path) -> Optional[PythonVersion]:
        """
        TODO: Get Python version from executable

        Hint: Run `python --version` and parse output
        """
        pass
```

**Acceptance Criteria**:
- [ ] Detects system Python on Linux/macOS/Windows
- [ ] Detects pyenv installations if available
- [ ] Detects conda environments if available
- [ ] Returns sorted list (newest first)
- [ ] Handles missing Python gracefully
- [ ] Unit tests with 80%+ coverage

---

### Task 2: Virtual Environment Manager (5-6 hours)

Implement virtual environment creation and management:

```python
# src/venv_manager.py

from pathlib import Path
from typing import List, Optional
import venv
import subprocess

class VenvManager:
    """Manage virtual environments"""

    def __init__(self, venv_dir: Path = Path.home() / ".pyenvman" / "venvs"):
        self.venv_dir = venv_dir
        self.venv_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        python_version: str,
        requirements: Optional[Path] = None,
        system_site_packages: bool = False
    ) -> Path:
        """
        Create a new virtual environment

        TODO:
        1. Find Python executable for specified version
        2. Create venv using venv module or virtualenv
        3. Upgrade pip, setuptools, wheel
        4. Install requirements if provided
        5. Create activation helper script
        6. Return path to created venv
        """
        pass

    def list_venvs(self) -> List[dict]:
        """
        TODO: List all virtual environments

        Return format:
        [
            {
                "name": "myproject",
                "path": Path("/home/user/.pyenvman/venvs/myproject"),
                "python_version": "3.11.5",
                "created": datetime(...),
                "size": "150 MB"
            },
            ...
        ]
        """
        pass

    def delete(self, name: str, confirm: bool = True) -> bool:
        """TODO: Delete virtual environment with confirmation"""
        pass

    def clone(self, source: str, target: str) -> Path:
        """
        TODO: Clone existing environment

        Steps:
        1. Export source env to requirements.txt
        2. Create new env with same Python version
        3. Install all packages from source
        """
        pass

    def export_requirements(self, name: str, output: Path) -> None:
        """TODO: Export environment to requirements.txt using pip freeze"""
        pass

    def activate_script(self, name: str) -> str:
        """
        TODO: Generate activation command

        Returns: Shell command to activate environment
        Example: "source /path/to/venv/bin/activate"
        """
        pass
```

**Acceptance Criteria**:
- [ ] Creates venvs with specific Python version
- [ ] Lists all venvs with metadata
- [ ] Deletes venvs with confirmation
- [ ] Clones venvs preserving packages
- [ ] Exports to requirements.txt
- [ ] Generates activation scripts
- [ ] Cross-platform (Linux/macOS/Windows)
- [ ] Unit tests with 80%+ coverage

---

### Task 3: Dependency Conflict Resolver (6-7 hours)

Build a tool to detect and resolve dependency conflicts:

```python
# src/dependency_resolver.py

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from packaging.requirements import Requirement
from packaging.version import Version
import requests

@dataclass
class PackageInfo:
    """Package metadata"""
    name: str
    version: str
    dependencies: List[str]
    python_requires: Optional[str] = None

@dataclass
class Conflict:
    """Dependency conflict"""
    package: str
    required_by: List[str]
    conflicting_versions: List[str]
    suggestion: Optional[str] = None

class DependencyResolver:
    """Resolve dependency conflicts"""

    def __init__(self):
        self.pypi_url = "https://pypi.org/pypi"

    def parse_requirements(self, requirements_file: Path) -> List[Requirement]:
        """
        TODO: Parse requirements.txt

        Handle:
        - Simple packages: "requests==2.31.0"
        - Version ranges: "numpy>=1.20.0,<2.0.0"
        - Git URLs: "git+https://..."
        - Local paths: "-e ./mypackage"
        - Comments and blank lines
        """
        pass

    def detect_conflicts(
        self,
        requirements: List[Requirement],
        python_version: str
    ) -> List[Conflict]:
        """
        TODO: Detect dependency conflicts

        Algorithm:
        1. Build dependency graph for all packages
        2. Check for conflicting version requirements
        3. Verify Python version compatibility
        4. Generate conflict report with suggestions
        """
        pass

    def get_package_info(self, package: str, version: Optional[str] = None) -> PackageInfo:
        """
        TODO: Fetch package info from PyPI

        Endpoint: GET https://pypi.org/pypi/{package}/json
        or: GET https://pypi.org/pypi/{package}/{version}/json
        """
        pass

    def suggest_resolution(self, conflict: Conflict) -> str:
        """
        TODO: Suggest resolution for conflict

        Example output:
        "Try using numpy>=1.21.0,<1.24.0 to satisfy both
         scikit-learn (requires numpy<1.24) and
         pandas (requires numpy>=1.21)"
        """
        pass

    def check_security(self, requirements: List[Requirement]) -> List[dict]:
        """
        TODO: Check for security vulnerabilities

        Use PyPI's safety database or integrate with pip-audit
        Return list of vulnerable packages with severity
        """
        pass

    def generate_lockfile(
        self,
        requirements: List[Requirement],
        python_version: str,
        output: Path
    ) -> None:
        """
        TODO: Generate pip-compile style lockfile

        Output format:
        # This file is autogenerated by pyenvman
        # Requirements:
        #   requests>=2.28.0
        # Generated for Python 3.11

        certifi==2023.7.22
            # via requests
        charset-normalizer==3.2.0
            # via requests
        ...
        """
        pass
```

**Acceptance Criteria**:
- [ ] Parses complex requirements.txt files
- [ ] Detects version conflicts
- [ ] Checks Python version compatibility
- [ ] Fetches package info from PyPI
- [ ] Suggests conflict resolutions
- [ ] Checks for security vulnerabilities
- [ ] Generates lockfiles with dependency tree
- [ ] Handles network errors gracefully
- [ ] Unit tests with mocked PyPI responses

---

### Task 4: Project Scaffolding (4-5 hours)

Create project initialization with templates:

```python
# src/project_init.py

from pathlib import Path
from typing import Optional
import subprocess

class ProjectInitializer:
    """Initialize new Python projects"""

    TEMPLATES = {
        "basic": "Basic Python project with tests",
        "fastapi": "FastAPI web service",
        "ml": "Machine learning project with MLflow",
        "cli": "CLI application with click/typer",
    }

    def init_project(
        self,
        path: Path,
        template: str,
        python_version: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """
        TODO: Initialize new project

        Steps:
        1. Create project directory structure
        2. Generate README.md
        3. Create pyproject.toml with modern config
        4. Set up git repository
        5. Add .gitignore
        6. Create virtual environment
        7. Install dev dependencies
        8. Set up pre-commit hooks
        9. Generate initial files from template
        """
        pass

    def create_structure(self, path: Path, template: str) -> None:
        """
        TODO: Create project directory structure

        Basic template:
        project/
        ├── src/
        │   └── __init__.py
        ├── tests/
        │   └── __init__.py
        ├── README.md
        ├── pyproject.toml
        ├── requirements.txt
        ├── requirements-dev.txt
        └── .gitignore

        FastAPI template:
        project/
        ├── app/
        │   ├── __init__.py
        │   ├── main.py
        │   ├── api/
        │   │   └── routes.py
        │   └── models/
        ├── tests/
        ├── Dockerfile
        ├── docker-compose.yml
        └── ...
        """
        pass

    def generate_pyproject_toml(
        self,
        path: Path,
        name: str,
        description: str,
        python_version: str
    ) -> None:
        """
        TODO: Generate modern pyproject.toml

        Include sections for:
        - [project] metadata
        - [build-system] (setuptools, hatchling, or poetry)
        - [tool.pytest.ini_options]
        - [tool.black]
        - [tool.ruff]
        - [tool.mypy]
        - [tool.coverage.run]
        """
        pass

    def setup_git(self, path: Path) -> None:
        """TODO: Initialize git repo with .gitignore"""
        pass

    def setup_precommit(self, path: Path) -> None:
        """
        TODO: Set up pre-commit hooks

        Include hooks for:
        - black (code formatting)
        - ruff (linting)
        - mypy (type checking)
        - pytest (tests must pass)
        """
        pass

    def generate_readme(
        self,
        path: Path,
        name: str,
        description: str
    ) -> None:
        """TODO: Generate README.md template"""
        pass
```

**Acceptance Criteria**:
- [ ] Creates project structure from templates
- [ ] Generates pyproject.toml with modern configuration
- [ ] Initializes git repository
- [ ] Sets up pre-commit hooks
- [ ] Creates comprehensive README
- [ ] Supports multiple templates (basic, FastAPI, ML, CLI)
- [ ] Creates virtual environment
- [ ] Installs dev dependencies
- [ ] Unit tests with 80%+ coverage

---

### Task 5: CLI Application (3-4 hours)

Build the command-line interface:

```python
# src/cli.py

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path

console = Console()

@click.group()
@click.version_option()
def cli():
    """
    pyenvman - Advanced Python Environment Manager

    Manage Python versions, virtual environments, and dependencies
    """
    pass

@cli.group()
def python():
    """Manage Python versions"""
    pass

@python.command("list")
def python_list():
    """
    List all Python versions

    TODO:
    1. Use PythonDetector to find all Python installations
    2. Display in rich table with columns:
       - Version
       - Path
       - Manager (system/pyenv/conda)
       - Default (mark current default)
    3. Highlight recommended versions (3.11+)
    """
    pass

@python.command("install")
@click.argument("version")
def python_install(version: str):
    """
    Install Python version

    TODO:
    1. Download Python from python.org or use pyenv
    2. Show progress bar during download
    3. Install to ~/.pyenvman/pythons/{version}
    4. Verify installation
    """
    pass

@cli.group()
def venv():
    """Manage virtual environments"""
    pass

@venv.command("create")
@click.argument("name")
@click.option("--python", "-p", help="Python version")
@click.option("--requirements", "-r", type=click.Path(), help="Install from requirements.txt")
def venv_create(name: str, python: str, requirements: str):
    """
    Create virtual environment

    TODO:
    1. Use VenvManager to create venv
    2. Show progress during creation
    3. Display activation command
    """
    pass

@venv.command("list")
def venv_list():
    """
    List virtual environments

    TODO:
    1. Use VenvManager to list venvs
    2. Display in rich table
    3. Show size, Python version, created date
    """
    pass

@venv.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def venv_delete(name: str, yes: bool):
    """Delete virtual environment"""
    pass

@cli.group()
def deps():
    """Manage dependencies"""
    pass

@deps.command("check")
@click.argument("requirements", type=click.Path(exists=True))
@click.option("--python", "-p", help="Python version")
def deps_check(requirements: str, python: str):
    """
    Check for dependency conflicts

    TODO:
    1. Use DependencyResolver to parse requirements
    2. Detect conflicts
    3. Display conflicts in red
    4. Show suggestions in green
    5. Exit with code 1 if conflicts found
    """
    pass

@deps.command("audit")
@click.argument("requirements", type=click.Path(exists=True))
def deps_audit(requirements: str):
    """
    Security audit dependencies

    TODO:
    1. Check for known vulnerabilities
    2. Display vulnerability report
    3. Show severity levels (critical/high/medium/low)
    4. Suggest safe versions
    """
    pass

@deps.command("lock")
@click.argument("requirements", type=click.Path(exists=True))
@click.option("--output", "-o", default="requirements.lock")
@click.option("--python", "-p", required=True)
def deps_lock(requirements: str, output: str, python: str):
    """Generate lockfile"""
    pass

@cli.group()
def project():
    """Initialize projects"""
    pass

@project.command("init")
@click.argument("path", type=click.Path())
@click.option("--template", "-t", type=click.Choice(["basic", "fastapi", "ml", "cli"]), default="basic")
@click.option("--name", "-n", help="Project name")
@click.option("--python", "-p", default="3.11", help="Python version")
def project_init(path: str, template: str, name: str, python: str):
    """
    Initialize new project

    TODO:
    1. Validate inputs
    2. Use ProjectInitializer to create project
    3. Show progress for each step
    4. Display next steps at completion
    """
    pass

if __name__ == "__main__":
    cli()
```

**Acceptance Criteria**:
- [ ] All commands work correctly
- [ ] Colorized output with rich
- [ ] Progress bars for long operations
- [ ] Interactive confirmations
- [ ] Clear error messages
- [ ] --help text for all commands
- [ ] Exit codes (0 success, 1 error)

---

## Testing Requirements

### Unit Tests

Create comprehensive unit tests in `tests/`:

```python
# tests/test_python_detector.py
def test_detect_system_python():
    """Test system Python detection"""
    pass

def test_detect_pyenv():
    """Test pyenv detection"""
    pass

# tests/test_venv_manager.py
def test_create_venv():
    """Test venv creation"""
    pass

def test_list_venvs():
    """Test venv listing"""
    pass

def test_clone_venv():
    """Test venv cloning"""
    pass

# tests/test_dependency_resolver.py
def test_parse_requirements():
    """Test requirements.txt parsing"""
    pass

def test_detect_conflicts():
    """Test conflict detection"""
    pass

@pytest.mark.parametrize("package,expected", [
    ("requests", "2.31.0"),
    ("numpy", "1.25.2"),
])
def test_get_latest_version(package, expected):
    """Test fetching latest versions"""
    pass

# tests/test_cli.py
def test_cli_python_list(cli_runner):
    """Test 'python list' command"""
    result = cli_runner.invoke(cli, ["python", "list"])
    assert result.exit_code == 0

def test_cli_venv_create(cli_runner, tmp_path):
    """Test 'venv create' command"""
    pass
```

**Test Coverage Target**: 80%+

### Integration Tests

```python
# tests/integration/test_full_workflow.py

def test_full_workflow(tmp_path):
    """
    Test complete workflow:
    1. Detect Python versions
    2. Create venv
    3. Install packages
    4. Check for conflicts
    5. Generate lockfile
    6. Clone venv
    """
    pass
```

---

## Deliverables

1. **Source Code** (`src/`)
   - `python_detector.py` - Python version detection
   - `venv_manager.py` - Virtual environment management
   - `dependency_resolver.py` - Dependency conflict resolution
   - `project_init.py` - Project scaffolding
   - `cli.py` - Command-line interface
   - `__init__.py` - Package initialization

2. **Tests** (`tests/`)
   - Unit tests for all modules (80%+ coverage)
   - Integration tests
   - Fixtures and mocks

3. **Documentation**
   - README.md with usage examples
   - API documentation (docstrings)
   - CONTRIBUTING.md

4. **Configuration Files**
   - `pyproject.toml` - Project configuration
   - `requirements.txt` - Production dependencies
   - `requirements-dev.txt` - Development dependencies
   - `.pre-commit-config.yaml` - Pre-commit hooks

5. **CI/CD**
   - `.github/workflows/test.yml` - Run tests on push
   - `.github/workflows/lint.yml` - Code quality checks

---

## Bonus Challenges

1. **Package to PyPI**
   - Create distribution package
   - Publish to test.pypi.org
   - Document installation process

2. **Shell Completion**
   - Add bash/zsh completion scripts
   - Support tab-completion for all commands

3. **Config File Support**
   - Support `.pyenvman.toml` for project settings
   - Automatically activate venv when entering project directory

4. **GUI/TUI Interface**
   - Build terminal UI with textual
   - Interactive environment management

5. **Docker Integration**
   - Generate Dockerfiles for projects
   - Use detected Python version in Docker images

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Functionality** | 30% | All features work correctly |
| **Code Quality** | 25% | Clean, well-structured, typed code |
| **Testing** | 20% | Comprehensive tests, 80%+ coverage |
| **Documentation** | 15% | Clear README, docstrings, examples |
| **User Experience** | 10% | Intuitive CLI, helpful output |

**Passing Score**: 70%
**Excellence**: 90%+

---

## Resources

### Documentation
- [venv module](https://docs.python.org/3/library/venv.html)
- [packaging library](https://packaging.pypa.io/en/stable/)
- [PyPI JSON API](https://warehouse.pypa.io/api-reference/json.html)
- [Click documentation](https://click.palletsprojects.com/)
- [Rich documentation](https://rich.readthedocs.io/)

### Example Projects
- [pyenv](https://github.com/pyenv/pyenv) - Python version management
- [pipx](https://github.com/pypa/pipx) - Install Python applications
- [poetry](https://github.com/python-poetry/poetry) - Dependency management

### Tools
- pip-compile (pip-tools)
- pip-audit (security scanning)
- safety (dependency security)

---

## Estimated Time

- **Task 1** (Python Detection): 4-5 hours
- **Task 2** (Venv Management): 5-6 hours
- **Task 3** (Dependency Resolution): 6-7 hours
- **Task 4** (Project Init): 4-5 hours
- **Task 5** (CLI): 3-4 hours
- **Testing**: 6-8 hours
- **Documentation**: 2-3 hours

**Total**: 30-38 hours

---

## Submission

1. Push code to GitHub repository
2. Ensure all tests pass in CI
3. Include comprehensive README with:
   - Installation instructions
   - Usage examples
   - Screenshots/demos
4. Record a 5-minute demo video (optional but recommended)

---

**Good luck! This is a portfolio-worthy project that demonstrates production Python development skills.**
