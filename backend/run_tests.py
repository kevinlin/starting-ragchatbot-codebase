#!/usr/bin/env python3
"""
Test runner script for the RAG chatbot backend.
Provides convenient commands for running different types of tests.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}")
    print("=" * len(description))

    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for RAG chatbot backend")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function (requires --file)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests"
    )

    args = parser.parse_args()

    # Change to backend directory
    backend_dir = Path(__file__).parent
    print(f"Running tests from: {backend_dir}")

    # Install dependencies if requested
    if args.install_deps:
        if not run_command("uv sync --dev", "Installing test dependencies"):
            return 1

    # Build base pytest command
    base_cmd = ["uv", "run", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.coverage:
        base_cmd.extend(["--cov=backend", "--cov-report=term-missing", "--cov-report=html"])

    # Handle specific file/function
    if args.file:
        test_path = f"tests/{args.file}" if not args.file.startswith("tests/") else args.file
        if args.function:
            test_path = f"{test_path}::{args.function}"
        base_cmd.append(test_path)
    else:
        # Handle test type filtering
        if args.type == "unit":
            base_cmd.extend(["-m", "unit"])
        elif args.type == "integration":
            base_cmd.extend(["-m", "integration"])
        # "all" runs everything (no marker filter)

    # Run the tests
    command = " ".join(base_cmd)
    success = run_command(command, f"Running {args.type} tests")

    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


def run_specific_component(component_name):
    """Run tests for a specific component."""
    test_file = f"tests/test_{component_name}.py"
    command = f"uv run pytest {test_file} -v"
    return run_command(command, f"Running tests for {component_name}")


def run_quick_tests():
    """Run a quick subset of tests for development."""
    command = "uv run pytest tests/ -x --disable-warnings"
    return run_command(command, "Running quick tests (fail fast)")


def run_coverage_report():
    """Generate a detailed coverage report."""
    commands = [
        ("uv run pytest --cov=backend --cov-report=html --cov-report=term", "Running tests with coverage"),
        ("echo 'Coverage report generated at htmlcov/index.html'", "Coverage report location")
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True


if __name__ == "__main__":
    # Allow running with shortcuts
    if len(sys.argv) > 1 and sys.argv[1] in ["quick", "coverage"]:
        if sys.argv[1] == "quick":
            sys.exit(0 if run_quick_tests() else 1)
        elif sys.argv[1] == "coverage":
            sys.exit(0 if run_coverage_report() else 1)

    # Check if running specific component
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        component = sys.argv[1]
        if component in ["session", "document", "vector", "search", "ai"]:
            component_map = {
                "session": "session_manager",
                "document": "document_processor",
                "vector": "vector_store",
                "search": "search_tools",
                "ai": "ai_generator"
            }
            sys.exit(0 if run_specific_component(component_map[component]) else 1)

    sys.exit(main())