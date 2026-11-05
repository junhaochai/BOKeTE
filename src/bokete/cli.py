"""
Command-line interface (CLI) for initialising BoKeTE experiment projects.

Key Functions:
  - init_project: Copies the starter example project into a new target directory.
  - main        : CLI entrypoint parser for the `bokete init` command.
"""

import argparse
from pathlib import Path
import shutil
import sys

from bokete.utils import Colours


def init_project(target_dir_name: str) -> None:
    target_path = Path.cwd() / target_dir_name
    if target_path.exists():
        print(
            f"{Colours.NAVY}[BOKeTE Error]{Colours.RESET} Target directory '{target_dir_name}' already exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Locate starter example directory inside installed package or local dev workspace
    example_dir = Path(__file__).parent / "example"
    if not example_dir.exists():
        example_dir = Path(__file__).resolve().parent.parent.parent / "example"

    if not example_dir.exists():
        print(
            f"{Colours.NAVY}[BOKeTE Error]{Colours.RESET} Starter example directory could not be located.",
            file=sys.stderr,
        )
        sys.exit(1)

    shutil.copytree(example_dir, target_path)

    # Customize project name in pyproject.toml
    project_toml = target_path / "pyproject.toml"
    if project_toml.exists():
        content = project_toml.read_text(encoding="utf-8")
        content = content.replace("bokete-starter", target_dir_name)
        project_toml.write_text(content, encoding="utf-8")

    print(f"{Colours.BOLD_CYAN}[BOKeTE]{Colours.RESET} Initialized new experiment project in '{Colours.BRIGHT_CYAN}{target_dir_name}{Colours.RESET}'!")
    print("\nNext steps:")
    print(f"  cd {target_dir_name}")
    print("  uv run main.py")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bokete", description="BoKeTE CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a new BoKeTE project template")
    init_parser.add_argument("name", help="Name of the project directory to create")

    args = parser.parse_args()
    if args.command == "init":
        init_project(args.name)


if __name__ == "__main__":
    main()
