import argparse
from pathlib import Path
import shutil
import sys


def init_project(target_dir_name: str) -> None:
    target_path = Path.cwd() / target_dir_name
    if target_path.exists():
        print(f"[BOKeTE Error] Target directory '{target_dir_name}' already exists.")
        sys.exit(1)

    # Locate template directory inside installed package or local dev workspace
    template_dir = Path(__file__).parent / "template"
    if not template_dir.exists():
        template_dir = Path(__file__).resolve().parent.parent.parent / "example"

    if not template_dir.exists():
        print("[BOKeTE Error] Template directory could not be located.")
        sys.exit(1)

    shutil.copytree(template_dir, target_path)

    # Customize project name in pyproject.toml
    project_toml = target_path / "pyproject.toml"
    if project_toml.exists():
        content = project_toml.read_text(encoding="utf-8")
        content = content.replace("bokete-starter", target_dir_name)
        project_toml.write_text(content, encoding="utf-8")

    print(f"[BOKeTE] Initialized new experiment project in '{target_dir_name}'!")
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
