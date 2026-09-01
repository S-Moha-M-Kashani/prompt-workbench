from __future__ import annotations

import argparse

from prompt_workbench import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prompt-workbench")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.parse_args(argv)
    print("Hello from prompt_workbench")


if __name__ == "__main__":
    main()
