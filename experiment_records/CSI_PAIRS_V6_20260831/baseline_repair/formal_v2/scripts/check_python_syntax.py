from __future__ import annotations

import argparse
from pathlib import Path


def _python_sources(inputs: list[str]) -> list[Path]:
    sources: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_symlink() or not path.exists():
            raise ValueError(f"syntax-check input is missing or unsafe: {path}")
        if path.is_dir():
            sources.update(
                candidate
                for candidate in path.rglob("*.py")
                if "__pycache__" not in candidate.parts and not candidate.is_symlink()
            )
        elif path.is_file() and path.suffix == ".py":
            sources.add(path)
        else:
            raise ValueError(f"syntax-check input is not Python source: {path}")
    if not sources:
        raise ValueError("syntax check found no Python source files")
    return sorted(sources)


def check_python_syntax(inputs: list[str]) -> int:
    sources = _python_sources(inputs)
    for source in sources:
        compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    return len(sources)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile Python source in memory without creating bytecode files."
    )
    parser.add_argument("inputs", nargs="+", help="Python files or directories to check")
    args = parser.parse_args(argv)
    count = check_python_syntax(args.inputs)
    print(f"syntax-ok files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
