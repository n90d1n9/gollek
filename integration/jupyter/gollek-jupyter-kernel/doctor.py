#!/usr/bin/env python3
"""Inspect the installed Gollek Jupyter kernelspec and jar wiring."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from install import latest_source_mtime, resolve_kernel_jar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the installed Gollek Jupyter kernelspec")
    parser.add_argument("--kernel-name", default="gollek", help="Kernelspec name to inspect.")
    parser.add_argument("--prefix", help="Inspect a specific Jupyter prefix instead of auto-discovery.")
    parser.add_argument("--verbose", action="store_true", help="Print the resolved kernelspec JSON.")
    return parser.parse_args()


def discover_kernel_dirs(prefix: str | None) -> list[Path]:
    if prefix:
        return [Path(prefix).expanduser() / "share" / "jupyter" / "kernels"]

    directories: list[Path] = []
    jupyter_bin = shutil.which("jupyter")
    if jupyter_bin:
        try:
            result = subprocess.run(
                [jupyter_bin, "--paths", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            for data_dir in payload.get("data", []):
                directories.append(Path(data_dir) / "kernels")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

    directories.extend(
        [
            Path.home() / ".local" / "share" / "jupyter" / "kernels",
            Path(sys.prefix) / "share" / "jupyter" / "kernels",
            Path("/usr/local/share/jupyter/kernels"),
            Path("/usr/share/jupyter/kernels"),
        ]
    )

    unique_directories: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique_directories.append(directory)
    return unique_directories


def find_installed_kernel(kernel_name: str, prefix: str | None) -> Path | None:
    for kernels_dir in discover_kernel_dirs(prefix):
        candidate = kernels_dir / kernel_name / "kernel.json"
        if candidate.exists():
            return candidate
    return None


def load_kernel_spec(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_jar_from_argv(argv: list[str]) -> Path | None:
    if "-jar" not in argv:
        return None
    jar_index = argv.index("-jar") + 1
    if jar_index >= len(argv):
        return None
    return Path(argv[jar_index]).expanduser()


def format_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    installed_kernel_json = find_installed_kernel(args.kernel_name, args.prefix)

    print("Gollek Jupyter kernel doctor")
    print(f"  module dir:           {base_dir}")
    print(f"  kernel name:          {args.kernel_name}")

    try:
        freshest_local_jar = resolve_kernel_jar(base_dir, None)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    latest_src_mtime, latest_src_path = latest_source_mtime(base_dir)
    print(f"  freshest local jar:   {freshest_local_jar}")
    print(f"  freshest local mtime: {format_timestamp(freshest_local_jar)}")
    print(f"  latest source file:   {latest_src_path}")
    print(
        "  latest source mtime:  "
        f"{datetime.fromtimestamp(latest_src_mtime).isoformat(sep=' ', timespec='seconds')}"
    )

    if installed_kernel_json is None:
        print("❌ installed kernelspec: not found")
        print("   hint: run `python3 install.py --user` after building a fresh jar.")
        return 1

    spec = load_kernel_spec(installed_kernel_json)
    installed_dir = installed_kernel_json.parent
    argv = spec.get("argv", [])
    installed_jar = extract_jar_from_argv(argv)

    print(f"  installed kernelspec: {installed_kernel_json}")
    print(f"  installed display:    {spec.get('display_name', '<missing>')}")

    if args.verbose:
        print("  installed argv:")
        for arg in argv:
            print(f"    - {arg}")

    if installed_jar is None:
        print("❌ kernelspec argv does not contain a valid `-jar` entry")
        return 1

    print(f"  installed jar argv:   {installed_jar}")

    status = 0

    if not installed_jar.exists():
        print("❌ installed jar path does not exist")
        print("   hint: reinstall with `python3 install.py --user`.")
        return 1

    print(f"  installed jar mtime:  {format_timestamp(installed_jar)}")

    if installed_jar.resolve() != freshest_local_jar.resolve():
        print("⚠️  installed jar differs from the freshest local build output")
        print("   note: Jupyter may still be launching an older copied artifact.")
        status = 1
    else:
        print("✅ installed jar matches the freshest local build path")

    if installed_jar.stat().st_mtime < latest_src_mtime:
        print("⚠️  installed jar is stale relative to current source files")
        print("   hint: rebuild and reinstall the standalone kernel.")
        status = 1
    else:
        print("✅ installed jar is fresh relative to current source files")

    expected_installed_jar = installed_dir / "gollek-kernel.jar"
    if installed_jar.resolve() != expected_installed_jar.resolve():
        print("⚠️  kernelspec does not point at the copied jar inside its install directory")
        print(f"   expected installed jar: {expected_installed_jar}")
        status = 1
    else:
        print("✅ kernelspec points at the copied install-directory jar")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
