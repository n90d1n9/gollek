#!/usr/bin/env python3
"""Refresh the active Gollek Jupyter kernelspec in the detected Jupyter data dir."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from doctor import extract_jar_from_argv, find_installed_kernel, load_kernel_spec
from install import (
    JAR_NAME,
    render_kernel_spec,
    resolve_install_base,
    resolve_kernel_jar,
    validate_kernel_jar_freshness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync the active Gollek Jupyter kernelspec")
    parser.add_argument("--kernel-name", default="gollek", help="Kernelspec name to repair.")
    parser.add_argument("--display-name", default="Gollek (Java 25 + AI/ML)")
    parser.add_argument("--jar", help="Use a specific runnable kernel jar.")
    parser.add_argument("--prefix", help="Repair a kernel under a specific Jupyter prefix.")
    parser.add_argument(
        "--allow-stale-jar",
        action="store_true",
        help="Use the jar even when sources are newer than the built artifact.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the repair plan without writing files.")
    return parser.parse_args()


def resolve_target_install_dir(base_dir: Path, args: argparse.Namespace) -> tuple[Path, Path | None]:
    installed_kernel_json = find_installed_kernel(args.kernel_name, args.prefix)
    if installed_kernel_json is not None:
        return installed_kernel_json.parent, installed_kernel_json

    install_base = resolve_install_base(
        SimpleNamespace(
            sys_prefix=False,
            prefix=args.prefix,
            user=True,
        )
    )
    return install_base / args.kernel_name, None


def backup_path(parent: Path, file_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return parent / ".backup" / stamp / file_name


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    kernel_template = base_dir / "src" / "main" / "resources" / "kernel.json"

    try:
        jar_src = resolve_kernel_jar(base_dir, args.jar)
        validate_kernel_jar_freshness(base_dir, jar_src, args.allow_stale_jar)
        install_dir, installed_kernel_json = resolve_target_install_dir(base_dir, args)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"❌ {exc}")
        return 1

    kernel_spec = render_kernel_spec(kernel_template, install_dir, args.display_name)
    target_kernel_json = install_dir / "kernel.json"
    target_kernel_jar = install_dir / JAR_NAME

    existing_kernel_spec = load_kernel_spec(installed_kernel_json) if installed_kernel_json else None
    existing_jar = extract_jar_from_argv(existing_kernel_spec.get("argv", [])) if existing_kernel_spec else None

    print("Gollek Jupyter kernel sync")
    print(f"  source jar:            {jar_src}")
    print(f"  target install dir:    {install_dir}")
    print(f"  target kernel.json:    {target_kernel_json}")
    print(f"  target copied jar:     {target_kernel_jar}")
    print(f"  dry run:               {args.dry_run}")

    if installed_kernel_json:
        print(f"  existing kernelspec:   {installed_kernel_json}")
    else:
        print("  existing kernelspec:   <not found; will create>")

    if existing_jar:
        print(f"  existing argv jar:     {existing_jar}")

    if target_kernel_json.exists():
        print(f"  backup kernel.json:    {backup_path(install_dir, 'kernel.json')}")
    if target_kernel_jar.exists():
        print(f"  backup copied jar:     {backup_path(install_dir, JAR_NAME)}")

    if args.dry_run:
        print("  planned argv:")
        for arg in kernel_spec["argv"]:
            print(f"    - {arg}")
        return 0

    install_dir.mkdir(parents=True, exist_ok=True)
    if target_kernel_json.exists():
        backup_json = backup_path(install_dir, "kernel.json")
        backup_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_kernel_json, backup_json)
    if target_kernel_jar.exists():
        backup_jar = backup_path(install_dir, JAR_NAME)
        backup_jar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_kernel_jar, backup_jar)

    shutil.copy2(jar_src, target_kernel_jar)
    target_kernel_json.write_text(json.dumps(kernel_spec, indent=2))

    print(f"✅ active Gollek kernelspec refreshed: {install_dir}")
    print("   next: python3 doctor.py")
    print("   then: python3 selftest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
