#!/usr/bin/env python3
"""Install the Gollek standalone Jupyter kernel."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

JAR_NAME = "gollek-kernel.jar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Gollek Jupyter Kernel")
    parser.add_argument("--user", action="store_true")
    parser.add_argument("--sys-prefix", action="store_true")
    parser.add_argument("--prefix")
    parser.add_argument("--kernel-name", default="gollek")
    parser.add_argument("--display-name", default="Gollek (Java 25 + AI/ML)")
    parser.add_argument("--jar", help="Use a specific runnable kernel jar.")
    parser.add_argument(
        "--allow-stale-jar",
        action="store_true",
        help="Use the jar even when sources are newer than the built artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the install plan without writing files.",
    )
    return parser.parse_args()


def resolve_kernel_jar(base_dir: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            base_dir / "build" / "libs" / JAR_NAME,
            base_dir / "target" / JAR_NAME,
            base_dir / "target" / "original-gollek-kernel.jar",
        ]
    )
    build_libs = base_dir / "build" / "libs"
    if build_libs.exists():
        candidates.extend(sorted(build_libs.glob("*kernel*.jar")))

    existing = [candidate.resolve() for candidate in candidates if candidate.exists() and candidate.is_file()]
    if existing:
        existing.sort(key=lambda path: (path.stat().st_mtime, str(path)))
        return existing[-1]

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Gollek kernel jar not found.\n"
        f"Searched:\n{searched}\n"
        "Build one first with either:\n"
        "  - mvn -f gollek/integration/jupyter/gollek-jupyter-kernel/pom.xml -DskipTests package\n"
        "  - ./gradlew :integration:jupyter:gollek-jupyter-kernel:build"
    )


def resolve_install_base(args: argparse.Namespace) -> Path:
    def detect_jupyter_user_data_dir() -> Path | None:
        jupyter_bin = shutil.which("jupyter")
        if not jupyter_bin:
            return None
        try:
            result = subprocess.run(
                [jupyter_bin, "--data-dir"],
                capture_output=True,
                text=True,
                check=True,
            )
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            return None

    if args.sys_prefix:
        return Path(sys.prefix) / "share" / "jupyter" / "kernels"
    if args.prefix:
        return Path(args.prefix).expanduser() / "share" / "jupyter" / "kernels"
    if args.user:
        detected = detect_jupyter_user_data_dir()
        if detected is not None:
            return detected / "kernels"
        if platform.system() == "Darwin":
            return Path.home() / "Library" / "Jupyter" / "kernels"
        return Path.home() / ".local" / "share" / "jupyter" / "kernels"

    jupyter_bin = shutil.which("jupyter")
    if not jupyter_bin:
        raise RuntimeError(
            "jupyter executable not found in PATH. Use --user/--prefix/--sys-prefix or install Jupyter first."
        )

    result = subprocess.run(
        [jupyter_bin, "--data-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()) / "kernels"


def render_kernel_spec(template_path: Path, install_dir: Path, display_name: str) -> dict:
    config = json.loads(template_path.read_text())
    config["display_name"] = display_name
    config["argv"] = [arg.replace("{install_dir}", str(install_dir)) for arg in config["argv"]]
    return config


def latest_source_mtime(base_dir: Path) -> tuple[float, Path]:
    candidates = [
        base_dir / "src" / "main" / "java",
        base_dir / "src" / "main" / "resources",
        base_dir / "pom.xml",
        base_dir / "build.gradle.kts",
    ]
    latest_path = base_dir
    latest_mtime = 0.0
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_file():
            entries = [candidate]
        else:
            entries = [path for path in candidate.rglob("*") if path.is_file()]
        for entry in entries:
            entry_mtime = entry.stat().st_mtime
            if entry_mtime > latest_mtime:
                latest_mtime = entry_mtime
                latest_path = entry
    return latest_mtime, latest_path


def validate_kernel_jar_freshness(base_dir: Path, jar_path: Path, allow_stale: bool) -> None:
    latest_mtime, latest_path = latest_source_mtime(base_dir)
    jar_mtime = jar_path.stat().st_mtime
    if jar_mtime >= latest_mtime or allow_stale:
        return

    jar_ts = datetime.fromtimestamp(jar_mtime).isoformat(sep=" ", timespec="seconds")
    src_ts = datetime.fromtimestamp(latest_mtime).isoformat(sep=" ", timespec="seconds")
    raise RuntimeError(
        "Kernel jar is stale relative to source files.\n"
        f"  jar:    {jar_path} ({jar_ts})\n"
        f"  source: {latest_path} ({src_ts})\n"
        "Rebuild before install/self-test with either:\n"
        "  - mvn -f gollek/integration/jupyter/gollek-jupyter-kernel/pom.xml -DskipTests package\n"
        "  - ./gradlew :integration:jupyter:gollek-jupyter-kernel:build\n"
        "If you intentionally want the old jar, rerun with --allow-stale-jar."
    )


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        jar_src = resolve_kernel_jar(base_dir, args.jar)
        validate_kernel_jar_freshness(base_dir, jar_src, args.allow_stale_jar)
        install_base = resolve_install_base(args)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"❌ {exc}")
        return 1

    install_dir = install_base / args.kernel_name
    kernel_template = base_dir / "src" / "main" / "resources" / "kernel.json"
    kernel_spec = render_kernel_spec(kernel_template, install_dir, args.display_name)

    if args.dry_run:
        print("Gollek Jupyter kernel dry run")
        print(f"  kernel jar:     {jar_src}")
        print(f"  install dir:    {install_dir}")
        print(f"  display name:   {args.display_name}")
        print(f"  kernel name:    {args.kernel_name}")
        print(f"  stale jar ok:   {args.allow_stale_jar}")
        print("  argv:")
        for arg in kernel_spec["argv"]:
            print(f"    - {arg}")
        return 0

    install_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jar_src, install_dir / JAR_NAME)
    (install_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2))

    print(f"✅ Gollek kernel installed: {install_dir}")
    print("   Verify: jupyter kernelspec list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
