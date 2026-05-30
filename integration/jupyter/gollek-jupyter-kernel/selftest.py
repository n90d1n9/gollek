#!/usr/bin/env python3
"""Smoke-test the standalone Gollek Jupyter kernel startup path."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from install import (
    resolve_kernel_jar,
    render_kernel_spec,
    validate_kernel_jar_freshness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test Gollek Jupyter kernel startup")
    parser.add_argument("--jar", help="Use a specific runnable kernel jar.")
    parser.add_argument("--java", default="java", help="Java executable to launch.")
    parser.add_argument("--timeout-secs", type=float, default=3.0, help="Seconds the kernel must stay alive.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for bind-collision startup failures.")
    parser.add_argument(
        "--allow-stale-jar",
        action="store_true",
        help="Use the jar even when sources are newer than the built artifact.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print captured stdout/stderr on success too.")
    return parser.parse_args()


def random_port() -> int:
    return 20000 + secrets.randbelow(30000)


def build_connection_payload() -> dict:
    ports = set()

    def unique_port() -> int:
        while True:
            candidate = random_port()
            if candidate not in ports:
                ports.add(candidate)
                return candidate

    return {
        "shell_port": unique_port(),
        "iopub_port": unique_port(),
        "stdin_port": unique_port(),
        "control_port": unique_port(),
        "hb_port": unique_port(),
        "ip": "127.0.0.1",
        "key": secrets.token_hex(16),
        "transport": "tcp",
        "signature_scheme": "hmac-sha256",
        "kernel_name": "gollek-selftest",
    }


def summarize_connection_ports(connection_payload: dict) -> str:
    return ", ".join(
        [
            f"shell={connection_payload['shell_port']}",
            f"iopub={connection_payload['iopub_port']}",
            f"stdin={connection_payload['stdin_port']}",
            f"control={connection_payload['control_port']}",
            f"hb={connection_payload['hb_port']}",
        ]
    )


def build_kernel_command(base_dir: Path, jar_path: Path, java_bin: str) -> list[str]:
    spec = render_kernel_spec(
        base_dir / "src" / "main" / "resources" / "kernel.json",
        jar_path.parent,
        "Gollek (Java 25 + AI/ML)",
    )
    argv = list(spec["argv"])
    argv[0] = java_bin
    jar_index = argv.index("-jar") + 1
    argv[jar_index] = str(jar_path)
    return argv


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def is_bind_collision(stderr: str) -> bool:
    return "Address already in use" in stderr or "Errno 48" in stderr


def probe_local_tcp_bind(attempts: int = 3) -> list[dict[str, str | int | bool]]:
    probe_results = []
    for _ in range(attempts):
        port = random_port()
        result = {"port": port, "ok": False, "error": ""}
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            result["ok"] = True
        except OSError as exc:
            result["error"] = f"{exc.__class__.__name__}: {exc}"
        finally:
            sock.close()
        probe_results.append(result)
    return probe_results


def print_bind_failure_diagnosis(bind_failures: list[dict[str, object]], probe_results: list[dict[str, object]]) -> None:
    print("❌ Gollek kernel self-test exhausted random-port retries during ZMQ bind startup.")
    print(f"   bind failures: {len(bind_failures)}")
    for failure in bind_failures:
        print(
            "   attempted ports: "
            f"{failure['ports']} "
            f"(attempt {failure['attempt']}/{failure['retries']})"
        )

    successful_probes = [result for result in probe_results if result["ok"]]
    failed_probes = [result for result in probe_results if not result["ok"]]

    if successful_probes and not failed_probes:
        print("   verdict: jeroMQ-or-kernel-bind-boundary")
        print("   note: plain Python TCP binds succeeded on fresh localhost ports,")
        print("         so this does not look like a normal OS-level port collision.")
        print("         The remaining boundary is likely JJava/JeroMQ startup in this environment.")
    elif failed_probes and not successful_probes:
        print("   verdict: environment-localhost-bind-restriction")
        print("   note: even plain Python TCP binds failed on fresh localhost ports,")
        print("         so this environment is restricting local socket startup broadly.")
    else:
        print("   verdict: mixed-bind-environment")
        print("   note: some plain Python localhost binds worked and some failed,")
        print("         so the environment is not giving a stable networking surface for the kernel.")

    print("   python bind probe:")
    for result in probe_results:
        status = "ok" if result["ok"] else "failed"
        detail = "" if result["ok"] else f" ({result['error']})"
        print(f"     - port {result['port']}: {status}{detail}")


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        jar_path = resolve_kernel_jar(base_dir, args.jar)
        validate_kernel_jar_freshness(base_dir, jar_path, args.allow_stale_jar)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"❌ {exc}")
        return 1

    command = build_kernel_command(base_dir, jar_path, args.java)
    bind_failures: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="gollek-jupyter-selftest-") as temp_dir:
        temp_path = Path(temp_dir)
        for attempt in range(1, args.retries + 1):
            connection_payload = build_connection_payload()
            connection_file = temp_path / f"connection-{attempt}.json"
            connection_file.write_text(json.dumps(connection_payload, indent=2))

            final_command = [arg.replace("{connection_file}", str(connection_file)) for arg in command]
            proc = subprocess.Popen(
                final_command,
                cwd=base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            started_at = time.time()

            try:
                while time.time() - started_at < args.timeout_secs:
                    if proc.poll() is not None:
                        stdout, stderr = proc.communicate()
                        if is_bind_collision(stderr) and attempt < args.retries:
                            bind_failures.append(
                                {
                                    "attempt": attempt,
                                    "retries": args.retries,
                                    "ports": summarize_connection_ports(connection_payload),
                                    "stderr": stderr.rstrip(),
                                }
                            )
                            print(f"Retrying self-test after bind collision on attempt {attempt}/{args.retries}.")
                            break
                        if is_bind_collision(stderr):
                            bind_failures.append(
                                {
                                    "attempt": attempt,
                                    "retries": args.retries,
                                    "ports": summarize_connection_ports(connection_payload),
                                    "stderr": stderr.rstrip(),
                                }
                            )
                            print_bind_failure_diagnosis(bind_failures, probe_local_tcp_bind())
                            print("   final stderr:")
                            print(stderr.rstrip())
                            return 1
                        print("❌ Kernel exited before startup window completed.")
                        print(f"   exit_code: {proc.returncode}")
                        print(f"   attempt: {attempt}/{args.retries}")
                        if stdout.strip():
                            print("   stdout:")
                            print(stdout.rstrip())
                        if stderr.strip():
                            print("   stderr:")
                            print(stderr.rstrip())
                        return 1
                    time.sleep(0.1)
                else:
                    stdout, stderr = proc.communicate(timeout=0) if proc.poll() is not None else ("", "")
                    print("✅ Gollek kernel self-test passed")
                    print(f"   kernel jar: {jar_path}")
                    print(f"   startup window: {args.timeout_secs:.1f}s")
                    print(f"   connection file: {connection_file}")
                    print(f"   attempts used: {attempt}")
                    print(f"   command: {' '.join(final_command)}")
                    if args.verbose:
                        if stdout.strip():
                            print("   stdout:")
                            print(stdout.rstrip())
                        if stderr.strip():
                            print("   stderr:")
                            print(stderr.rstrip())
                    return 0
            finally:
                terminate_process(proc)

    print("❌ Kernel self-test exhausted all retry attempts.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
