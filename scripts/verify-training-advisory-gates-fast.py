#!/usr/bin/env python3
"""Fast training metadata gate for Gollek training modules.

This script intentionally avoids Gradle project configuration. It derives the
same training metadata from checked-in settings/build files, writes the regular
Gollek training gate JSON reports, and validates hard-check/advisory lock drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_FILE_SNAPSHOT: dict[Path, str | None] | None = None

TRAINING_FAST_GATE_SUMMARY_FORMAT = "gollek.training.fast-gate-summary.v1"
TRAINING_PROMOTION_PLAN_FORMAT = "gollek.training.promotion-plan.v1"
TRAINING_PROMOTION_PLAN_VALIDATION_FORMAT = "gollek.training.promotion-plan-validation.v1"
TRAINING_COVERAGE_FORMAT = "gollek.training.module-coverage.v2"
TRAINING_CHECKS_FORMAT = "gollek.training.module-checks.v2"
TRAINING_CHECKS_LOCK_FORMAT = "gollek.training.module-checks.lock.v1"
TRAINING_CHECKS_LOCK_DRIFT_FORMAT = "gollek.training.module-checks.lock-drift.v1"
TRAINING_CANDIDATE_SELECTION_FORMAT = "gollek.training.module-candidate-selection.v2"
TRAINING_CANDIDATE_LOCK_FORMAT = "gollek.training.module-candidates.lock.v1"
TRAINING_CANDIDATE_LOCK_DRIFT_FORMAT = "gollek.training.module-candidates.lock-drift.v1"
TRAINING_CANDIDATE_SELECTION_PROPERTY = "gollekTrainingCandidate"
TRAINING_PROMOTION_MUTATION_FILES = [
    "buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts",
    "gradle/training-module-checks.lock.json",
    "gradle/training-module-candidates.lock.json",
]

TRAINING_CHECKS_FIELDS = [
    "format",
    "schemaVersion",
    "registryFingerprintAlgorithm",
    "registryFingerprint",
    "checkCount",
    "checks",
]
TRAINING_CHECK_FIELDS = ["label", "projectPath", "taskName", "taskPath", "projectExists", "taskExists"]
TRAINING_CHECK_LOCK_FIELDS = [
    "format",
    "schemaVersion",
    "reportFormat",
    "registryFingerprintAlgorithm",
    "registryFingerprint",
    "checkCount",
    "checks",
]
TRAINING_CHECK_LOCK_DRIFT_FIELDS = [
    "format",
    "schemaVersion",
    "passed",
    "missingLock",
    "lockFile",
    "expectedFingerprint",
    "actualFingerprint",
    "fingerprintDrift",
    "addedLabels",
    "removedLabels",
    "changedChecks",
]
TRAINING_CHECK_LOCK_ENTRY_FIELDS = ["label", "projectPath", "taskName", "taskPath"]
TRAINING_COVERAGE_FIELDS = [
    "format",
    "schemaVersion",
    "projectCount",
    "registeredProjectCount",
    "testProjectCount",
    "unregisteredTestProjectCount",
    "unregisteredTestProjects",
    "candidateCheckCount",
    "candidateChecks",
    "projects",
]
TRAINING_COVERAGE_PROJECT_FIELDS = [
    "projectPath",
    "projectDirectory",
    "hasTestSources",
    "testTaskExists",
    "registered",
    "labels",
    "taskPaths",
]
TRAINING_CANDIDATE_FIELDS = ["label", "projectPath", "taskName", "taskPath"]
TRAINING_CANDIDATE_LOCK_DRIFT_FIELDS = [
    "format",
    "schemaVersion",
    "passed",
    "missingLock",
    "lockFile",
    "expectedFingerprint",
    "actualFingerprint",
    "fingerprintDrift",
    "addedLabels",
    "removedLabels",
    "changedCandidates",
]
TRAINING_PROMOTION_PLAN_FIELDS = [
    "format",
    "schemaVersion",
    "promotionFingerprintAlgorithm",
    "promotionFingerprint",
    "selectionProperty",
    "selectionValue",
    "selectedLabels",
    "selectedTaskPaths",
    "candidateCount",
    "dryRun",
    "applied",
    "transactional",
    "registryFile",
    "checkLockFile",
    "candidateLockFile",
    "mutationFiles",
    "promotionPlanFile",
    "candidates",
    "promotionSnippets",
    "commands",
]
TRAINING_PROMOTION_PLAN_COMMAND_FIELDS = [
    "validateSelection",
    "runCandidateTests",
    "writeChecksLock",
    "writeCandidateLock",
    "verifyFastGate",
    "verifyGradleGate",
]
TRAINING_PROMOTION_PLAN_VALIDATION_FIELDS = [
    "format",
    "schemaVersion",
    "passed",
    "planFile",
    "planFormat",
    "planSchemaVersion",
    "promotionFingerprintAlgorithm",
    "promotionFingerprint",
    "candidateCount",
    "selectedLabels",
    "selectedTaskPaths",
    "dryRun",
    "applied",
    "transactional",
    "mutationFiles",
    "durationMs",
    "error",
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot_files(paths: list[Path]) -> dict[Path, str | None]:
    return {path: path.read_text(encoding="utf-8") if path.is_file() else None for path in paths}


def restore_file_snapshots(snapshots: dict[Path, str | None]) -> None:
    for path, content in snapshots.items():
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def begin_file_transaction(paths: list[Path]) -> None:
    global ACTIVE_FILE_SNAPSHOT
    if ACTIVE_FILE_SNAPSHOT is not None:
        raise RuntimeError("file transaction already active")
    ACTIVE_FILE_SNAPSHOT = snapshot_files(paths)


def commit_file_transaction() -> None:
    global ACTIVE_FILE_SNAPSHOT
    ACTIVE_FILE_SNAPSHOT = None


def rollback_active_file_transaction() -> None:
    global ACTIVE_FILE_SNAPSHOT
    if ACTIVE_FILE_SNAPSHOT is not None:
        restore_file_snapshots(ACTIVE_FILE_SNAPSHOT)
        ACTIVE_FILE_SNAPSHOT = None
        print("Rolled back training promotion file changes.", file=sys.stderr)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint_from_rows(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        "|".join(
            [
                str(row.get("label", "")),
                str(row.get("projectPath", "")),
                str(row.get("taskName", "")),
                str(row.get("taskPath", "")),
            ]
        )
        + "\n"
        for row in rows
    )
    return sha256_hex(body)


def promotion_fingerprint(selection_value: str, selected_candidates: list[dict[str, Any]]) -> str:
    canonical = {
        "selectionProperty": TRAINING_CANDIDATE_SELECTION_PROPERTY,
        "selectionValue": selection_value,
        "mutationFiles": TRAINING_PROMOTION_MUTATION_FILES,
        "candidates": [
            {field: str(candidate.get(field, "")) for field in TRAINING_CANDIDATE_FIELDS}
            for candidate in selected_candidates
        ],
    }
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(body)


def parse_training_projects(settings_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'project\("(?P<project>:[^"]+)"\)\.projectDir\s*=\s*file\("(?P<directory>training/[^"]+)"\)'
    )
    projects = []
    for match in pattern.finditer(settings_text):
        project_path = match.group("project")
        directory = match.group("directory")
        project_dir = ROOT / directory
        if (project_dir / "build.gradle.kts").is_file() or (project_dir / "build.gradle").is_file():
            projects.append((project_path, directory))
    return sorted(projects, key=lambda item: item[0])


def parse_registered_checks(plugin_text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r'TrainingModuleCheck\(\s*label\s*=\s*"(?P<label>[^"]+)"\s*,\s*'
        r'projectPath\s*=\s*"(?P<project>[^"]+)"',
        re.DOTALL,
    )
    checks = []
    for match in pattern.finditer(plugin_text):
        project_path = match.group("project")
        checks.append(
            {
                "label": match.group("label"),
                "projectPath": project_path,
                "taskName": "test",
                "taskPath": f"{project_path}:test",
            }
        )
    return sorted(checks, key=lambda item: item["label"])


def check_report_payload(projects: list[tuple[str, str]], checks: list[dict[str, str]]) -> dict[str, Any]:
    project_directories = dict(projects)
    rows = []
    for check in checks:
        project_directory = project_directories.get(check["projectPath"])
        project_exists = project_directory is not None
        project_dir = ROOT / project_directory if project_directory else None
        task_exists = bool(
            project_dir
            and ((project_dir / "build.gradle.kts").is_file() or (project_dir / "build.gradle").is_file())
        )
        rows.append(
            {
                "label": check["label"],
                "projectPath": check["projectPath"],
                "taskName": check["taskName"],
                "taskPath": check["taskPath"],
                "projectExists": project_exists,
                "taskExists": task_exists,
            }
        )

    return {
        "format": TRAINING_CHECKS_FORMAT,
        "schemaVersion": 2,
        "registryFingerprintAlgorithm": "SHA-256",
        "registryFingerprint": fingerprint_from_rows(rows),
        "checkCount": len(rows),
        "checks": rows,
    }


def check_lock_rows(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": check["label"],
            "projectPath": check["projectPath"],
            "taskName": check["taskName"],
            "taskPath": check["taskPath"],
        }
        for check in checks
    ]


def check_lock_payload(checks: list[dict[str, Any]]) -> dict[str, Any]:
    lock_rows = check_lock_rows(checks)
    return {
        "format": TRAINING_CHECKS_LOCK_FORMAT,
        "schemaVersion": 1,
        "reportFormat": TRAINING_CHECKS_FORMAT,
        "registryFingerprintAlgorithm": "SHA-256",
        "registryFingerprint": fingerprint_from_rows(lock_rows),
        "checkCount": len(lock_rows),
        "checks": lock_rows,
    }


def check_lock_drift_payload(
    lock_file: Path,
    lock_payload: dict[str, Any] | None,
    current_payload: dict[str, Any],
) -> dict[str, Any]:
    current = rows_by_label(current_payload.get("checks"))
    locked = rows_by_label(lock_payload.get("checks") if lock_payload else None)
    added = sorted(set(current).difference(locked))
    removed = sorted(set(locked).difference(current))
    changed = []
    for label in sorted(set(current).intersection(locked)):
        current_row = current[label]
        locked_row = locked[label]
        if any(str(current_row.get(field, "")) != str(locked_row.get(field, "")) for field in TRAINING_CHECK_LOCK_ENTRY_FIELDS):
            changed.append(
                {
                    "label": label,
                    "expected": {field: str(locked_row.get(field, "")) for field in TRAINING_CHECK_LOCK_ENTRY_FIELDS},
                    "actual": {field: str(current_row.get(field, "")) for field in TRAINING_CHECK_LOCK_ENTRY_FIELDS},
                }
            )

    expected_fingerprint = str(lock_payload.get("registryFingerprint")) if lock_payload else None
    actual_fingerprint = str(current_payload["registryFingerprint"])
    missing_lock = lock_payload is None
    fingerprint_drift = (not missing_lock) and expected_fingerprint != actual_fingerprint
    passed = not missing_lock and not fingerprint_drift and not added and not removed and not changed
    return {
        "format": TRAINING_CHECKS_LOCK_DRIFT_FORMAT,
        "schemaVersion": 1,
        "passed": passed,
        "missingLock": missing_lock,
        "lockFile": str(lock_file.resolve()),
        "expectedFingerprint": expected_fingerprint,
        "actualFingerprint": actual_fingerprint,
        "fingerprintDrift": fingerprint_drift,
        "addedLabels": added,
        "removedLabels": removed,
        "changedChecks": changed,
    }


def suggested_check_label(project_path: str) -> str:
    label = project_path.rsplit(":", 1)[-1]
    for prefix in ("gollek-", "ml-", "train-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label if label else project_path.strip(":").replace(":", "-")


def candidate_row(project_path: str) -> dict[str, str]:
    return {
        "label": suggested_check_label(project_path),
        "projectPath": project_path,
        "taskName": "test",
        "taskPath": f"{project_path}:test",
    }


def coverage_payload(projects: list[tuple[str, str]], checks: list[dict[str, str]]) -> dict[str, Any]:
    checks_by_project: dict[str, list[dict[str, str]]] = {}
    for check in checks:
        checks_by_project.setdefault(check["projectPath"], []).append(check)

    rows = []
    for project_path, directory in projects:
        project_dir = ROOT / directory
        registered_checks = sorted(checks_by_project.get(project_path, []), key=lambda item: item["label"])
        build_file_exists = (project_dir / "build.gradle.kts").is_file() or (project_dir / "build.gradle").is_file()
        row = {
            "projectPath": project_path,
            "projectDirectory": directory,
            "hasTestSources": (project_dir / "src/test").is_dir(),
            "testTaskExists": build_file_exists,
            "registered": bool(registered_checks),
            "labels": [check["label"] for check in registered_checks],
            "taskPaths": [check["taskPath"] for check in registered_checks],
        }
        rows.append(row)

    test_projects = [row for row in rows if row["hasTestSources"] and row["testTaskExists"]]
    registered_projects = [row for row in rows if row["registered"]]
    unregistered_test_projects = sorted(
        row["projectPath"] for row in test_projects if not row["registered"]
    )
    candidates = sorted(
        [candidate_row(project_path) for project_path in unregistered_test_projects],
        key=lambda item: item["label"],
    )
    return {
        "format": TRAINING_COVERAGE_FORMAT,
        "schemaVersion": 2,
        "projectCount": len(rows),
        "registeredProjectCount": len(registered_projects),
        "testProjectCount": len(test_projects),
        "unregisteredTestProjectCount": len(unregistered_test_projects),
        "unregisteredTestProjects": unregistered_test_projects,
        "candidateCheckCount": len(candidates),
        "candidateChecks": candidates,
        "projects": rows,
    }


def selection_labels(selection_value: str) -> set[str]:
    labels = [item.strip() for item in selection_value.split(",") if item.strip()]
    if not labels or any(item == "*" or item.lower() == "all" for item in labels):
        return set()
    return set(labels)


def candidates_for_selection(
    selection_value: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = selection_labels(selection_value)
    if not labels:
        return candidates

    by_label = {str(candidate["label"]): candidate for candidate in candidates}
    missing = sorted(labels.difference(by_label))
    if missing:
        available = ", ".join(sorted(by_label)) or "none"
        raise SystemExit(
            "Unknown advisory training module candidate(s): "
            + ", ".join(missing)
            + f". Available labels: {available}"
        )
    return [by_label[label] for label in sorted(labels)]


def promotion_snippet(candidate: dict[str, Any]) -> str:
    return (
        "TrainingModuleCheck(\n"
        f"    label = \"{candidate['label']}\",\n"
        f"    projectPath = \"{candidate['projectPath']}\",\n"
        "),"
    )


def indented_promotion_snippet(candidate: dict[str, Any]) -> str:
    return "\n".join(f"    {line}" if line else "" for line in promotion_snippet(candidate).splitlines())


def update_registry_text(plugin_text: str, candidates: list[dict[str, Any]]) -> str:
    existing_labels = {check["label"] for check in parse_registered_checks(plugin_text)}
    new_candidates = [candidate for candidate in candidates if candidate["label"] not in existing_labels]
    if not new_candidates:
        return plugin_text

    anchor = "    // Add new Gradle-wired training module checks here as modules mature.\n"
    if anchor not in plugin_text:
        raise SystemExit("Could not locate trainingModuleChecks insertion anchor in gollek.training-gates.gradle.kts")

    insertion = "".join(indented_promotion_snippet(candidate) + "\n" for candidate in new_candidates)
    return plugin_text.replace(anchor, anchor + insertion, 1)


def promotion_plan_payload(
    *,
    selection_value: str,
    selected_candidates: list[dict[str, Any]],
    promotion_plan_file: Path,
    dry_run: bool,
    applied: bool,
) -> dict[str, Any]:
    selected_labels = [candidate["label"] for candidate in selected_candidates]
    selected_task_paths = [candidate["taskPath"] for candidate in selected_candidates]
    selection_suffix = selection_command_suffix(selection_value)
    mutation_files = [
        str(ROOT / relative_path)
        for relative_path in TRAINING_PROMOTION_MUTATION_FILES
    ]
    return {
        "format": TRAINING_PROMOTION_PLAN_FORMAT,
        "schemaVersion": 1,
        "promotionFingerprintAlgorithm": "SHA-256",
        "promotionFingerprint": promotion_fingerprint(selection_value, selected_candidates),
        "selectionProperty": TRAINING_CANDIDATE_SELECTION_PROPERTY,
        "selectionValue": selection_value,
        "selectedLabels": selected_labels,
        "selectedTaskPaths": selected_task_paths,
        "candidateCount": len(selected_candidates),
        "dryRun": dry_run,
        "applied": applied,
        "transactional": applied,
        "registryFile": mutation_files[0],
        "checkLockFile": mutation_files[1],
        "candidateLockFile": mutation_files[2],
        "mutationFiles": mutation_files,
        "promotionPlanFile": str(promotion_plan_file),
        "candidates": selected_candidates,
        "promotionSnippets": [promotion_snippet(candidate) for candidate in selected_candidates],
        "commands": {
            "validateSelection": f"./gradlew --no-daemon validateTrainingModuleCandidateSelection{selection_suffix}",
            "runCandidateTests": f"./gradlew --no-daemon checkTrainingModuleCandidateTests{selection_suffix}",
            "writeChecksLock": "./gradlew --no-daemon writeTrainingModuleChecksLock",
            "writeCandidateLock": "scripts/verify-training-advisory-gates-fast.py --write-candidate-lock",
            "verifyFastGate": "scripts/verify-training-advisory-gates-fast.py",
            "verifyGradleGate": "./gradlew --no-daemon verifyTrainingGates",
        },
    }


def selection_command_suffix(selection_value: str) -> str:
    trimmed = selection_value.strip()
    return f" -P{TRAINING_CANDIDATE_SELECTION_PROPERTY}={trimmed}" if trimmed else ""


def selection_next_commands(selection_value: str) -> list[str]:
    suffix = selection_command_suffix(selection_value)
    return [
        f"./gradlew --no-daemon validateTrainingModuleCandidateSelection{suffix}",
        f"./gradlew --no-daemon checkTrainingModuleCandidateTests{suffix}",
        f"./gradlew --no-daemon printTrainingModuleCandidatePromotionSnippets{suffix}",
        "./gradlew --no-daemon writeTrainingModuleChecksLock",
        "./gradlew --no-daemon verifyTrainingGates",
    ]


def selection_payload(selection_value: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected_labels = sorted(selection_labels(selection_value))
    selected_candidates = candidates_for_selection(selection_value, candidates)
    return {
        "format": TRAINING_CANDIDATE_SELECTION_FORMAT,
        "schemaVersion": 2,
        "selectionProperty": TRAINING_CANDIDATE_SELECTION_PROPERTY,
        "selectionValue": selection_value,
        "selectionFingerprintAlgorithm": "SHA-256",
        "selectionFingerprint": fingerprint_from_rows(selected_candidates),
        "selectedAll": not selected_labels,
        "selectedLabelCount": len(selected_labels),
        "selectedLabels": selected_labels,
        "candidateCount": len(selected_candidates),
        "taskPaths": [candidate["taskPath"] for candidate in selected_candidates],
        "candidates": selected_candidates,
        "promotionSnippets": [promotion_snippet(candidate) for candidate in selected_candidates],
        "nextCommands": selection_next_commands(selection_value),
    }


def candidate_lock_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": TRAINING_CANDIDATE_LOCK_FORMAT,
        "schemaVersion": 1,
        "coverageFormat": TRAINING_COVERAGE_FORMAT,
        "candidateFingerprintAlgorithm": "SHA-256",
        "candidateFingerprint": fingerprint_from_rows(candidates),
        "candidateCount": len(candidates),
        "candidates": candidates,
    }


def rows_by_label(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        if isinstance(row, dict):
            label = str(row.get("label", ""))
            if label:
                result[label] = row
    return result


def lock_drift_payload(
    lock_file: Path,
    lock_payload: dict[str, Any] | None,
    current_payload: dict[str, Any],
) -> dict[str, Any]:
    current = rows_by_label(current_payload.get("candidates"))
    locked = rows_by_label(lock_payload.get("candidates") if lock_payload else None)
    added = sorted(set(current).difference(locked))
    removed = sorted(set(locked).difference(current))
    changed = []
    for label in sorted(set(current).intersection(locked)):
        current_row = current[label]
        locked_row = locked[label]
        if any(str(current_row.get(field, "")) != str(locked_row.get(field, "")) for field in TRAINING_CANDIDATE_FIELDS):
            changed.append(
                {
                    "label": label,
                    "expected": {field: str(locked_row.get(field, "")) for field in TRAINING_CANDIDATE_FIELDS},
                    "actual": {field: str(current_row.get(field, "")) for field in TRAINING_CANDIDATE_FIELDS},
                }
            )

    expected_fingerprint = str(lock_payload.get("candidateFingerprint")) if lock_payload else None
    actual_fingerprint = str(current_payload["candidateFingerprint"])
    missing_lock = lock_payload is None
    fingerprint_drift = (not missing_lock) and expected_fingerprint != actual_fingerprint
    passed = not missing_lock and not fingerprint_drift and not added and not removed and not changed
    return {
        "format": TRAINING_CANDIDATE_LOCK_DRIFT_FORMAT,
        "schemaVersion": 1,
        "passed": passed,
        "missingLock": missing_lock,
        "lockFile": str(lock_file.resolve()),
        "expectedFingerprint": expected_fingerprint,
        "actualFingerprint": actual_fingerprint,
        "fingerprintDrift": fingerprint_drift,
        "addedLabels": added,
        "removedLabels": removed,
        "changedCandidates": changed,
    }


def fast_gate_summary_payload(
    *,
    duration_ms: int,
    output_dir: Path,
    summary_file: Path,
    check_report: dict[str, Any],
    check_drift: dict[str, Any],
    coverage: dict[str, Any],
    selection: dict[str, Any],
    candidate_drift: dict[str, Any],
    promotion_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    check_lock_passed = check_drift["passed"] is True
    candidate_lock_passed = candidate_drift["passed"] is True
    report_paths = {
        "checks": output_dir / "training-module-checks.json",
        "checksLockDrift": output_dir / "training-module-checks-lock-drift.json",
        "coverage": output_dir / "training-module-coverage.json",
        "candidateSelection": output_dir / "training-module-candidate-selection.json",
        "candidatesLockDrift": output_dir / "training-module-candidates-lock-drift.json",
        "summary": summary_file,
    }
    if promotion_plan is not None:
        report_paths["promotionPlan"] = Path(promotion_plan["promotionPlanFile"])
    return {
        "format": TRAINING_FAST_GATE_SUMMARY_FORMAT,
        "schemaVersion": 1,
        "passed": check_lock_passed and candidate_lock_passed,
        "durationMs": duration_ms,
        "rootDirectory": str(ROOT),
        "outputDirectory": str(output_dir),
        "selectionProperty": TRAINING_CANDIDATE_SELECTION_PROPERTY,
        "selectionValue": selection["selectionValue"],
        "counts": {
            "hardCheckCount": check_report["checkCount"],
            "projectCount": coverage["projectCount"],
            "testProjectCount": coverage["testProjectCount"],
            "registeredProjectCount": coverage["registeredProjectCount"],
            "candidateCount": coverage["candidateCheckCount"],
            "selectedCandidateCount": selection["candidateCount"],
        },
        "fingerprints": {
            "registry": check_report["registryFingerprint"],
            "candidate": candidate_drift["actualFingerprint"],
            "selection": selection["selectionFingerprint"],
        },
        "locks": {
            "checks": {
                "passed": check_lock_passed,
                "missingLock": check_drift["missingLock"],
                "fingerprintDrift": check_drift["fingerprintDrift"],
                "addedLabels": check_drift["addedLabels"],
                "removedLabels": check_drift["removedLabels"],
                "changedLabels": [item["label"] for item in check_drift["changedChecks"]],
                "lockFile": check_drift["lockFile"],
            },
            "candidates": {
                "passed": candidate_lock_passed,
                "missingLock": candidate_drift["missingLock"],
                "fingerprintDrift": candidate_drift["fingerprintDrift"],
                "addedLabels": candidate_drift["addedLabels"],
                "removedLabels": candidate_drift["removedLabels"],
                "changedLabels": [item["label"] for item in candidate_drift["changedCandidates"]],
                "lockFile": candidate_drift["lockFile"],
            },
        },
        "selectedLabels": selection["selectedLabels"],
        "selectedTaskPaths": selection["taskPaths"],
        "promotion": {
            "requested": promotion_plan is not None,
            "applied": bool(promotion_plan and promotion_plan["applied"]),
            "dryRun": bool(promotion_plan and promotion_plan["dryRun"]),
            "selectedLabels": promotion_plan["selectedLabels"] if promotion_plan else [],
        },
        "reports": {name: str(path) for name, path in report_paths.items()},
        "repairCommands": {
            "checksLock": "scripts/verify-training-advisory-gates-fast.py --write-check-lock",
            "candidatesLock": "scripts/verify-training-advisory-gates-fast.py --write-candidate-lock",
            "gradleGate": "./gradlew --no-daemon verifyTrainingGates",
        },
        "nextCommands": selection["nextCommands"],
    }


def require_fields(name: str, payload: dict[str, Any], fields: list[str]) -> None:
    actual = set(payload)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        raise SystemExit(f"{name} fields drifted ({'; '.join(details)})")


def require_coverage_contract(payload: dict[str, Any]) -> None:
    require_fields("coverage report", payload, TRAINING_COVERAGE_FIELDS)
    if payload["format"] != TRAINING_COVERAGE_FORMAT or payload["schemaVersion"] != 2:
        raise SystemExit("coverage report format/schemaVersion drifted")
    projects = payload["projects"]
    candidates = payload["candidateChecks"]
    if payload["projectCount"] != len(projects):
        raise SystemExit("coverage report projectCount does not match projects")
    if payload["candidateCheckCount"] != len(candidates):
        raise SystemExit("coverage report candidateCheckCount does not match candidateChecks")
    labels = set()
    task_paths = set()
    for project in projects:
        require_fields("coverage project", project, TRAINING_COVERAGE_PROJECT_FIELDS)
        if not str(project["projectPath"]).startswith(":"):
            raise SystemExit(f"coverage projectPath should be absolute: {project['projectPath']}")
        if not str(project["projectDirectory"]).startswith("training/"):
            raise SystemExit(f"coverage projectDirectory should be under training/: {project['projectDirectory']}")
    for candidate in candidates:
        require_fields("coverage candidate", candidate, TRAINING_CANDIDATE_FIELDS)
        label = str(candidate["label"])
        task_path = str(candidate["taskPath"])
        expected_task_path = f"{candidate['projectPath']}:{candidate['taskName']}"
        if not label or label in labels:
            raise SystemExit(f"coverage candidate label should be unique and non-blank: {label}")
        if task_path in task_paths or task_path != expected_task_path:
            raise SystemExit(f"coverage candidate taskPath drifted: {task_path}")
        labels.add(label)
        task_paths.add(task_path)


def require_check_report_contract(payload: dict[str, Any]) -> None:
    require_fields("training checks report", payload, TRAINING_CHECKS_FIELDS)
    if payload.get("format") != TRAINING_CHECKS_FORMAT or payload.get("schemaVersion") != 2:
        raise SystemExit("training checks report format/schemaVersion drifted")
    if payload.get("registryFingerprintAlgorithm") != "SHA-256":
        raise SystemExit("training checks report registryFingerprintAlgorithm should be SHA-256")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise SystemExit("training checks report checks should be a list")
    if payload.get("checkCount") != len(checks):
        raise SystemExit("training checks report checkCount does not match checks")
    labels = set()
    task_paths = set()
    for check in checks:
        require_fields("training check", check, TRAINING_CHECK_FIELDS)
        label = str(check["label"])
        task_path = str(check["taskPath"])
        expected_task_path = f"{check['projectPath']}:{check['taskName']}"
        if not label or label in labels:
            raise SystemExit(f"training check label should be unique and non-blank: {label}")
        if task_path in task_paths or task_path != expected_task_path:
            raise SystemExit(f"training check taskPath drifted: {task_path}")
        if check["projectExists"] is not True or check["taskExists"] is not True:
            raise SystemExit(f"training check should resolve to an existing project/task: {label}")
        labels.add(label)
        task_paths.add(task_path)
    if payload.get("registryFingerprint") != fingerprint_from_rows(checks):
        raise SystemExit("training checks report registryFingerprint drifted")


def require_check_lock_contract(payload: dict[str, Any]) -> None:
    require_fields("training checks lock", payload, TRAINING_CHECK_LOCK_FIELDS)
    if payload.get("format") != TRAINING_CHECKS_LOCK_FORMAT:
        raise SystemExit("training checks lock format drifted")
    if payload.get("reportFormat") != TRAINING_CHECKS_FORMAT:
        raise SystemExit("training checks lock reportFormat drifted")
    if payload.get("registryFingerprintAlgorithm") != "SHA-256":
        raise SystemExit("training checks lock registryFingerprintAlgorithm should be SHA-256")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise SystemExit("training checks lock checks should be a non-empty list")
    if payload.get("checkCount") != len(checks):
        raise SystemExit("training checks lock checkCount does not match checks")
    for check in checks:
        require_fields("training checks lock entry", check, TRAINING_CHECK_LOCK_ENTRY_FIELDS)
        if str(check.get("taskPath", "")) != f"{check.get('projectPath', '')}:{check.get('taskName', '')}":
            raise SystemExit(f"training checks lock taskPath drifted: {check.get('taskPath')}")
    if payload.get("registryFingerprint") != fingerprint_from_rows(checks):
        raise SystemExit("training checks lock registryFingerprint drifted")


def require_candidate_lock_contract(payload: dict[str, Any]) -> None:
    if payload.get("format") != TRAINING_CANDIDATE_LOCK_FORMAT:
        raise SystemExit("candidate lock format drifted")
    if payload.get("coverageFormat") != TRAINING_COVERAGE_FORMAT:
        raise SystemExit("candidate lock coverageFormat drifted")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("candidate lock candidates should be a list")
    if payload.get("candidateCount") != len(candidates):
        raise SystemExit("candidate lock candidateCount does not match candidates")
    if payload.get("candidateFingerprint") != fingerprint_from_rows(candidates):
        raise SystemExit("candidate lock candidateFingerprint drifted")


def require_promotion_plan_contract(payload: dict[str, Any]) -> None:
    require_fields("promotion plan", payload, TRAINING_PROMOTION_PLAN_FIELDS)
    if payload.get("format") != TRAINING_PROMOTION_PLAN_FORMAT or payload.get("schemaVersion") != 1:
        raise SystemExit("promotion plan format/schemaVersion drifted")
    if payload.get("promotionFingerprintAlgorithm") != "SHA-256":
        raise SystemExit("promotion plan promotionFingerprintAlgorithm should be SHA-256")
    if payload.get("selectionProperty") != TRAINING_CANDIDATE_SELECTION_PROPERTY:
        raise SystemExit("promotion plan selectionProperty drifted")
    for field in ("dryRun", "applied", "transactional"):
        if not isinstance(payload.get(field), bool):
            raise SystemExit(f"promotion plan {field} should be boolean")
    if payload["dryRun"] and payload["applied"]:
        raise SystemExit("promotion plan cannot be both dryRun and applied")
    if payload["applied"] and not payload["transactional"]:
        raise SystemExit("applied promotion plan should be transactional")

    candidates = payload.get("candidates")
    snippets = payload.get("promotionSnippets")
    if not isinstance(candidates, list) or not isinstance(snippets, list):
        raise SystemExit("promotion plan candidates and promotionSnippets should be lists")
    if payload.get("candidateCount") != len(candidates) or len(snippets) != len(candidates):
        raise SystemExit("promotion plan candidateCount does not match candidates/snippets")

    expected_labels = []
    expected_task_paths = []
    expected_snippets = []
    for candidate in candidates:
        require_fields("promotion plan candidate", candidate, TRAINING_CANDIDATE_FIELDS)
        expected_task_path = f"{candidate['projectPath']}:{candidate['taskName']}"
        if candidate["taskPath"] != expected_task_path:
            raise SystemExit(f"promotion plan candidate taskPath drifted: {candidate['taskPath']}")
        expected_labels.append(candidate["label"])
        expected_task_paths.append(candidate["taskPath"])
        expected_snippets.append(promotion_snippet(candidate))
    if payload["selectedLabels"] != expected_labels:
        raise SystemExit("promotion plan selectedLabels do not match candidates")
    if payload["selectedTaskPaths"] != expected_task_paths:
        raise SystemExit("promotion plan selectedTaskPaths do not match candidates")
    if snippets != expected_snippets:
        raise SystemExit("promotion plan snippets do not match candidates")
    expected_fingerprint = promotion_fingerprint(str(payload["selectionValue"]), candidates)
    if payload.get("promotionFingerprint") != expected_fingerprint:
        raise SystemExit("promotion plan promotionFingerprint drifted")

    mutation_files = payload.get("mutationFiles")
    expected_mutation_files = [
        payload["registryFile"],
        payload["checkLockFile"],
        payload["candidateLockFile"],
    ]
    if mutation_files != expected_mutation_files:
        raise SystemExit("promotion plan mutationFiles should match registry/check/candidate lock files")
    for field in ("registryFile", "checkLockFile", "candidateLockFile", "promotionPlanFile"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise SystemExit(f"promotion plan {field} should be a non-empty string")

    commands = payload.get("commands")
    if not isinstance(commands, dict):
        raise SystemExit("promotion plan commands should be an object")
    require_fields("promotion plan commands", commands, TRAINING_PROMOTION_PLAN_COMMAND_FIELDS)
    for field in TRAINING_PROMOTION_PLAN_COMMAND_FIELDS:
        if not isinstance(commands.get(field), str) or not commands[field].strip():
            raise SystemExit(f"promotion plan command {field} should be a non-empty string")
    selection_suffix = selection_command_suffix(str(payload["selectionValue"]))
    if selection_suffix and selection_suffix not in commands["validateSelection"]:
        raise SystemExit("promotion plan validateSelection command lost selected candidates")
    if selection_suffix and selection_suffix not in commands["runCandidateTests"]:
        raise SystemExit("promotion plan runCandidateTests command lost selected candidates")


def load_check_lock(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(read_text(path))
    require_check_lock_contract(payload)
    return payload


def load_candidate_lock(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(read_text(path))
    require_candidate_lock_contract(payload)
    return payload


def load_promotion_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(read_text(path))
    require_promotion_plan_contract(payload)
    return payload


def promotion_plan_validation_payload(
    *,
    plan_file: Path,
    plan: dict[str, Any] | None,
    duration_ms: int,
    error: str = "",
) -> dict[str, Any]:
    return {
        "format": TRAINING_PROMOTION_PLAN_VALIDATION_FORMAT,
        "schemaVersion": 1,
        "passed": not error,
        "planFile": str(plan_file),
        "planFormat": str(plan.get("format", "")) if plan else "",
        "planSchemaVersion": plan.get("schemaVersion", 0) if plan else 0,
        "promotionFingerprintAlgorithm": str(plan.get("promotionFingerprintAlgorithm", "")) if plan else "",
        "promotionFingerprint": str(plan.get("promotionFingerprint", "")) if plan else "",
        "candidateCount": plan.get("candidateCount", 0) if plan else 0,
        "selectedLabels": plan.get("selectedLabels", []) if plan else [],
        "selectedTaskPaths": plan.get("selectedTaskPaths", []) if plan else [],
        "dryRun": bool(plan.get("dryRun", False)) if plan else False,
        "applied": bool(plan.get("applied", False)) if plan else False,
        "transactional": bool(plan.get("transactional", False)) if plan else False,
        "mutationFiles": plan.get("mutationFiles", []) if plan else [],
        "durationMs": duration_ms,
        "error": error,
    }


def require_promotion_plan_validation_contract(payload: dict[str, Any]) -> None:
    require_fields("promotion plan validation report", payload, TRAINING_PROMOTION_PLAN_VALIDATION_FIELDS)
    if payload.get("format") != TRAINING_PROMOTION_PLAN_VALIDATION_FORMAT or payload.get("schemaVersion") != 1:
        raise SystemExit("promotion plan validation report format/schemaVersion drifted")
    if not isinstance(payload.get("passed"), bool):
        raise SystemExit("promotion plan validation report passed should be boolean")
    if payload["passed"] and payload.get("error"):
        raise SystemExit("passed promotion plan validation report should not include an error")
    if not payload["passed"] and not payload.get("error"):
        raise SystemExit("failed promotion plan validation report should include an error")
    for field in ("selectedLabels", "selectedTaskPaths", "mutationFiles"):
        if not isinstance(payload.get(field), list):
            raise SystemExit(f"promotion plan validation report {field} should be a list")
    if payload["passed"]:
        if payload.get("promotionFingerprintAlgorithm") != "SHA-256":
            raise SystemExit("passed promotion plan validation report should include SHA-256 fingerprint algorithm")
        if not isinstance(payload.get("promotionFingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", payload["promotionFingerprint"]):
            raise SystemExit("passed promotion plan validation report should include a SHA-256 promotionFingerprint")
    for field in ("planFile", "planFormat", "promotionFingerprintAlgorithm", "promotionFingerprint", "error"):
        if not isinstance(payload.get(field), str):
            raise SystemExit(f"promotion plan validation report {field} should be a string")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="",
        help="Comma-separated advisory candidate labels. Empty, '*', or 'all' selects all candidates.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "build/reports/gollek"),
        help="Directory for generated training metadata JSON reports.",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help="Path for the aggregate fast gate summary JSON. Default: <output-dir>/training-module-fast-gate-summary.json.",
    )
    parser.add_argument(
        "--promotion-plan-file",
        default="",
        help="Path for a selected-candidate promotion plan JSON. Default with --promote-selected: <output-dir>/training-module-promotion-plan.json.",
    )
    parser.add_argument(
        "--validate-promotion-plan",
        default="",
        help="Validate an existing promotion plan JSON and exit without generating reports.",
    )
    parser.add_argument(
        "--promotion-plan-validation-report-file",
        default="",
        help="Path for a JSON validation report when using --validate-promotion-plan.",
    )
    parser.add_argument(
        "--promote-selected",
        action="store_true",
        help="Promote explicitly selected advisory candidates into the Gradle training check registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --promote-selected or --promotion-plan-file, write the promotion plan without editing registry or lock files.",
    )
    parser.add_argument(
        "--write-candidate-lock",
        action="store_true",
        help="Write gradle/training-module-candidates.lock.json instead of only validating drift.",
    )
    parser.add_argument(
        "--write-check-lock",
        action="store_true",
        help="Write gradle/training-module-checks.lock.json instead of only validating drift.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    if args.validate_promotion_plan:
        if (
            args.candidate
            or args.promotion_plan_file
            or args.promote_selected
            or args.dry_run
            or args.write_candidate_lock
            or args.write_check_lock
            or args.summary_file
        ):
            raise SystemExit("--validate-promotion-plan cannot be combined with report generation or promotion options")
        promotion_plan_path = Path(args.validate_promotion_plan)
        if not promotion_plan_path.is_absolute():
            promotion_plan_path = ROOT / promotion_plan_path
        validation_report_file = Path(args.promotion_plan_validation_report_file) if args.promotion_plan_validation_report_file else None
        if validation_report_file is not None and not validation_report_file.is_absolute():
            validation_report_file = ROOT / validation_report_file
        try:
            plan = load_promotion_plan(promotion_plan_path)
        except SystemExit as exc:
            if validation_report_file is not None:
                report = promotion_plan_validation_payload(
                    plan_file=promotion_plan_path,
                    plan=None,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    error=str(exc),
                )
                require_promotion_plan_validation_contract(report)
                write_json(validation_report_file, report)
            raise
        if validation_report_file is not None:
            report = promotion_plan_validation_payload(
                plan_file=promotion_plan_path,
                plan=plan,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            require_promotion_plan_validation_contract(report)
            write_json(validation_report_file, report)
        labels = ", ".join(plan["selectedLabels"]) or "none"
        print(
            "Validated training module promotion plan: "
            f"{promotion_plan_path} "
            f"({plan['candidateCount']} candidate(s): {labels})"
        )
        if validation_report_file is not None:
            print(f"Wrote promotion plan validation report: {validation_report_file}")
        return 0

    if args.promotion_plan_validation_report_file:
        raise SystemExit("--promotion-plan-validation-report-file requires --validate-promotion-plan")

    if args.dry_run and not (args.promote_selected or args.promotion_plan_file):
        raise SystemExit("--dry-run is only valid with --promote-selected or --promotion-plan-file")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    summary_file = Path(args.summary_file) if args.summary_file else output_dir / "training-module-fast-gate-summary.json"
    if not summary_file.is_absolute():
        summary_file = ROOT / summary_file
    promotion_plan_file = Path(args.promotion_plan_file) if args.promotion_plan_file else None
    if promotion_plan_file is not None and not promotion_plan_file.is_absolute():
        promotion_plan_file = ROOT / promotion_plan_file

    settings_text = read_text(ROOT / "settings.gradle.kts")
    plugin_file = ROOT / "buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts"
    check_lock_file = ROOT / "gradle/training-module-checks.lock.json"
    candidate_lock_file = ROOT / "gradle/training-module-candidates.lock.json"
    plugin_text = read_text(plugin_file)

    projects = parse_training_projects(settings_text)
    checks = parse_registered_checks(plugin_text)
    check_report = check_report_payload(projects, checks)
    require_check_report_contract(check_report)
    check_lock = check_lock_payload(check_report["checks"])
    require_check_lock_contract(check_lock)

    coverage = coverage_payload(projects, checks)
    require_coverage_contract(coverage)

    candidates = coverage["candidateChecks"]
    selection = selection_payload(args.candidate, candidates)
    promotion_plan = None
    wants_promotion_plan = args.promote_selected or promotion_plan_file is not None
    if wants_promotion_plan:
        if not selection_labels(args.candidate):
            raise SystemExit("--promote-selected and --promotion-plan-file require concrete --candidate labels")
        if promotion_plan_file is None:
            promotion_plan_file = output_dir / "training-module-promotion-plan.json"
        selected_candidates = list(selection["candidates"])
        if args.promote_selected and not args.dry_run:
            begin_file_transaction([plugin_file, check_lock_file, candidate_lock_file])
            updated_plugin_text = update_registry_text(plugin_text, selected_candidates)
            if updated_plugin_text != plugin_text:
                plugin_file.write_text(updated_plugin_text, encoding="utf-8")
                plugin_text = updated_plugin_text
                checks = parse_registered_checks(plugin_text)
                check_report = check_report_payload(projects, checks)
                require_check_report_contract(check_report)
                check_lock = check_lock_payload(check_report["checks"])
                require_check_lock_contract(check_lock)
                coverage = coverage_payload(projects, checks)
                require_coverage_contract(coverage)
                candidates = coverage["candidateChecks"]
                selection = selection_payload("", candidates)
            promotion_applied = True
        else:
            promotion_applied = False
        promotion_plan = promotion_plan_payload(
            selection_value=args.candidate,
            selected_candidates=selected_candidates,
            promotion_plan_file=promotion_plan_file,
            dry_run=args.dry_run or not args.promote_selected,
            applied=promotion_applied,
        )
        require_promotion_plan_contract(promotion_plan)

    candidate_lock = candidate_lock_payload(candidates)
    require_candidate_lock_contract(candidate_lock)

    if args.write_check_lock or (args.promote_selected and not args.dry_run):
        write_json(check_lock_file, check_lock)
    check_drift = check_lock_drift_payload(check_lock_file, load_check_lock(check_lock_file), check_lock)
    require_fields("training checks lock drift report", check_drift, TRAINING_CHECK_LOCK_DRIFT_FIELDS)

    if args.write_candidate_lock or (args.promote_selected and not args.dry_run):
        write_json(candidate_lock_file, candidate_lock)

    candidate_drift = lock_drift_payload(candidate_lock_file, load_candidate_lock(candidate_lock_file), candidate_lock)
    require_fields("candidate lock drift report", candidate_drift, TRAINING_CANDIDATE_LOCK_DRIFT_FIELDS)

    write_json(output_dir / "training-module-checks.json", check_report)
    write_json(output_dir / "training-module-checks-lock-drift.json", check_drift)
    write_json(output_dir / "training-module-coverage.json", coverage)
    write_json(output_dir / "training-module-candidate-selection.json", selection)
    write_json(output_dir / "training-module-candidates-lock-drift.json", candidate_drift)
    if promotion_plan is not None:
        write_json(Path(promotion_plan["promotionPlanFile"]), promotion_plan)
    summary = fast_gate_summary_payload(
        duration_ms=int((time.perf_counter() - started) * 1000),
        output_dir=output_dir,
        summary_file=summary_file,
        check_report=check_report,
        check_drift=check_drift,
        coverage=coverage,
        selection=selection,
        candidate_drift=candidate_drift,
        promotion_plan=promotion_plan,
    )
    write_json(summary_file, summary)

    print(f"Wrote training module checks report: {output_dir / 'training-module-checks.json'}")
    print(f"Wrote training module checks lock drift report: {output_dir / 'training-module-checks-lock-drift.json'}")
    print(f"Wrote training module coverage report: {output_dir / 'training-module-coverage.json'}")
    print(f"Wrote training module candidate selection report: {output_dir / 'training-module-candidate-selection.json'}")
    print(f"Wrote training module candidate lock drift report: {output_dir / 'training-module-candidates-lock-drift.json'}")
    if promotion_plan is not None:
        print(f"Wrote training module promotion plan: {promotion_plan['promotionPlanFile']}")
        if promotion_plan["applied"]:
            print("Promoted training module candidate(s): " + ", ".join(promotion_plan["selectedLabels"]))
        elif promotion_plan["dryRun"]:
            print("Promotion dry-run candidate(s): " + ", ".join(promotion_plan["selectedLabels"]))
    print(f"Wrote training module fast gate summary: {summary_file}")
    print(
        "Fast training metadata: "
        f"{check_report['checkCount']} hard check(s), "
        f"{coverage['projectCount']} project(s), "
        f"{coverage['candidateCheckCount']} candidate(s), "
        f"checks lock passed={str(check_drift['passed']).lower()}, "
        f"candidate lock passed={str(candidate_drift['passed']).lower()}"
    )
    if not check_drift["passed"]:
        print(
            "Training module checks lock validation failed. "
            "Run scripts/verify-training-advisory-gates-fast.py --write-check-lock "
            "when the hard-check registry change is intentional.",
            file=sys.stderr,
        )
        return 1
    if not candidate_drift["passed"]:
        print(
            "Training module candidate lock validation failed. "
            "Run scripts/verify-training-advisory-gates-fast.py --write-candidate-lock "
            "when the advisory candidate change is intentional.",
            file=sys.stderr,
        )
        return 1
    commit_file_transaction()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BaseException:
        rollback_active_file_transaction()
        raise
