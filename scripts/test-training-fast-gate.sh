#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-training-fast-gate.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

FAST_GATE="$ROOT_DIR/scripts/verify-training-advisory-gates-fast.py"
PARITY_GATE="$ROOT_DIR/scripts/test-training-fast-gradle-parity.sh"

assert_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo "Expected non-empty file: $path" >&2
        find "$TMP_DIR" -maxdepth 3 -type f -print >&2 || true
        exit 1
    fi
}

assert_jq() {
    local file="$1"
    local expression="$2"
    if ! jq -e "$expression" "$file" >/dev/null; then
        echo "JSON assertion failed for $file: $expression" >&2
        jq . "$file" >&2 || true
        exit 1
    fi
}

prepare_apply_fixture() {
    local fixture_root="$1"
    mkdir -p \
        "$fixture_root/scripts" \
        "$fixture_root/buildSrc/src/main/kotlin" \
        "$fixture_root/gradle"
    cp "$FAST_GATE" "$fixture_root/scripts/verify-training-advisory-gates-fast.py"
    cp "$ROOT_DIR/settings.gradle.kts" "$fixture_root/settings.gradle.kts"
    cp "$ROOT_DIR/buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts" \
        "$fixture_root/buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts"
    cp "$ROOT_DIR/gradle/training-module-checks.lock.json" \
        "$fixture_root/gradle/training-module-checks.lock.json"
    cp "$ROOT_DIR/gradle/training-module-candidates.lock.json" \
        "$fixture_root/gradle/training-module-candidates.lock.json"

    for project_dir in \
        training/gollek-train-recursive-reasoning \
        training/gollek-nn-audio \
        training/gollek-nn-vision
    do
        mkdir -p "$fixture_root/$project_dir/src/test"
        printf 'plugins {}\n' > "$fixture_root/$project_dir/build.gradle.kts"
    done
}

ALL_DIR="$TMP_DIR/all"
"$FAST_GATE" --output-dir "$ALL_DIR" >"$TMP_DIR/all.out" 2>"$TMP_DIR/all.err"

assert_file "$ALL_DIR/training-module-checks.json"
assert_file "$ALL_DIR/training-module-checks-lock-drift.json"
assert_file "$ALL_DIR/training-module-coverage.json"
assert_file "$ALL_DIR/training-module-candidate-selection.json"
assert_file "$ALL_DIR/training-module-candidates-lock-drift.json"
assert_file "$ALL_DIR/training-module-fast-gate-summary.json"

assert_jq "$ALL_DIR/training-module-fast-gate-summary.json" \
    '.format == "gollek.training.fast-gate-summary.v1"
        and .passed == true
        and .counts.hardCheckCount >= 1
        and .counts.projectCount >= .counts.testProjectCount
        and .counts.testProjectCount >= .counts.registeredProjectCount
        and .counts.candidateCount == .counts.selectedCandidateCount
        and .locks.checks.passed == true
        and .locks.candidates.passed == true
        and (.reports.summary | endswith("/training-module-fast-gate-summary.json"))
        and (.repairCommands.checksLock | contains("--write-check-lock"))
        and (.repairCommands.candidatesLock | contains("--write-candidate-lock"))'

assert_jq "$ALL_DIR/training-module-checks.json" \
    '.format == "gollek.training.module-checks.v2"
        and .checkCount >= 1
        and (.checks | length) == .checkCount
        and (.checks[] | select(.label == "recursive-reasoning" and .projectExists == true and .taskExists == true))'

assert_jq "$ALL_DIR/training-module-coverage.json" \
    '.format == "gollek.training.module-coverage.v2"
        and .projectCount == (.projects | length)
        and .candidateCheckCount == (.candidateChecks | length)
        and (.candidateChecks[] | select(.label == "audio" and .taskPath == ":ml:gollek-ml-audio:test"))
        and (.candidateChecks[] | select(.label == "vision" and .taskPath == ":ml:gollek-ml-vision:test"))'

assert_jq "$ALL_DIR/training-module-checks-lock-drift.json" \
    '.format == "gollek.training.module-checks.lock-drift.v1"
        and .passed == true
        and .missingLock == false
        and .fingerprintDrift == false
        and (.addedLabels | length) == 0
        and (.removedLabels | length) == 0
        and (.changedChecks | length) == 0'

assert_jq "$ALL_DIR/training-module-candidates-lock-drift.json" \
    '.format == "gollek.training.module-candidates.lock-drift.v1"
        and .passed == true
        and .missingLock == false
        and .fingerprintDrift == false
        and (.addedLabels | length) == 0
        and (.removedLabels | length) == 0
        and (.changedCandidates | length) == 0'

SELECTED_DIR="$TMP_DIR/selected"
SELECTED_SUMMARY="$TMP_DIR/selected-summary.json"
"$FAST_GATE" \
    --candidate audio,vision \
    --output-dir "$SELECTED_DIR" \
    --summary-file "$SELECTED_SUMMARY" >"$TMP_DIR/selected.out" 2>"$TMP_DIR/selected.err"

assert_file "$SELECTED_SUMMARY"
assert_file "$SELECTED_DIR/training-module-candidate-selection.json"
if [[ -e "$SELECTED_DIR/training-module-fast-gate-summary.json" ]]; then
    echo "Expected --summary-file to move the summary out of the report directory" >&2
    exit 1
fi

assert_jq "$SELECTED_SUMMARY" \
    '.passed == true
        and .selectionValue == "audio,vision"
        and .selectedLabels == ["audio", "vision"]
        and .counts.candidateCount >= .counts.selectedCandidateCount
        and .counts.selectedCandidateCount == 2
        and .selectedTaskPaths == [":ml:gollek-ml-audio:test", ":ml:gollek-ml-vision:test"]
        and (.reports.summary | endswith("/selected-summary.json"))'

assert_jq "$SELECTED_DIR/training-module-candidate-selection.json" \
    '.format == "gollek.training.module-candidate-selection.v2"
        and .selectedAll == false
        and .selectedLabels == ["audio", "vision"]
        and .candidateCount == 2
        and .taskPaths == [":ml:gollek-ml-audio:test", ":ml:gollek-ml-vision:test"]'

PROMOTION_DIR="$TMP_DIR/promotion"
PROMOTION_PLAN="$TMP_DIR/promotion-plan.json"
"$FAST_GATE" \
    --candidate audio,vision \
    --promote-selected \
    --dry-run \
    --output-dir "$PROMOTION_DIR" \
    --promotion-plan-file "$PROMOTION_PLAN" >"$TMP_DIR/promotion.out" 2>"$TMP_DIR/promotion.err"

assert_file "$PROMOTION_PLAN"
assert_file "$PROMOTION_DIR/training-module-fast-gate-summary.json"
assert_jq "$PROMOTION_PLAN" \
    '.format == "gollek.training.promotion-plan.v1"
        and .promotionFingerprintAlgorithm == "SHA-256"
        and (.promotionFingerprint | test("^[0-9a-f]{64}$"))
        and .selectionValue == "audio,vision"
        and .selectedLabels == ["audio", "vision"]
        and .candidateCount == 2
        and .dryRun == true
        and .applied == false
        and (.promotionSnippets | length) == 2
        and (.commands.runCandidateTests | contains("-PgollekTrainingCandidate=audio,vision"))'

assert_jq "$PROMOTION_DIR/training-module-fast-gate-summary.json" \
    '.promotion.requested == true
        and .promotion.dryRun == true
        and .promotion.applied == false
        and .promotion.selectedLabels == ["audio", "vision"]
        and (.reports.promotionPlan | endswith("/promotion-plan.json"))'

PROMOTION_VALIDATION_REPORT="$TMP_DIR/promotion-validation-report.json"
"$FAST_GATE" \
    --validate-promotion-plan "$PROMOTION_PLAN" \
    --promotion-plan-validation-report-file "$PROMOTION_VALIDATION_REPORT" \
    >"$TMP_DIR/promotion-validate.out" 2>"$TMP_DIR/promotion-validate.err"
if ! grep -q "Validated training module promotion plan" "$TMP_DIR/promotion-validate.out"; then
    echo "Expected promotion plan validation success message" >&2
    cat "$TMP_DIR/promotion-validate.out" >&2
    cat "$TMP_DIR/promotion-validate.err" >&2
    exit 1
fi
assert_jq "$PROMOTION_VALIDATION_REPORT" \
    '.format == "gollek.training.promotion-plan-validation.v1"
        and .passed == true
        and .error == ""
        and .candidateCount == 2
        and .selectedLabels == ["audio", "vision"]
        and .promotionFingerprintAlgorithm == "SHA-256"
        and (.promotionFingerprint | test("^[0-9a-f]{64}$"))
        and .dryRun == true
        and .applied == false
        and .transactional == false'

BAD_PROMOTION_PLAN="$TMP_DIR/bad-promotion-plan.json"
jq '.selectedTaskPaths = []' "$PROMOTION_PLAN" > "$BAD_PROMOTION_PLAN"
BAD_PROMOTION_VALIDATION_REPORT="$TMP_DIR/bad-promotion-validation-report.json"
if "$FAST_GATE" \
        --validate-promotion-plan "$BAD_PROMOTION_PLAN" \
        --promotion-plan-validation-report-file "$BAD_PROMOTION_VALIDATION_REPORT" \
        >"$TMP_DIR/bad-promotion-plan.out" 2>"$TMP_DIR/bad-promotion-plan.err"; then
    echo "Expected malformed promotion plan validation to fail" >&2
    exit 1
fi
if ! grep -q "selectedTaskPaths" "$TMP_DIR/bad-promotion-plan.err"; then
    echo "Expected malformed promotion plan failure to mention selectedTaskPaths" >&2
    cat "$TMP_DIR/bad-promotion-plan.out" >&2
    cat "$TMP_DIR/bad-promotion-plan.err" >&2
    exit 1
fi
assert_jq "$BAD_PROMOTION_VALIDATION_REPORT" \
    '.format == "gollek.training.promotion-plan-validation.v1"
        and .passed == false
        and (.error | contains("selectedTaskPaths"))
        and .candidateCount == 0
        and .selectedLabels == []'

BAD_PROMOTION_FINGERPRINT_PLAN="$TMP_DIR/bad-promotion-fingerprint-plan.json"
jq '.promotionFingerprint = "0000000000000000000000000000000000000000000000000000000000000000"' \
    "$PROMOTION_PLAN" > "$BAD_PROMOTION_FINGERPRINT_PLAN"
if "$FAST_GATE" --validate-promotion-plan "$BAD_PROMOTION_FINGERPRINT_PLAN" >"$TMP_DIR/bad-promotion-fingerprint.out" 2>"$TMP_DIR/bad-promotion-fingerprint.err"; then
    echo "Expected stale promotion fingerprint validation to fail" >&2
    exit 1
fi
if ! grep -q "promotionFingerprint" "$TMP_DIR/bad-promotion-fingerprint.err"; then
    echo "Expected stale promotion fingerprint failure to mention promotionFingerprint" >&2
    cat "$TMP_DIR/bad-promotion-fingerprint.out" >&2
    cat "$TMP_DIR/bad-promotion-fingerprint.err" >&2
    exit 1
fi

python3 - "$ROOT_DIR" "$TMP_DIR" <<'PY'
import importlib.util
import copy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root = Path(sys.argv[1])
tmp = Path(sys.argv[2])
module_path = root / "scripts/verify-training-advisory-gates-fast.py"
spec = importlib.util.spec_from_file_location("fast_gate", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

existing = tmp / "rollback-existing.txt"
missing = tmp / "rollback-missing.txt"
existing.write_text("original", encoding="utf-8")
snapshots = module.snapshot_files([existing, missing])

existing.write_text("changed", encoding="utf-8")
missing.write_text("created", encoding="utf-8")
module.restore_file_snapshots(snapshots)

assert existing.read_text(encoding="utf-8") == "original"
assert not missing.exists()

candidate = {
    "label": "audio",
    "projectPath": ":ml:gollek-ml-audio",
    "taskName": "test",
    "taskPath": ":ml:gollek-ml-audio:test",
}
plan = module.promotion_plan_payload(
    selection_value="audio",
    selected_candidates=[candidate],
    promotion_plan_file=tmp / "contract-plan.json",
    dry_run=True,
    applied=False,
)
module.require_promotion_plan_contract(plan)

bad_plan = copy.deepcopy(plan)
bad_plan["selectedTaskPaths"] = []
try:
    module.require_promotion_plan_contract(bad_plan)
except SystemExit as exc:
    assert "selectedTaskPaths" in str(exc)
else:
    raise AssertionError("Expected malformed promotion plan to fail contract validation")
PY

APPLY_ROOT="$TMP_DIR/apply-fixture"
APPLY_FAST_GATE="$APPLY_ROOT/scripts/verify-training-advisory-gates-fast.py"
prepare_apply_fixture "$APPLY_ROOT"

APPLY_DIR="$TMP_DIR/apply-output"
APPLY_PLAN="$TMP_DIR/apply-plan.json"
"$APPLY_FAST_GATE" \
    --candidate audio \
    --promote-selected \
    --output-dir "$APPLY_DIR" \
    --promotion-plan-file "$APPLY_PLAN" >"$TMP_DIR/apply.out" 2>"$TMP_DIR/apply.err"

assert_file "$APPLY_PLAN"
assert_file "$APPLY_DIR/training-module-fast-gate-summary.json"
assert_jq "$APPLY_PLAN" \
    '.format == "gollek.training.promotion-plan.v1"
        and .promotionFingerprintAlgorithm == "SHA-256"
        and (.promotionFingerprint | test("^[0-9a-f]{64}$"))
        and .selectedLabels == ["audio"]
        and .dryRun == false
        and .applied == true
        and .transactional == true
        and (.mutationFiles | length) == 3'

assert_jq "$APPLY_DIR/training-module-fast-gate-summary.json" \
    '.passed == true
        and .promotion.requested == true
        and .promotion.applied == true
        and .promotion.dryRun == false
        and .counts.hardCheckCount == 2
        and .counts.candidateCount == 1'

assert_jq "$APPLY_DIR/training-module-checks.json" \
    '.checkCount == 2
        and (.checks[] | select(.label == "audio"
            and .projectPath == ":ml:gollek-ml-audio"
            and .projectExists == true
            and .taskExists == true))'

assert_jq "$APPLY_DIR/training-module-coverage.json" \
    '.candidateCheckCount == 1
        and .candidateChecks == [{
            "label": "vision",
            "projectPath": ":ml:gollek-ml-vision",
            "taskName": "test",
            "taskPath": ":ml:gollek-ml-vision:test"
        }]'

assert_jq "$APPLY_ROOT/gradle/training-module-checks.lock.json" \
    '.checkCount == 2
        and (.checks[] | select(.label == "audio" and .taskPath == ":ml:gollek-ml-audio:test"))'

assert_jq "$APPLY_ROOT/gradle/training-module-candidates.lock.json" \
    '.candidateCount == 1
        and .candidates[0].label == "vision"'

if ! grep -q 'label = "audio"' "$APPLY_ROOT/buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts"; then
    echo "Expected apply fixture registry to include promoted audio check" >&2
    exit 1
fi

ROLLBACK_ROOT="$TMP_DIR/rollback-fixture"
ROLLBACK_FAST_GATE="$ROLLBACK_ROOT/scripts/verify-training-advisory-gates-fast.py"
prepare_apply_fixture "$ROLLBACK_ROOT"
ROLLBACK_REGISTRY="$ROLLBACK_ROOT/buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts"
ROLLBACK_CHECK_LOCK="$ROLLBACK_ROOT/gradle/training-module-checks.lock.json"
ROLLBACK_CANDIDATE_LOCK="$ROLLBACK_ROOT/gradle/training-module-candidates.lock.json"
cp "$ROLLBACK_REGISTRY" "$TMP_DIR/rollback-registry.before"
cp "$ROLLBACK_CHECK_LOCK" "$TMP_DIR/rollback-check-lock.before"
cp "$ROLLBACK_CANDIDATE_LOCK" "$TMP_DIR/rollback-candidate-lock.before"

ROLLBACK_OUTPUT_FILE="$TMP_DIR/rollback-output-is-file"
printf 'not a directory\n' > "$ROLLBACK_OUTPUT_FILE"
if "$ROLLBACK_FAST_GATE" \
        --candidate audio \
        --promote-selected \
        --output-dir "$ROLLBACK_OUTPUT_FILE" \
        --promotion-plan-file "$TMP_DIR/rollback-plan.json" >"$TMP_DIR/rollback.out" 2>"$TMP_DIR/rollback.err"; then
    echo "Expected apply fixture promotion to fail when output-dir is a file" >&2
    exit 1
fi

if ! grep -q "Rolled back training promotion file changes." "$TMP_DIR/rollback.err"; then
    echo "Expected rollback failure path to report file rollback" >&2
    cat "$TMP_DIR/rollback.out" >&2
    cat "$TMP_DIR/rollback.err" >&2
    exit 1
fi
if ! cmp -s "$ROLLBACK_REGISTRY" "$TMP_DIR/rollback-registry.before"; then
    echo "Expected rollback fixture registry to be restored" >&2
    diff -u "$TMP_DIR/rollback-registry.before" "$ROLLBACK_REGISTRY" >&2 || true
    exit 1
fi
if ! cmp -s "$ROLLBACK_CHECK_LOCK" "$TMP_DIR/rollback-check-lock.before"; then
    echo "Expected rollback fixture check lock to be restored" >&2
    diff -u "$TMP_DIR/rollback-check-lock.before" "$ROLLBACK_CHECK_LOCK" >&2 || true
    exit 1
fi
if ! cmp -s "$ROLLBACK_CANDIDATE_LOCK" "$TMP_DIR/rollback-candidate-lock.before"; then
    echo "Expected rollback fixture candidate lock to be restored" >&2
    diff -u "$TMP_DIR/rollback-candidate-lock.before" "$ROLLBACK_CANDIDATE_LOCK" >&2 || true
    exit 1
fi
if grep -q 'label = "audio"' "$ROLLBACK_REGISTRY"; then
    echo "Expected rollback fixture registry to exclude failed audio promotion" >&2
    exit 1
fi
if [[ -e "$TMP_DIR/rollback-plan.json" ]]; then
    echo "Expected failed rollback promotion not to leave a promotion plan" >&2
    exit 1
fi

if "$FAST_GATE" --candidate does-not-exist --output-dir "$TMP_DIR/bad" >"$TMP_DIR/bad.out" 2>"$TMP_DIR/bad.err"; then
    echo "Expected unknown advisory candidate selection to fail" >&2
    exit 1
fi
if ! grep -q "Unknown advisory training module candidate" "$TMP_DIR/bad.err"; then
    echo "Expected unknown candidate failure message" >&2
    cat "$TMP_DIR/bad.out" >&2
    cat "$TMP_DIR/bad.err" >&2
    exit 1
fi

"$FAST_GATE" --help >"$TMP_DIR/help.out"
if ! grep -q -- "--summary-file" "$TMP_DIR/help.out" \
        || ! grep -q -- "--promotion-plan-file" "$TMP_DIR/help.out" \
        || ! grep -q -- "--validate-promotion-plan" "$TMP_DIR/help.out" \
        || ! grep -q -- "--promotion-plan-validation-report-file" "$TMP_DIR/help.out" \
        || ! grep -q -- "--promote-selected" "$TMP_DIR/help.out" \
        || ! grep -q -- "--dry-run" "$TMP_DIR/help.out" \
        || ! grep -q -- "--write-check-lock" "$TMP_DIR/help.out" \
        || ! grep -q -- "--write-candidate-lock" "$TMP_DIR/help.out"; then
    echo "Expected fast training gate help to list summary and lock options" >&2
    cat "$TMP_DIR/help.out" >&2
    exit 1
fi

"$PARITY_GATE" --help >"$TMP_DIR/parity-help.out"
if ! grep -q -- "--candidate" "$TMP_DIR/parity-help.out" \
        || ! grep -q -- "--artifact-dir" "$TMP_DIR/parity-help.out"; then
    echo "Expected parity help to list candidate and artifact options" >&2
    cat "$TMP_DIR/parity-help.out" >&2
    exit 1
fi

if "$PARITY_GATE" --candidate >"$TMP_DIR/parity-missing-candidate.out" 2>"$TMP_DIR/parity-missing-candidate.err"; then
    echo "Expected parity script to fail when --candidate has no value" >&2
    exit 1
fi
if ! grep -q "Missing value for --candidate" "$TMP_DIR/parity-missing-candidate.err"; then
    echo "Expected parity missing-candidate failure message" >&2
    cat "$TMP_DIR/parity-missing-candidate.out" >&2
    cat "$TMP_DIR/parity-missing-candidate.err" >&2
    exit 1
fi

if "$PARITY_GATE" --unknown-option >"$TMP_DIR/parity-unknown.out" 2>"$TMP_DIR/parity-unknown.err"; then
    echo "Expected parity script to fail for unknown options" >&2
    exit 1
fi
if ! grep -q "Unknown option: --unknown-option" "$TMP_DIR/parity-unknown.err"; then
    echo "Expected parity unknown-option failure message" >&2
    cat "$TMP_DIR/parity-unknown.out" >&2
    cat "$TMP_DIR/parity-unknown.err" >&2
    exit 1
fi

PARITY_ARTIFACT_DIR="$TMP_DIR/parity-artifacts"
if "$PARITY_GATE" \
        --artifact-dir "$PARITY_ARTIFACT_DIR" \
        --unknown-option >"$TMP_DIR/parity-artifact-failure.out" 2>"$TMP_DIR/parity-artifact-failure.err"; then
    echo "Expected parity script artifact failure path to fail for unknown options" >&2
    exit 1
fi
assert_file "$PARITY_ARTIFACT_DIR/training-gradle-parity-summary.json"
assert_file "$PARITY_ARTIFACT_DIR/README.md"
assert_jq "$PARITY_ARTIFACT_DIR/training-gradle-parity-summary.json" \
    '.format == "gollek.training.gradle-parity-summary.v1"
        and .schemaVersion == 1
        and .passed == false
        and .exitCode == 2
        and .comparedReportCount == 0
        and .comparedReports == []
        and .failedReport == ""'
if ! grep -q "Training Gradle Parity Artifacts" "$PARITY_ARTIFACT_DIR/README.md" \
        || ! grep -q "training-gradle-parity-summary.json" "$PARITY_ARTIFACT_DIR/README.md"; then
    echo "Expected parity artifact README to explain the bundle" >&2
    cat "$PARITY_ARTIFACT_DIR/README.md" >&2
    exit 1
fi

echo "PASS: training fast gate metadata contract"
