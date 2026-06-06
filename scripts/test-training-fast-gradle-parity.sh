#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-training-gradle-parity.XXXXXX")"
START_EPOCH_SECONDS="$(date +%s)"

FAST_GATE="$ROOT_DIR/scripts/verify-training-advisory-gates-fast.py"
GRADLE_REPORT_DIR="$ROOT_DIR/build/reports/gollek"
PARITY_CANDIDATES="${GOLLEK_TRAINING_PARITY_CANDIDATES:-}"
PARITY_ARTIFACT_DIR="${GOLLEK_TRAINING_PARITY_ARTIFACT_DIR:-}"
SELECTED_CANDIDATES=""
FAILED_REPORT=""
COMPARED_REPORTS=()

SHARED_REPORTS=(
    training-module-checks.json
    training-module-checks-lock-drift.json
    training-module-coverage.json
    training-module-candidate-selection.json
    training-module-candidates-lock-drift.json
)

cleanup() {
    rm -rf "$TMP_DIR"
}

print_usage() {
    printf '%s\n' \
        "Usage: bash scripts/test-training-fast-gradle-parity.sh [options]" \
        "" \
        "Options:" \
        "  --candidate LABELS     Comma-separated advisory labels for selected-candidate parity." \
        "                         Defaults to GOLLEK_TRAINING_PARITY_CANDIDATES or the first" \
        "                         available advisory candidates from coverage." \
        "  --artifact-dir DIR     Keep raw reports, normalized JSON, diffs, logs, and summary" \
        "                         under DIR. Defaults to GOLLEK_TRAINING_PARITY_ARTIFACT_DIR." \
        "  -h, --help             Show this help."
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --candidate)
                if [[ "$#" -lt 2 || "$2" == --* ]]; then
                    echo "Missing value for --candidate" >&2
                    print_usage >&2
                    exit 2
                fi
                PARITY_CANDIDATES="$2"
                shift 2
                ;;
            --artifact-dir)
                if [[ "$#" -lt 2 || "$2" == --* ]]; then
                    echo "Missing value for --artifact-dir" >&2
                    print_usage >&2
                    exit 2
                fi
                PARITY_ARTIFACT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_usage >&2
                exit 2
                ;;
        esac
    done
    if [[ "$#" -gt 0 ]]; then
        echo "Unexpected positional argument: $1" >&2
        print_usage >&2
        exit 2
    fi
}

json_array_from_lines() {
    if [[ "$#" -eq 0 ]]; then
        printf '[]'
        return
    fi
    printf '%s\n' "$@" | jq -R . | jq -s .
}

prepare_artifact_dir() {
    if [[ -z "$PARITY_ARTIFACT_DIR" ]]; then
        return
    fi
    mkdir -p \
        "$PARITY_ARTIFACT_DIR/logs" \
        "$PARITY_ARTIFACT_DIR/raw" \
        "$PARITY_ARTIFACT_DIR/normalized" \
        "$PARITY_ARTIFACT_DIR/diffs"
}

snapshot_json_dir() {
    local source_dir="$1"
    local target_name="$2"
    shift 2
    if [[ -z "$PARITY_ARTIFACT_DIR" || ! -d "$source_dir" ]]; then
        return
    fi
    local target_dir="$PARITY_ARTIFACT_DIR/raw/$target_name"
    mkdir -p "$target_dir"
    if [[ "$#" -gt 0 ]]; then
        local report
        for report in "$@"; do
            if [[ -f "$source_dir/$report" ]]; then
                cp "$source_dir/$report" "$target_dir/"
            fi
        done
    else
        find "$source_dir" -maxdepth 1 -type f -name '*.json' -exec cp {} "$target_dir/" \;
    fi
}

write_artifact_readme() {
    local status="$1"
    local duration_seconds="$2"
    local passed="$3"
    local selected="${SELECTED_CANDIDATES:-none}"
    local failed="${FAILED_REPORT:-none}"
    if [[ -z "$failed" ]]; then
        failed="none"
    fi

    {
        printf '# Training Gradle Parity Artifacts\n\n'
        printf 'This bundle compares the Gradle-generated training gate reports with the fast metadata gate reports.\n\n'
        printf '## Result\n\n'
        printf -- '- Passed: `%s`\n' "$passed"
        printf -- '- Exit code: `%s`\n' "$status"
        printf -- '- Duration: `%ss`\n' "$duration_seconds"
        printf -- '- Selected candidates: `%s`\n' "$selected"
        printf -- '- Compared reports: `%s`\n' "${#COMPARED_REPORTS[@]}"
        printf -- '- Failed report: `%s`\n\n' "$failed"
        printf '## Files\n\n'
        printf -- '- `training-gradle-parity-summary.json`: machine-readable parity receipt\n'
        printf -- '- `raw/gradle-all`: Gradle reports for default candidate selection\n'
        printf -- '- `raw/fast-all`: fast-gate reports for default candidate selection\n'
        printf -- '- `raw/gradle-selected`: Gradle report for selected-candidate parity\n'
        printf -- '- `raw/fast-selected`: fast-gate reports for selected-candidate parity\n'
        printf -- '- `normalized`: `jq -S` normalized report pairs used for comparison\n'
        printf -- '- `diffs`: unified diffs for each compared report; empty files mean matching reports\n'
        printf -- '- `logs`: captured stdout/stderr from Gradle and fast-gate runs\n'
        if [[ "${#COMPARED_REPORTS[@]}" -gt 0 ]]; then
            printf '\n## Compared Reports\n\n'
            local report
            for report in "${COMPARED_REPORTS[@]}"; do
                printf -- '- `%s`\n' "$report"
            done
        fi
    } >"$PARITY_ARTIFACT_DIR/README.md"
}

write_artifact_summary() {
    local status="$1"
    if [[ -z "$PARITY_ARTIFACT_DIR" ]]; then
        return
    fi
    set +e
    prepare_artifact_dir
    find "$TMP_DIR" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) \
        -exec cp {} "$PARITY_ARTIFACT_DIR/logs/" \;

    local end_epoch_seconds
    end_epoch_seconds="$(date +%s)"
    local duration_seconds=$((end_epoch_seconds - START_EPOCH_SECONDS))
    local passed_json=false
    if [[ "$status" -eq 0 ]]; then
        passed_json=true
    fi
    local compared_reports_json="[]"
    if [[ "${#COMPARED_REPORTS[@]}" -gt 0 ]]; then
        compared_reports_json="$(json_array_from_lines "${COMPARED_REPORTS[@]}")"
    fi

    jq -n \
        --arg format "gollek.training.gradle-parity-summary.v1" \
        --arg selectedCandidates "$SELECTED_CANDIDATES" \
        --arg failedReport "$FAILED_REPORT" \
        --arg artifactRoot "$PARITY_ARTIFACT_DIR" \
        --argjson passed "$passed_json" \
        --argjson exitCode "$status" \
        --argjson durationSeconds "$duration_seconds" \
        --argjson comparedReports "$compared_reports_json" \
        '{
            format: $format,
            schemaVersion: 1,
            passed: $passed,
            exitCode: $exitCode,
            durationSeconds: $durationSeconds,
            selectedCandidates: $selectedCandidates,
            comparedReportCount: ($comparedReports | length),
            comparedReports: $comparedReports,
            failedReport: $failedReport,
            artifacts: {
                root: $artifactRoot,
                rawReports: "raw",
                normalizedReports: "normalized",
                diffs: "diffs",
                logs: "logs"
            }
        }' >"$PARITY_ARTIFACT_DIR/training-gradle-parity-summary.json"
    write_artifact_readme "$status" "$duration_seconds" "$passed_json"
    set -e
}

on_exit() {
    local status="$?"
    write_artifact_summary "$status" || true
    cleanup
}
trap on_exit EXIT

run_gradle_reports() {
    local log_prefix="$1"
    shift
    (
        cd "$ROOT_DIR"
        ./gradlew --no-daemon "$@"
    ) >"$TMP_DIR/$log_prefix.out" 2>"$TMP_DIR/$log_prefix.err" || {
        echo "Gradle report generation failed for $log_prefix" >&2
        cat "$TMP_DIR/$log_prefix.out" >&2
        cat "$TMP_DIR/$log_prefix.err" >&2
        exit 1
    }
}

run_fast_gate() {
    local log_prefix="$1"
    local output_dir="$2"
    shift 2
    "$FAST_GATE" --output-dir "$output_dir" "$@" >"$TMP_DIR/$log_prefix.out" 2>"$TMP_DIR/$log_prefix.err" || {
        echo "Fast training gate generation failed for $log_prefix" >&2
        cat "$TMP_DIR/$log_prefix.out" >&2
        cat "$TMP_DIR/$log_prefix.err" >&2
        exit 1
    }
}

assert_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo "Expected non-empty report: $path" >&2
        find "$TMP_DIR" -maxdepth 4 -type f -print >&2 || true
        exit 1
    fi
}

compare_json_report() {
    local label="$1"
    local gradle_file="$2"
    local fast_file="$3"
    local safe_label="${label%.json}"
    safe_label="${safe_label//[^A-Za-z0-9_.-]/_}"
    local gradle_normalized="$TMP_DIR/gradle-$safe_label.json"
    local fast_normalized="$TMP_DIR/fast-$safe_label.json"
    local diff_file="$TMP_DIR/diff-$safe_label.diff"

    assert_file "$gradle_file"
    assert_file "$fast_file"
    jq -S . "$gradle_file" >"$gradle_normalized"
    jq -S . "$fast_file" >"$fast_normalized"

    if [[ -n "$PARITY_ARTIFACT_DIR" ]]; then
        prepare_artifact_dir
        cp "$gradle_normalized" "$PARITY_ARTIFACT_DIR/normalized/gradle-$safe_label.json"
        cp "$fast_normalized" "$PARITY_ARTIFACT_DIR/normalized/fast-$safe_label.json"
    fi

    if ! diff -u "$gradle_normalized" "$fast_normalized" >"$diff_file"; then
        FAILED_REPORT="$label"
        cat "$diff_file" >&2
        if [[ -n "$PARITY_ARTIFACT_DIR" ]]; then
            cp "$diff_file" "$PARITY_ARTIFACT_DIR/diffs/$safe_label.diff"
        fi
        echo "Training fast gate Gradle parity failed for $label" >&2
        exit 1
    fi
    if [[ -n "$PARITY_ARTIFACT_DIR" ]]; then
        cp "$diff_file" "$PARITY_ARTIFACT_DIR/diffs/$safe_label.diff"
    fi
    COMPARED_REPORTS+=("$label")
}

choose_selected_candidates() {
    local coverage_report="$1"
    if [[ -n "$PARITY_CANDIDATES" ]]; then
        printf '%s\n' "$PARITY_CANDIDATES"
        return
    fi
    jq -r '.candidateChecks[0:2] | map(.label) | join(",")' "$coverage_report"
}

ALL_FAST_DIR="$TMP_DIR/fast-all"
SELECTED_FAST_DIR="$TMP_DIR/fast-selected"
parse_args "$@"
prepare_artifact_dir

run_gradle_reports \
    gradle-all \
    writeTrainingModuleChecksReport \
    writeTrainingModuleChecksLockDriftReport \
    writeTrainingModuleCoverageReport \
    writeTrainingModuleCandidateSelectionReport \
    writeTrainingModuleCandidateLockDriftReport
snapshot_json_dir "$GRADLE_REPORT_DIR" "gradle-all" "${SHARED_REPORTS[@]}"

run_fast_gate fast-all "$ALL_FAST_DIR"
snapshot_json_dir "$ALL_FAST_DIR" "fast-all"

for report in "${SHARED_REPORTS[@]}"; do
    compare_json_report "$report" "$GRADLE_REPORT_DIR/$report" "$ALL_FAST_DIR/$report"
done

SELECTED_CANDIDATES="$(choose_selected_candidates "$ALL_FAST_DIR/training-module-coverage.json")"
if [[ -n "$SELECTED_CANDIDATES" ]]; then
    run_gradle_reports \
        gradle-selected \
        "-PgollekTrainingCandidate=$SELECTED_CANDIDATES" \
        writeTrainingModuleCandidateSelectionReport
    snapshot_json_dir "$GRADLE_REPORT_DIR" "gradle-selected" training-module-candidate-selection.json

    run_fast_gate fast-selected "$SELECTED_FAST_DIR" --candidate "$SELECTED_CANDIDATES"
    snapshot_json_dir "$SELECTED_FAST_DIR" "fast-selected"

    compare_json_report \
        "selected-training-module-candidate-selection.json" \
        "$GRADLE_REPORT_DIR/training-module-candidate-selection.json" \
        "$SELECTED_FAST_DIR/training-module-candidate-selection.json"
else
    echo "No advisory candidates found; skipped selected-candidate parity check."
fi

echo "PASS: training fast gate Gradle parity"
