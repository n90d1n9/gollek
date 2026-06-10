package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CI-friendly gate for blocking trainer runtime regressions against a baseline report.
 */
public final class TrainingReportRuntimeRegressionGate {
    private TrainingReportRuntimeRegressionGate() {
    }

    public record Finding(
            String code,
            String severity,
            String message,
            String action,
            Map<String, Object> evidence) {
        public Finding {
            code = code == null || code.isBlank() ? "runtime-regression" : code.trim();
            severity = severity == null || severity.isBlank() ? "warning" : severity.trim();
            message = message == null ? "" : message.trim();
            action = action == null ? "" : action.trim();
            evidence = evidence == null ? Map.of() : Map.copyOf(evidence);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("code", code);
            map.put("severity", severity);
            map.put("message", message);
            map.put("action", action);
            map.put("evidence", evidence);
            return Map.copyOf(map);
        }
    }

    public record Result(
            boolean available,
            List<Finding> findings,
            TrainingReportRuntimeRegressionSummary runtimeRegression) {
        public Result {
            findings = findings == null ? List.of() : List.copyOf(findings);
            runtimeRegression = runtimeRegression == null
                    ? TrainingReportRuntimeRegressionSummary.empty()
                    : runtimeRegression;
            available = available || runtimeRegression.available();
        }

        public boolean passed() {
            return findings.isEmpty();
        }

        public String message() {
            if (!available) {
                return "Runtime regression metadata is not available.";
            }
            if (passed()) {
                return "Runtime regression gate passed.";
            }
            return "Runtime regression gate found " + findings.size() + " regression warning(s).";
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available);
            map.put("passed", passed());
            map.put("message", message());
            map.put("findingCount", findings.size());
            map.put("findings", findings.stream().map(Finding::toMap).toList());
            map.put("runtimeRegression", runtimeRegression.toMap());
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportRuntimeRegressionGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportRuntimeRegressionGateJUnitXml.render(this);
        }
    }

    public static Result evaluate(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return evaluate(TrainingReportAdvisor.runtimeRegressionSummary(baselineReport, candidateReport));
    }

    public static Result evaluate(
            TrainingReport baseline,
            TrainingReport candidate) {
        Objects.requireNonNull(baseline, "baseline must not be null");
        Objects.requireNonNull(candidate, "candidate must not be null");
        return evaluate(baseline.payload(), candidate.payload());
    }

    public static Result evaluate(TrainingReportRuntimeRegressionSummary summary) {
        TrainingReportRuntimeRegressionSummary resolved =
                summary == null ? TrainingReportRuntimeRegressionSummary.empty() : summary;
        List<Finding> findings = new ArrayList<>();
        resolved.primaryGroupAverage()
                .filter(TrainingReportRuntimeRegressionSummary.Entry::regressed)
                .ifPresent(entry -> findings.add(timingFinding(
                        "runtime-regression-primary-group-average",
                        "Primary runtime group `" + entry.key() + "` average regressed.",
                        "Compare child phases in `" + entry.key() + "` against the baseline before broad trainer changes.",
                        entry)));
        resolved.primaryHotspotAverage()
                .filter(TrainingReportRuntimeRegressionSummary.Entry::regressed)
                .ifPresent(entry -> findings.add(timingFinding(
                        "runtime-regression-primary-hotspot-average",
                        "Primary runtime hotspot `" + entry.key() + "` average regressed.",
                        "Inspect the candidate phase `" + entry.key() + "` with the same seed and data order.",
                        entry)));
        resolved.accountedWallTime()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .ifPresent(entry -> findings.add(efficiencyFinding(
                        "runtime-regression-accounted-wall-time",
                        "Candidate runtime profile accounts for less wall-clock time.",
                        "Add scoped profiling around new callbacks, logging, checkpointing, and trainer loop glue.",
                        entry)));
        resolved.wallClockOverhead()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .ifPresent(entry -> findings.add(efficiencyFinding(
                        "runtime-regression-wall-clock-overhead",
                        "Candidate wall-clock overhead regressed in `" + entry.key() + "`.",
                        "Compare candidate overhead scope timing against the baseline before changing model math.",
                        entry)));
        resolved.dominantBottleneck()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .ifPresent(entry -> findings.add(efficiencyFinding(
                        "runtime-regression-dominant-bottleneck",
                        "Candidate bottleneck concentration regressed in `" + entry.key() + "`.",
                        "Break down `" + entry.key() + "` child phases and add a focused regression benchmark.",
                        entry)));
        return new Result(resolved.available(), findings, resolved);
    }

    private static Finding timingFinding(
            String code,
            String message,
            String action,
            TrainingReportRuntimeRegressionSummary.Entry entry) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("key", entry.key());
        evidence.put("kind", entry.kind());
        evidence.put("baselineAverageMillis", entry.baselineAverageMillis());
        evidence.put("candidateAverageMillis", entry.candidateAverageMillis());
        evidence.put("ratio", entry.ratio());
        evidence.put("threshold", entry.threshold());
        return new Finding(code, "warning", message, action, evidence);
    }

    private static Finding efficiencyFinding(
            String code,
            String message,
            String action,
            TrainingReportRuntimeRegressionSummary.EfficiencyEntry entry) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("key", entry.key());
        evidence.put("kind", entry.kind());
        evidence.put("direction", entry.direction());
        evidence.put("baselineValue", entry.baselineValue());
        evidence.put("candidateValue", entry.candidateValue());
        evidence.put("delta", entry.delta());
        evidence.put("threshold", entry.threshold());
        evidence.put("unit", entry.unit());
        return new Finding(code, "warning", message, action, evidence);
    }
}
