package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CI-friendly gate for enforcing trainer runtime-efficiency budgets.
 */
public final class TrainingReportRuntimeEfficiencyGate {
    private TrainingReportRuntimeEfficiencyGate() {
    }

    public record Policy(
            double minAccountedWallPercent,
            double maxWallClockOverheadPercent,
            double maxBottleneckPercent) {
        public Policy {
            minAccountedWallPercent = positiveOrDefault(minAccountedWallPercent, 75.0);
            maxWallClockOverheadPercent = positiveOrDefault(maxWallClockOverheadPercent, 35.0);
            maxBottleneckPercent = positiveOrDefault(maxBottleneckPercent, 75.0);
        }

        public static Policy defaults() {
            return new Policy(75.0, 35.0, 75.0);
        }

        public static Policy strict() {
            return new Policy(85.0, 25.0, 65.0);
        }

        public static Policy permissive() {
            return new Policy(50.0, 60.0, 90.0);
        }

        public static Policy fromMap(Map<String, ?> policy) {
            if (policy == null || policy.isEmpty()) {
                return defaults();
            }
            Policy defaults = defaults();
            return new Policy(
                    doubleEntry(policy, "minAccountedWallPercent", defaults.minAccountedWallPercent()),
                    doubleEntry(policy, "maxWallClockOverheadPercent", defaults.maxWallClockOverheadPercent()),
                    doubleEntry(policy, "maxBottleneckPercent", defaults.maxBottleneckPercent()));
        }

        public Policy withMinAccountedWallPercent(double threshold) {
            return new Policy(threshold, maxWallClockOverheadPercent, maxBottleneckPercent);
        }

        public Policy withMaxWallClockOverheadPercent(double threshold) {
            return new Policy(minAccountedWallPercent, threshold, maxBottleneckPercent);
        }

        public Policy withMaxBottleneckPercent(double threshold) {
            return new Policy(minAccountedWallPercent, maxWallClockOverheadPercent, threshold);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("minAccountedWallPercent", minAccountedWallPercent);
            map.put("maxWallClockOverheadPercent", maxWallClockOverheadPercent);
            map.put("maxBottleneckPercent", maxBottleneckPercent);
            return Map.copyOf(map);
        }
    }

    public record Finding(
            String code,
            String severity,
            String message,
            String action,
            Map<String, Object> evidence) {
        public Finding {
            code = code == null || code.isBlank() ? "runtime-efficiency" : code.trim();
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
            Policy policy,
            boolean available,
            List<Finding> findings,
            Map<String, Object> efficiency) {
        public Result {
            policy = policy == null ? Policy.defaults() : policy;
            findings = findings == null ? List.of() : List.copyOf(findings);
            efficiency = efficiency == null ? Map.of() : Map.copyOf(efficiency);
        }

        public boolean passed() {
            return findings.isEmpty();
        }

        public String message() {
            if (!available) {
                return "Runtime efficiency metadata is not available.";
            }
            if (passed()) {
                return "Runtime efficiency gate passed.";
            }
            return "Runtime efficiency gate found " + findings.size() + " budget warning(s).";
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
            map.put("policy", policy.toMap());
            map.put("findingCount", findings.size());
            map.put("findings", findings.stream().map(Finding::toMap).toList());
            map.put("efficiency", efficiency);
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportRuntimeEfficiencyGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportRuntimeEfficiencyGateJUnitXml.render(this);
        }
    }

    public static Result evaluate(TrainingReport report) {
        return evaluate(report, Policy.defaults());
    }

    public static Result evaluate(TrainingReport report, Policy policy) {
        Objects.requireNonNull(report, "report must not be null");
        Policy resolvedPolicy = policy == null ? Policy.defaults() : policy;
        TrainingReportRuntimeEfficiency efficiency = report.runtimeEfficiency();
        if (!efficiency.available()) {
            return new Result(resolvedPolicy, false, List.of(), efficiency.toMap());
        }
        List<Finding> findings = new ArrayList<>();
        addAccountedWallFinding(findings, efficiency, resolvedPolicy);
        addWallClockOverheadFinding(findings, efficiency, resolvedPolicy);
        addBottleneckFinding(findings, efficiency, resolvedPolicy);
        return new Result(resolvedPolicy, true, findings, efficiency.toMap());
    }

    public static Result evaluate(Map<String, ?> report) {
        return evaluate(TrainingReport.of(report), Policy.defaults());
    }

    public static Result evaluate(Map<String, ?> report, Policy policy) {
        return evaluate(TrainingReport.of(report), policy);
    }

    private static void addAccountedWallFinding(
            List<Finding> findings,
            TrainingReportRuntimeEfficiency efficiency,
            Policy policy) {
        efficiency.accountedPercent().ifPresent(percent -> {
            if (percent >= policy.minAccountedWallPercent()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-efficiency-low-accounted-wall-time",
                    "warning",
                    "Only " + format(percent) + "% of wall-clock trainer time is accounted for by measured phases.",
                    "Inspect unprofiled trainer orchestration, callbacks, logging, and data movement around the hot scopes.",
                    evidence(
                            "accountedPercent",
                            percent,
                            "thresholdPercent",
                            policy.minAccountedWallPercent(),
                            efficiency)));
        });
    }

    private static void addWallClockOverheadFinding(
            List<Finding> findings,
            TrainingReportRuntimeEfficiency efficiency,
            Policy policy) {
        efficiency.overheadPercent().ifPresent(percent -> {
            if (percent <= policy.maxWallClockOverheadPercent()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-efficiency-wall-clock-overhead",
                    percent >= 50.0 ? "failure" : "warning",
                    "Wall-clock overhead in `" + efficiency.overheadScope() + "` is " + format(percent) + "%.",
                    "Reduce unprofiled work in `" + efficiency.overheadScope()
                            + "` before tuning tensor kernels or optimizer math.",
                    evidence(
                            "overheadPercent",
                            percent,
                            "thresholdPercent",
                            policy.maxWallClockOverheadPercent(),
                            efficiency)));
        });
    }

    private static void addBottleneckFinding(
            List<Finding> findings,
            TrainingReportRuntimeEfficiency efficiency,
            Policy policy) {
        efficiency.bottleneckPercent().ifPresent(percent -> {
            if (percent <= policy.maxBottleneckPercent()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-efficiency-dominant-bottleneck",
                    percent >= 85.0 ? "failure" : "warning",
                    "Runtime is concentrated in `" + efficiency.bottleneck() + "` at " + format(percent) + "%.",
                    "Break down `" + efficiency.bottleneck()
                            + "` with the runtime profile action plan before broad trainer rewrites.",
                    evidence(
                            "bottleneckPercent",
                            percent,
                            "thresholdPercent",
                            policy.maxBottleneckPercent(),
                            efficiency)));
        });
    }

    private static Map<String, Object> evidence(
            String metricKey,
            double metricValue,
            String thresholdKey,
            double thresholdValue,
            TrainingReportRuntimeEfficiency efficiency) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put(metricKey, metricValue);
        evidence.put(thresholdKey, thresholdValue);
        evidence.put("status", efficiency.status().name());
        evidence.put("overheadScope", efficiency.overheadScope());
        evidence.put("bottleneck", efficiency.bottleneck());
        efficiency.measuredMillis().ifPresent(value -> evidence.put("measuredMillis", value));
        efficiency.wallMillis().ifPresent(value -> evidence.put("wallMillis", value));
        efficiency.overheadMillis().ifPresent(value -> evidence.put("overheadMillis", value));
        evidence.put("primaryHotspot", efficiency.primaryHotspot());
        return Map.copyOf(evidence);
    }

    private static double doubleEntry(Map<String, ?> source, String key, double fallback) {
        var parsed = optionalDouble(source.get(key));
        return parsed.isPresent() ? parsed.getAsDouble() : fallback;
    }

    private static double positiveOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value > 0.0 ? value : fallback;
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.ROOT, "%.3f", value) : "n/a";
    }
}
