package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CI-friendly gate for trainer acceleration and throughput performance health.
 */
public final class TrainingReportPerformanceGate {
    private TrainingReportPerformanceGate() {
    }

    public record Policy(
            boolean failOnAcceleratorFallback,
            double minTrainSamplesPerSecond,
            double maxValidationToTrainAverageBatchMillisRatio) {
        public Policy {
            minTrainSamplesPerSecond = nonNegativeOrDefault(minTrainSamplesPerSecond, 1.0e-9);
            maxValidationToTrainAverageBatchMillisRatio =
                    positiveOrDefault(maxValidationToTrainAverageBatchMillisRatio, 2.0);
        }

        public static Policy defaults() {
            return new Policy(true, 1.0e-9, 2.0);
        }

        public static Policy strict() {
            return new Policy(true, 1.0, 1.5);
        }

        public static Policy permissive() {
            return new Policy(false, 0.0, 5.0);
        }

        public static Policy fromMap(Map<String, ?> policy) {
            if (policy == null || policy.isEmpty()) {
                return defaults();
            }
            Policy defaults = defaults();
            return new Policy(
                    TrainingReportMapValues.booleanValue(
                            policy,
                            "failOnAcceleratorFallback",
                            defaults.failOnAcceleratorFallback()),
                    TrainingReportValues.optionalDouble(policy.get("minTrainSamplesPerSecond"))
                            .orElse(defaults.minTrainSamplesPerSecond()),
                    TrainingReportValues.optionalDouble(policy.get("maxValidationToTrainAverageBatchMillisRatio"))
                            .orElse(defaults.maxValidationToTrainAverageBatchMillisRatio()));
        }

        public Policy withFailOnAcceleratorFallback(boolean enabled) {
            return new Policy(enabled, minTrainSamplesPerSecond, maxValidationToTrainAverageBatchMillisRatio);
        }

        public Policy withMinTrainSamplesPerSecond(double threshold) {
            return new Policy(failOnAcceleratorFallback, threshold, maxValidationToTrainAverageBatchMillisRatio);
        }

        public Policy withMaxValidationToTrainAverageBatchMillisRatio(double threshold) {
            return new Policy(failOnAcceleratorFallback, minTrainSamplesPerSecond, threshold);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("failOnAcceleratorFallback", failOnAcceleratorFallback);
            map.put("minTrainSamplesPerSecond", minTrainSamplesPerSecond);
            map.put(
                    "maxValidationToTrainAverageBatchMillisRatio",
                    maxValidationToTrainAverageBatchMillisRatio);
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
            code = code == null || code.isBlank() ? "performance" : code.trim();
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
            Map<String, Object> acceleration,
            Map<String, Object> throughput) {
        public Result {
            policy = policy == null ? Policy.defaults() : policy;
            findings = findings == null ? List.of() : List.copyOf(findings);
            acceleration = acceleration == null ? Map.of() : Map.copyOf(acceleration);
            throughput = throughput == null ? Map.of() : Map.copyOf(throughput);
        }

        public boolean passed() {
            return findings.isEmpty();
        }

        public String message() {
            if (!available) {
                return "Trainer performance metadata is not available.";
            }
            if (passed()) {
                return "Trainer performance gate passed.";
            }
            return "Trainer performance gate found " + findings.size() + " performance warning(s).";
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
            map.put("acceleration", acceleration);
            map.put("throughput", throughput);
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportPerformanceGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportPerformanceGateJUnitXml.render(this);
        }
    }

    public static Result evaluate(TrainingReport report) {
        return evaluate(report, Policy.defaults());
    }

    public static Result evaluate(TrainingReport report, Policy policy) {
        Objects.requireNonNull(report, "report must not be null");
        Policy resolvedPolicy = policy == null ? Policy.defaults() : policy;
        TrainingReportAcceleration acceleration = report.acceleration();
        TrainingReportThroughput throughput = report.throughput();
        boolean available = acceleration.available() || throughput.available();
        if (!available) {
            return new Result(resolvedPolicy, false, List.of(), acceleration.toMap(), throughput.toMap());
        }
        List<Finding> findings = new ArrayList<>();
        addAccelerationFallbackFinding(findings, acceleration, resolvedPolicy);
        addTrainThroughputFinding(findings, throughput, resolvedPolicy);
        addValidationSkewFinding(findings, throughput, resolvedPolicy);
        return new Result(resolvedPolicy, true, findings, acceleration.toMap(), throughput.toMap());
    }

    public static Result evaluate(Map<String, ?> report) {
        return evaluate(TrainingReport.of(report), Policy.defaults());
    }

    public static Result evaluate(Map<String, ?> report, Policy policy) {
        return evaluate(TrainingReport.of(report), policy);
    }

    private static void addAccelerationFallbackFinding(
            List<Finding> findings,
            TrainingReportAcceleration acceleration,
            Policy policy) {
        if (!policy.failOnAcceleratorFallback()
                || !acceleration.available()
                || !acceleration.requestedAcceleratorFellBack()) {
            return;
        }
        findings.add(new Finding(
                "performance.accelerator_fallback",
                "failure",
                "Requested accelerator `" + acceleration.requestedDevice()
                        + "` resolved to `" + acceleration.executionBackend() + "`.",
                "Enable the requested accelerator or set the trainer device to the observed backend.",
                acceleration.toMap()));
    }

    private static void addTrainThroughputFinding(
            List<Finding> findings,
            TrainingReportThroughput throughput,
            Policy policy) {
        TrainingReportThroughput.Phase train = throughput.train();
        if (train.batchCount().orElse(0L) <= 0L || train.sampleCount().orElse(0L) <= 0L) {
            return;
        }
        double samplesPerSecond = train.samplesPerSecond().orElse(Double.POSITIVE_INFINITY);
        if (samplesPerSecond >= policy.minTrainSamplesPerSecond()) {
            return;
        }
        findings.add(new Finding(
                "performance.train_throughput_below_minimum",
                "failure",
                "Train throughput is below the configured minimum samples/sec.",
                "Profile data loading, backend placement, and batch compute before tuning model quality.",
                phaseEvidence("train", train)));
    }

    private static void addValidationSkewFinding(
            List<Finding> findings,
            TrainingReportThroughput throughput,
            Policy policy) {
        double trainAverage = throughput.train().averageBatchMillis().orElse(0.0);
        double validationAverage = throughput.validation().averageBatchMillis().orElse(0.0);
        if (trainAverage <= 0.0 || validationAverage <= 0.0) {
            return;
        }
        double ratio = validationAverage / trainAverage;
        if (ratio < policy.maxValidationToTrainAverageBatchMillisRatio()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.putAll(phaseEvidence("train", throughput.train()));
        evidence.putAll(phaseEvidence("validation", throughput.validation()));
        evidence.put("validationToTrainAverageBatchMillisRatio", ratio);
        findings.add(new Finding(
                "performance.validation_batch_time_skew",
                "warning",
                "Validation average batch time is high compared with train average batch time.",
                "Cache validation transforms or reduce validation cadence after confirming metric cost.",
                evidence));
    }

    private static Map<String, Object> phaseEvidence(
            String phase,
            TrainingReportThroughput.Phase throughput) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        throughput.batchCount().ifPresent(value -> evidence.put(phase + "BatchCount", value));
        throughput.sampleCount().ifPresent(value -> evidence.put(phase + "SampleCount", value));
        throughput.samplesPerSecond().ifPresent(value -> evidence.put(phase + "SamplesPerSecond", value));
        throughput.averageBatchMillis().ifPresent(value -> evidence.put(phase + "AverageBatchMillis", value));
        return Map.copyOf(evidence);
    }

    private static double nonNegativeOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value >= 0.0 ? value : fallback;
    }

    private static double positiveOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value > 0.0 ? value : fallback;
    }
}
