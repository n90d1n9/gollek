package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CI-friendly gate for detecting runtime input-pipeline bottlenecks.
 */
public final class TrainingReportRuntimeInputProfileGate {
    private TrainingReportRuntimeInputProfileGate() {
    }

    public record Policy(
            double maxDominantScopePercent,
            double maxDominantStagePercent,
            double maxTrainToValidationTotalRatio) {
        public Policy {
            maxDominantScopePercent = positiveOrDefault(maxDominantScopePercent, 85.0);
            maxDominantStagePercent = positiveOrDefault(maxDominantStagePercent, 75.0);
            maxTrainToValidationTotalRatio = positiveOrDefault(maxTrainToValidationTotalRatio, 8.0);
        }

        public static Policy defaults() {
            return new Policy(85.0, 75.0, 8.0);
        }

        public static Policy strict() {
            return new Policy(80.0, 70.0, 5.0);
        }

        public static Policy permissive() {
            return new Policy(95.0, 95.0, 20.0);
        }

        public static Policy fromMap(Map<String, ?> policy) {
            if (policy == null || policy.isEmpty()) {
                return defaults();
            }
            return new Policy(
                    doubleEntry(policy, "maxDominantScopePercent", defaults().maxDominantScopePercent()),
                    doubleEntry(policy, "maxDominantStagePercent", defaults().maxDominantStagePercent()),
                    doubleEntry(policy, "maxTrainToValidationTotalRatio", defaults().maxTrainToValidationTotalRatio()));
        }

        public Policy withMaxDominantScopePercent(double threshold) {
            return new Policy(threshold, maxDominantStagePercent, maxTrainToValidationTotalRatio);
        }

        public Policy withMaxDominantStagePercent(double threshold) {
            return new Policy(maxDominantScopePercent, threshold, maxTrainToValidationTotalRatio);
        }

        public Policy withMaxTrainToValidationTotalRatio(double threshold) {
            return new Policy(maxDominantScopePercent, maxDominantStagePercent, threshold);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("maxDominantScopePercent", maxDominantScopePercent);
            map.put("maxDominantStagePercent", maxDominantStagePercent);
            map.put("maxTrainToValidationTotalRatio", maxTrainToValidationTotalRatio);
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
            code = code == null ? "" : code;
            severity = severity == null || severity.isBlank() ? "warning" : severity;
            message = message == null ? "" : message;
            action = action == null ? "" : action;
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
            Map<String, Object> inputProfile) {
        public Result {
            policy = policy == null ? Policy.defaults() : policy;
            findings = findings == null ? List.of() : List.copyOf(findings);
            inputProfile = inputProfile == null ? Map.of() : Map.copyOf(inputProfile);
        }

        public boolean passed() {
            return findings.isEmpty();
        }

        public String message() {
            if (!available) {
                return "Runtime input profile is not available.";
            }
            if (passed()) {
                return "Runtime input profile gate passed.";
            }
            return "Runtime input profile gate found " + findings.size() + " bottleneck warning(s).";
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
            map.put("inputProfile", inputProfile);
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportRuntimeInputProfileGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportRuntimeInputProfileGateJUnitXml.render(this);
        }
    }

    public static Result evaluate(TrainingReport report) {
        return evaluate(report, Policy.defaults());
    }

    public static Result evaluate(TrainingReport report, Policy policy) {
        Objects.requireNonNull(report, "report must not be null");
        return evaluate(report.payload(), policy);
    }

    public static Result evaluate(Map<String, ?> report) {
        return evaluate(report, Policy.defaults());
    }

    public static Result evaluate(Map<String, ?> report, Policy policy) {
        Policy resolvedPolicy = policy == null ? Policy.defaults() : policy;
        Map<String, Object> inputProfile = TrainingReportReader.runtimeInputProfile(report);
        boolean available = booleanValue(inputProfile.get("available"));
        if (!available) {
            return new Result(resolvedPolicy, false, List.of(), inputProfile);
        }
        List<Finding> findings = new ArrayList<>();
        addDominantScopeFinding(findings, inputProfile, resolvedPolicy);
        addDominantStageFinding(findings, inputProfile, resolvedPolicy);
        addTrainValidationSkewFinding(findings, inputProfile, resolvedPolicy);
        TrainingReportRuntimeInputPrefetchAdvisor.addFindings(
                findings,
                TrainingReportReader.metadata(report),
                inputProfile);
        return new Result(resolvedPolicy, true, findings, inputProfile);
    }

    private static void addDominantScopeFinding(
            List<Finding> findings,
            Map<String, ?> inputProfile,
            Policy policy) {
        double percent = doubleValue(inputProfile.get("dominantScopePercent"), 0.0);
        if (percent <= policy.maxDominantScopePercent()) {
            return;
        }
        String scope = stringValue(inputProfile.get("dominantScope"));
        findings.add(new Finding(
                "runtime-input-dominant-scope",
                "warning",
                "Input pipeline time is concentrated in `" + scope + "`.",
                "Inspect the `" + scope + "` loader path before tuning unrelated trainer compute.",
                evidence(inputProfile, "dominantScope", "dominantScopePercent", "dominantScopeTotalMillis")));
    }

    private static void addDominantStageFinding(
            List<Finding> findings,
            Map<String, ?> inputProfile,
            Policy policy) {
        StageView stage = dominantStage(inputProfile);
        if (stage.percent() <= policy.maxDominantStagePercent()) {
            return;
        }
        findings.add(new Finding(
                "runtime-input-dominant-stage",
                "warning",
                "Input pipeline time is concentrated in `" + stage.scope() + "." + stage.stage() + "()`.",
                "Prioritize `" + stage.scope() + "." + stage.stage()
                        + "()` batching, decoding, caching, or prefetching before changing model math.",
                stage.evidence()));
    }

    private static void addTrainValidationSkewFinding(
            List<Finding> findings,
            Map<String, ?> inputProfile,
            Policy policy) {
        double ratio = doubleValue(inputProfile.get("trainToValidationTotalRatio"), 0.0);
        if (ratio <= policy.maxTrainToValidationTotalRatio()) {
            return;
        }
        findings.add(new Finding(
                "runtime-input-train-validation-skew",
                "warning",
                "Training input time is much higher than validation input time.",
                "Compare train/validation transforms and sampling to make sure training-only preprocessing is intentional.",
                evidence(inputProfile, "trainToValidationTotalRatio", "dominantScope", "totalMillis")));
    }

    private static StageView dominantStage(Map<String, ?> inputProfile) {
        StageView train = stageView(inputProfile, "train");
        StageView validation = stageView(inputProfile, "validation");
        return train.totalMillis() >= validation.totalMillis() ? train : validation;
    }

    private static StageView stageView(Map<String, ?> inputProfile, String scopeName) {
        Map<String, ?> scope = mapValue(inputProfile.get(scopeName));
        String stage = stringValue(scope.get("dominantStage"));
        double totalMillis = doubleValue(scope.get("dominantStageTotalMillis"), 0.0);
        double percent = doubleValue(scope.get("dominantStagePercent"), 0.0);
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("scope", scopeName);
        evidence.put("stage", stage);
        evidence.put("stageTotalMillis", totalMillis);
        evidence.put("stagePercent", percent);
        evidence.put("scopeTotalMillis", doubleValue(scope.get("totalMillis"), 0.0));
        return new StageView(scopeName, stage, totalMillis, percent, Map.copyOf(evidence));
    }

    private static Map<String, Object> evidence(Map<String, ?> source, String... keys) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        for (String key : keys) {
            if (source.containsKey(key)) {
                evidence.put(key, source.get(key));
            }
        }
        return Map.copyOf(evidence);
    }

    private static double doubleValue(Object value, double fallback) {
        var parsed = optionalDouble(value);
        return parsed.isPresent() ? parsed.getAsDouble() : fallback;
    }

    private static Map<String, ?> mapValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> typed = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                typed.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            return Map.copyOf(typed);
        }
        return Map.of();
    }

    private static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        return Boolean.parseBoolean(String.valueOf(value));
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static double doubleEntry(Map<String, ?> source, String key, double fallback) {
        var parsed = optionalDouble(source.get(key));
        return parsed.isPresent() ? parsed.getAsDouble() : fallback;
    }

    private static double positiveOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value > 0.0 ? value : fallback;
    }

    private record StageView(
            String scope,
            String stage,
            double totalMillis,
            double percent,
            Map<String, Object> evidence) {
    }
}
