package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Condenses trainer diagnostics into a CI/SDK-friendly run health contract.
 */
final class TrainerRunHealthMetadata {
    private static final String ERROR = "error";
    private static final String WARNING = "warning";

    private TrainerRunHealthMetadata() {
    }

    static void put(Map<String, Object> metadata) {
        List<Map<String, Object>> issues = new ArrayList<>();
        addTerminalFailureIssues(metadata, issues);
        addCheckpointResumeIssue(metadata, issues);
        addRuntimeCheckpointIssue(metadata, issues);
        addAcceleratorFallbackIssue(metadata, issues);
        addMixedPrecisionIssue(metadata, issues);
        addStructuredHealthIssues(metadata, issues, "dataLoaderPlanHealthIssues");
        addStructuredHealthIssues(metadata, issues, "dataDistributionHealthIssues");
        publish(metadata, issues);
    }

    private static void addTerminalFailureIssues(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        if (flag(metadata, "nonFiniteDetected")) {
            issues.add(issue(
                    "training-failure",
                    "non-finite-detected",
                    ERROR,
                    true,
                    "trainer",
                    text(metadata, "nonFiniteMessage", "training produced a non-finite value"),
                    "inspect data and loss scale, reduce learning rate, or enable gradient clipping"));
        }
        if (flag(metadata, "invalidBatchDetected")) {
            issues.add(issue(
                    "training-failure",
                    "invalid-batch-detected",
                    ERROR,
                    true,
                    "data",
                    text(metadata, "invalidBatchMessage", "training batch failed validation"),
                    "fix dataset collation, batch shapes, or label encoding before rerunning"));
        }
        if (flag(metadata, "invalidLossShapeDetected")) {
            issues.add(issue(
                    "training-failure",
                    "invalid-loss-shape",
                    ERROR,
                    true,
                    "loss",
                    text(metadata, "invalidLossShapeMessage", "loss tensor has an invalid shape"),
                    "return a scalar loss or reduce the loss tensor before backward"));
        }
        if (flag(metadata, "invalidMetricDetected")) {
            issues.add(issue(
                    "training-failure",
                    "invalid-metric-detected",
                    ERROR,
                    true,
                    "metric",
                    text(metadata, "invalidMetricMessage", "metric evaluation failed validation"),
                    "fix the metric implementation or remove the metric from the training run"));
        }
    }

    private static void addCheckpointResumeIssue(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        if (flag(metadata, "checkpointResumeBlockingIssueDetected")) {
            issues.add(issue(
                    "checkpoint-resume",
                    text(metadata, "checkpointResumePrimaryBlockingIssueCode", "checkpoint-resume-blocking-issue"),
                    ERROR,
                    true,
                    text(metadata, "checkpointResumePrimaryBlockingAffectedArtifact", "checkpoint"),
                    text(metadata, "checkpointResumePrimaryBlockingIssueMessage", "checkpoint resume has a blocking issue"),
                    text(
                            metadata,
                            "checkpointResumePrimaryBlockingRecommendedAction",
                            "start fresh or provide a compatible checkpoint set")));
        } else if (flag(metadata, "checkpointResumeIssueDetected")) {
            issues.add(issue(
                    "checkpoint-resume",
                    text(metadata, "checkpointResumePrimaryIssueCode", "checkpoint-resume-partial"),
                    WARNING,
                    false,
                    text(metadata, "checkpointResumePrimaryAffectedArtifact", "checkpoint"),
                    text(metadata, "checkpointResumePrimaryIssueMessage", "checkpoint resume is incomplete"),
                    text(
                            metadata,
                            "checkpointResumePrimaryRecommendedAction",
                            "restore missing checkpoint artifacts or continue without resume state")));
        }
    }

    private static void addRuntimeCheckpointIssue(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        if (!flag(metadata, "runtimeCheckpointIntegrityMismatch") && !flag(metadata, "runtimeCheckpointLoadFailed")) {
            return;
        }
        issues.add(issue(
                "runtime-checkpoint",
                text(metadata, "runtimeCheckpointResumeDecision", "runtime-checkpoint-unusable"),
                WARNING,
                false,
                "runtime",
                flag(metadata, "runtimeCheckpointIntegrityMismatch")
                        ? "runtime checkpoint failed integrity validation"
                        : "runtime checkpoint failed to load",
                text(
                        metadata,
                        "runtimeCheckpointRecommendedAction",
                        "skip runtime checkpoint and rebuild runtime state from trainer artifacts")));
    }

    private static void addAcceleratorFallbackIssue(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        if (!flag(metadata, "executionFallback")) {
            return;
        }
        String requested = text(metadata, "requestedDevice", "accelerator");
        String actual = text(metadata, "executionBackend", "cpu");
        issues.add(issue(
                "accelerator",
                "accelerator-fallback",
                WARNING,
                false,
                "execution",
                "requested device '" + requested + "' fell back to '" + actual + "'",
                "install or enable the requested accelerator, or set device to the observed backend"));
    }

    private static void addMixedPrecisionIssue(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        if (!flag(metadata, "mixedPrecisionOverflowDetected")) {
            return;
        }
        issues.add(issue(
                "mixed-precision",
                "mixed-precision-overflow",
                WARNING,
                false,
                "optimizer",
                "mixed precision overflow was detected during training",
                "lower the initial loss scale or let GradScaler continue adapting"));
    }

    private static void addStructuredHealthIssues(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues,
            String metadataKey) {
        Object rawIssues = metadata.get(metadataKey);
        if (!(rawIssues instanceof Iterable<?> iterable)) {
            return;
        }
        for (Object rawIssue : iterable) {
            if (!(rawIssue instanceof Map<?, ?> source)) {
                continue;
            }
            String severity = textAny(source, "severity", WARNING);
            if (!ERROR.equals(severity) && !WARNING.equals(severity)) {
                continue;
            }
            issues.add(issue(
                    textAny(source, "kind", "data-loader"),
                    textAny(source, "code", "data-loader-plan-health-issue"),
                    severity,
                    Boolean.TRUE.equals(source.get("blocking")),
                    textAny(source, "artifact", "data-loader"),
                    textAny(source, "message", "data loader plan health issue detected"),
                    textAny(source, "action", "inspect data loader plan metadata")));
        }
    }

    private static void publish(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues) {
        long blocking = issues.stream()
                .filter(issue -> Boolean.TRUE.equals(issue.get("blocking")))
                .count();
        boolean hasError = issues.stream()
                .anyMatch(issue -> ERROR.equals(issue.get("severity")));
        String status = hasError ? "failed" : issues.isEmpty() ? "healthy" : "warning";
        metadata.put("trainingRunHealthStatus", status);
        metadata.put("trainingRunHealthy", issues.isEmpty());
        metadata.put("trainingRunIssueDetected", !issues.isEmpty());
        metadata.put("trainingRunIssueCount", issues.size());
        metadata.put("trainingRunBlockingIssueDetected", blocking > 0L);
        metadata.put("trainingRunBlockingIssueCount", (int) blocking);
        metadata.put("trainingRunGatePassed", blocking == 0L);
        metadata.put("trainingRunIssueCodes", immutableValues(issues, "code"));
        metadata.put("trainingRunIssueSeverities", immutableDistinctValues(issues, "severity"));
        metadata.put("trainingRunRecommendedActions", immutableDistinctValues(issues, "action"));
        metadata.put("trainingRunIssueCountsByKind", immutableCountsBy(issues, "kind"));
        metadata.put("trainingRunIssueCountsBySeverity", immutableCountsBy(issues, "severity"));
        putPrimaryIssue(metadata, issues, false);
        putPrimaryIssue(metadata, issues, true);
        metadata.put(
                "trainingRunRecommendedAction",
                issues.isEmpty()
                        ? "continue monitoring training"
                        : String.valueOf(primaryIssue(issues, blocking > 0L).get("action")));
        metadata.put("trainingRunIssues", Collections.unmodifiableList(issues));
    }

    private static void putPrimaryIssue(
            Map<String, Object> metadata,
            List<Map<String, Object>> issues,
            boolean blockingOnly) {
        String prefix = blockingOnly ? "trainingRunPrimaryBlockingIssue" : "trainingRunPrimaryIssue";
        Map<String, Object> issue = primaryIssue(issues, blockingOnly);
        metadata.put(prefix + "Available", issue != null);
        if (issue == null) {
            return;
        }
        metadata.put(prefix + "Kind", issue.get("kind"));
        metadata.put(prefix + "Code", issue.get("code"));
        metadata.put(prefix + "Severity", issue.get("severity"));
        metadata.put(prefix + "Blocking", issue.get("blocking"));
        metadata.put(prefix + "Artifact", issue.get("artifact"));
        metadata.put(prefix + "Message", issue.get("message"));
        metadata.put(prefix + "RecommendedAction", issue.get("action"));
    }

    private static Map<String, Object> issue(
            String kind,
            String code,
            String severity,
            boolean blocking,
            String artifact,
            String message,
            String action) {
        Map<String, Object> issue = new LinkedHashMap<>();
        issue.put("kind", kind);
        issue.put("code", code);
        issue.put("severity", severity);
        issue.put("blocking", blocking);
        issue.put("artifact", artifact);
        issue.put("message", message);
        issue.put("action", action);
        return Collections.unmodifiableMap(issue);
    }

    private static List<String> immutableValues(
            List<Map<String, Object>> issues,
            String key) {
        List<String> values = new ArrayList<>();
        for (Map<String, Object> issue : issues) {
            values.add(String.valueOf(issue.get(key)));
        }
        return Collections.unmodifiableList(values);
    }

    private static List<String> immutableDistinctValues(
            List<Map<String, Object>> issues,
            String key) {
        List<String> values = new ArrayList<>();
        for (Map<String, Object> issue : issues) {
            String value = String.valueOf(issue.get(key));
            if (!values.contains(value)) {
                values.add(value);
            }
        }
        return Collections.unmodifiableList(values);
    }

    private static Map<String, Integer> immutableCountsBy(
            List<Map<String, Object>> issues,
            String key) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map<String, Object> issue : issues) {
            String value = String.valueOf(issue.get(key));
            counts.put(value, counts.getOrDefault(value, 0) + 1);
        }
        return Collections.unmodifiableMap(counts);
    }

    private static Map<String, Object> primaryIssue(
            List<Map<String, Object>> issues,
            boolean blockingOnly) {
        for (Map<String, Object> issue : issues) {
            if (!blockingOnly || Boolean.TRUE.equals(issue.get("blocking"))) {
                return issue;
            }
        }
        return null;
    }

    private static boolean flag(Map<String, Object> metadata, String key) {
        return Boolean.TRUE.equals(metadata.get(key));
    }

    private static String text(Map<String, Object> metadata, String key, String fallback) {
        Object value = metadata.get(key);
        return value == null || value.toString().isBlank() ? fallback : value.toString();
    }

    private static String textAny(Map<?, ?> metadata, String key, String fallback) {
        Object value = metadata.get(key);
        return value == null || value.toString().isBlank() ? fallback : value.toString();
    }
}
