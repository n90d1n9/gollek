package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Single-report validation policy for CI, CLI, SDKs, and promotion preflight.
 */
public record TrainingReportValidationPolicy(
        TrainingReportDiagnostics.Severity maxDiagnosticSeverity,
        boolean requireRunHealthGate,
        boolean requireDataHealthGate,
        boolean requireDataHealthAvailable,
        boolean requireFreshDiagnostics,
        boolean requireValidation,
        boolean requireCheckpointIntegrity) {
    public TrainingReportValidationPolicy {
        maxDiagnosticSeverity = maxDiagnosticSeverity == null
                ? TrainingReportDiagnostics.Severity.INFO
                : maxDiagnosticSeverity;
    }

    public TrainingReportValidationPolicy(
            TrainingReportDiagnostics.Severity maxDiagnosticSeverity,
            boolean requireRunHealthGate,
            boolean requireDataHealthGate,
            boolean requireFreshDiagnostics,
            boolean requireValidation,
            boolean requireCheckpointIntegrity) {
        this(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireDataHealthGate,
                false,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy(
            TrainingReportDiagnostics.Severity maxDiagnosticSeverity,
            boolean requireRunHealthGate,
            boolean requireFreshDiagnostics,
            boolean requireValidation,
            boolean requireCheckpointIntegrity) {
        this(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireRunHealthGate,
                false,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public static TrainingReportValidationPolicy strict() {
        return new TrainingReportValidationPolicy(
                TrainingReportDiagnostics.Severity.INFO,
                true,
                true,
                true,
                true);
    }

    public static TrainingReportValidationPolicy defaultPolicy() {
        return strict();
    }

    public static TrainingReportValidationPolicy permissive() {
        return new TrainingReportValidationPolicy(
                TrainingReportDiagnostics.Severity.CRITICAL,
                false,
                false,
                false,
                false);
    }

    public static TrainingReportValidationPolicy fromMap(Map<String, ?> policy) {
        if (policy == null || policy.isEmpty()) {
            return defaultPolicy();
        }
        return new TrainingReportValidationPolicy(
                severityValue(policy.get("maxDiagnosticSeverity"), TrainingReportDiagnostics.Severity.INFO),
                booleanEntry(policy, "requireRunHealthGate", true),
                booleanEntry(
                        policy,
                        "requireDataHealthGate",
                        booleanEntry(policy, "requireRunHealthGate", true)),
                booleanEntry(policy, "requireDataHealthAvailable", false),
                booleanEntry(policy, "requireFreshDiagnostics", true),
                booleanEntry(policy, "requireValidation", true),
                booleanEntry(policy, "requireCheckpointIntegrity", true));
    }

    public TrainingReportValidationPolicy withMaxDiagnosticSeverity(
            TrainingReportDiagnostics.Severity severity) {
        return new TrainingReportValidationPolicy(
                severity,
                requireRunHealthGate,
                requireDataHealthGate,
                requireDataHealthAvailable,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireRunHealthGate(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                required,
                requireDataHealthGate,
                requireDataHealthAvailable,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireDataHealthGate(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                required,
                requireDataHealthAvailable,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireDataHealthAvailable(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireDataHealthGate,
                required,
                requireFreshDiagnostics,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireFreshDiagnostics(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireDataHealthGate,
                requireDataHealthAvailable,
                required,
                requireValidation,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireValidation(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireDataHealthGate,
                requireDataHealthAvailable,
                requireFreshDiagnostics,
                required,
                requireCheckpointIntegrity);
    }

    public TrainingReportValidationPolicy withRequireCheckpointIntegrity(boolean required) {
        return new TrainingReportValidationPolicy(
                maxDiagnosticSeverity,
                requireRunHealthGate,
                requireDataHealthGate,
                requireDataHealthAvailable,
                requireFreshDiagnostics,
                requireValidation,
                required);
    }

    public Result validate(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        TrainingReportDiagnostics.GateResult diagnosticGate = report.diagnosticGate(maxDiagnosticSeverity);
        TrainingReportRunHealth runHealth = report.runHealth();
        TrainingReportDataHealth dataHealth = report.dataHealth();
        TrainingReportDiagnostics.Provenance provenance = report.diagnosticProvenance();
        boolean validationAvailable = validationAvailable(report);
        CheckpointIntegrity checkpointIntegrity = checkpointIntegrity(report.metadata());

        List<String> failureCodes = new ArrayList<>();
        List<String> reasons = new ArrayList<>();
        if (!diagnosticGate.passed()) {
            failureCodes.add("diagnostics.exceeded");
            reasons.add(diagnosticGate.message());
        }
        if (requireRunHealthGate && (!runHealth.gatePassed() || runHealth.blockingIssueDetected())) {
            failureCodes.add("run_health.gate_failed");
            reasons.add("Trainer run health did not pass: " + runHealth.recommendedAction() + ".");
        }
        if (requireDataHealthGate && dataHealthGateFailed(dataHealth)) {
            failureCodes.add("data_health.gate_failed");
            reasons.add("Trainer data health did not pass: " + dataHealthReason(dataHealth) + ".");
        }
        if (requireDataHealthAvailable && !dataHealth.available()) {
            failureCodes.add("data_health.missing");
            reasons.add("Trainer data health was not recorded; regenerate the report with data-loader "
                    + "plan and distribution health capture enabled.");
        }
        if (requireFreshDiagnostics && provenance.stale()) {
            failureCodes.add("diagnostics.not_fresh");
            reasons.add("Diagnostics are " + provenance.status() + "; regenerate the report to persist current findings.");
        }
        if (requireValidation && !validationAvailable) {
            failureCodes.add("validation.missing");
            reasons.add("Validation loss is not available; attach validation data or disable the validation requirement.");
        }
        if (requireCheckpointIntegrity && checkpointIntegrity.available() && !checkpointIntegrity.passed()) {
            failureCodes.add("checkpoint_integrity.failed");
            reasons.add("Checkpoint integrity metadata reports failures: "
                    + checkpointIntegrity.failureKeys() + ".");
        }

        return new Result(
                this,
                diagnosticGate,
                runHealth,
                dataHealth,
                provenance,
                validationAvailable,
                checkpointIntegrity.available(),
                checkpointIntegrity.passed(),
                checkpointIntegrity.failureKeys(),
                failureCodes,
                reasons);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("maxDiagnosticSeverity", maxDiagnosticSeverity.name());
        map.put("requireRunHealthGate", requireRunHealthGate);
        map.put("requireDataHealthGate", requireDataHealthGate);
        map.put("requireDataHealthAvailable", requireDataHealthAvailable);
        map.put("requireFreshDiagnostics", requireFreshDiagnostics);
        map.put("requireValidation", requireValidation);
        map.put("requireCheckpointIntegrity", requireCheckpointIntegrity);
        return Map.copyOf(map);
    }

    public record Result(
            TrainingReportValidationPolicy policy,
            TrainingReportDiagnostics.GateResult diagnosticGate,
            TrainingReportRunHealth runHealth,
            TrainingReportDataHealth dataHealth,
            TrainingReportDiagnostics.Provenance diagnosticProvenance,
            boolean validationAvailable,
            boolean checkpointIntegrityAvailable,
            boolean checkpointIntegrityPassed,
            List<String> checkpointIntegrityFailureKeys,
            List<String> failureCodes,
            List<String> reasons) {
        public Result(
                TrainingReportValidationPolicy policy,
                TrainingReportDiagnostics.GateResult diagnosticGate,
                TrainingReportRunHealth runHealth,
                TrainingReportDiagnostics.Provenance diagnosticProvenance,
                boolean validationAvailable,
                boolean checkpointIntegrityAvailable,
                boolean checkpointIntegrityPassed,
                List<String> checkpointIntegrityFailureKeys,
                List<String> failureCodes,
                List<String> reasons) {
            this(
                    policy,
                    diagnosticGate,
                    runHealth,
                    TrainingReportDataHealth.fromMap(Map.of()),
                    diagnosticProvenance,
                    validationAvailable,
                    checkpointIntegrityAvailable,
                    checkpointIntegrityPassed,
                    checkpointIntegrityFailureKeys,
                    failureCodes,
                    reasons);
        }

        public Result {
            policy = policy == null ? TrainingReportValidationPolicy.defaultPolicy() : policy;
            diagnosticGate = Objects.requireNonNull(diagnosticGate, "diagnosticGate must not be null");
            runHealth = runHealth == null ? TrainingReportRunHealth.defaultHealthy() : runHealth;
            dataHealth = dataHealth == null ? TrainingReportDataHealth.fromMap(Map.of()) : dataHealth;
            diagnosticProvenance = Objects.requireNonNull(
                    diagnosticProvenance,
                    "diagnosticProvenance must not be null");
            checkpointIntegrityFailureKeys = List.copyOf(
                    checkpointIntegrityFailureKeys == null ? List.of() : checkpointIntegrityFailureKeys);
            failureCodes = List.copyOf(failureCodes == null ? List.of() : failureCodes);
            reasons = List.copyOf(reasons == null ? List.of() : reasons);
            checkpointIntegrityPassed = !checkpointIntegrityAvailable || checkpointIntegrityPassed;
        }

        public boolean passed() {
            return failureCodes.isEmpty();
        }

        public boolean failed() {
            return !passed();
        }

        public String message() {
            if (passed()) {
                return "Training report validation passed.";
            }
            return "Training report validation failed: " + String.join(" ", reasons);
        }

        public void requirePassed() {
            if (failed()) {
                throw new IllegalStateException(message());
            }
        }

        public static Result fromMap(Map<String, ?> result) {
            Objects.requireNonNull(result, "result must not be null");
            return new Result(
                    TrainingReportValidationPolicy.fromMap(mapValue(result, "policy")),
                    TrainingReportDiagnostics.GateResult.fromMap(mapValue(result, "diagnosticGate")),
                    TrainingReportRunHealth.fromMap(mapValue(result, "runHealth")),
                    TrainingReportDataHealth.fromMap(mapValue(result, "dataHealth")),
                    TrainingReportDiagnostics.Provenance.fromMap(mapValue(result, "diagnosticProvenance")),
                    flag(result.get("validationAvailable")),
                    flag(result.get("checkpointIntegrityAvailable")),
                    flag(result.get("checkpointIntegrityPassed")),
                    stringList(result.get("checkpointIntegrityFailureKeys")),
                    stringList(result.get("failureCodes")),
                    stringList(result.get("reasons")));
        }

        public String markdown() {
            return TrainingReportValidationMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportValidationJUnitXml.render(this);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("failed", failed());
            map.put("message", message());
            map.put("policy", policy.toMap());
            map.put("diagnosticGate", diagnosticGateToMap(diagnosticGate));
            map.put("runHealth", runHealth.toMap());
            map.put("dataHealth", dataHealth.toMap());
            map.put("diagnosticProvenance", diagnosticProvenance.toMap());
            map.put("validationAvailable", validationAvailable);
            map.put("checkpointIntegrityAvailable", checkpointIntegrityAvailable);
            map.put("checkpointIntegrityPassed", checkpointIntegrityPassed);
            map.put("checkpointIntegrityFailureKeys", checkpointIntegrityFailureKeys);
            map.put("failureCodes", failureCodes);
            map.put("reasons", reasons);
            return Map.copyOf(map);
        }
    }

    private static boolean validationAvailable(TrainingReport report) {
        return report.latestValidationLoss().isPresent()
                || report.validationLoss().available()
                || report.validationLossSeries().available();
    }

    private static boolean dataHealthGateFailed(TrainingReportDataHealth dataHealth) {
        return dataHealth != null
                && dataHealth.available()
                && (!dataHealth.gatePassed() || dataHealth.errorCount() > 0);
    }

    private static String dataHealthReason(TrainingReportDataHealth dataHealth) {
        if (dataHealth == null) {
            return "data-health status is unavailable";
        }
        if (!dataHealth.recommendedActions().isEmpty()) {
            return dataHealth.recommendedActions().get(0);
        }
        if (!dataHealth.issueCodes().isEmpty()) {
            return "inspect data-health issue " + dataHealth.issueCodes().get(0);
        }
        return "inspect trainer data-loader and distribution health before promotion";
    }

    private static CheckpointIntegrity checkpointIntegrity(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return new CheckpointIntegrity(false, true, List.of());
        }
        List<String> failureKeys = new ArrayList<>();
        boolean available = false;
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            String key = entry.getKey();
            if (!checkpointIntegrityKey(key)) {
                continue;
            }
            available = true;
            if (checkpointFailureKey(key) && flag(entry.getValue())) {
                failureKeys.add(key);
            }
        }
        return new CheckpointIntegrity(available, failureKeys.isEmpty(), failureKeys);
    }

    private static boolean checkpointIntegrityKey(String key) {
        if (key == null || key.isBlank()) {
            return false;
        }
        return key.startsWith("checkpointManifest")
                || key.startsWith("runtimeCheckpoint")
                || key.contains("CheckpointIntegrity")
                || key.contains("checkpointIntegrity");
    }

    private static boolean checkpointFailureKey(String key) {
        return key.endsWith("IntegrityMismatch")
                || key.endsWith("LoadFailed")
                || key.endsWith("SaveFailed")
                || key.endsWith("MissingOnResume");
    }

    private static Map<String, Object> diagnosticGateToMap(TrainingReportDiagnostics.GateResult gate) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("passed", gate.passed());
        map.put("message", gate.message());
        map.put("maxAllowedSeverity", gate.maxAllowedSeverity().name());
        map.put("summary", gate.summary().toMap());
        map.put("failingCodes", gate.failingCodes());
        map.put("failingFindings", TrainingReportDiagnostics.toMaps(gate.failingFindings()));
        return Map.copyOf(map);
    }

    private record CheckpointIntegrity(
            boolean available,
            boolean passed,
            List<String> failureKeys) {
        private CheckpointIntegrity {
            failureKeys = List.copyOf(failureKeys == null ? List.of() : failureKeys);
            passed = !available || passed;
        }
    }

    private static TrainingReportDiagnostics.Severity severityValue(
            Object value,
            TrainingReportDiagnostics.Severity fallback) {
        if (value instanceof TrainingReportDiagnostics.Severity severity) {
            return severity;
        }
        if (value != null) {
            String text = String.valueOf(value).trim().toUpperCase();
            for (TrainingReportDiagnostics.Severity severity : TrainingReportDiagnostics.Severity.values()) {
                if (severity.name().equals(text)) {
                    return severity;
                }
            }
        }
        return fallback;
    }

    private static Map<String, Object> mapValue(Map<String, ?> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item != null) {
                String text = String.valueOf(item).trim();
                if (!text.isEmpty()) {
                    values.add(text);
                }
            }
        }
        return List.copyOf(values);
    }

    private static boolean booleanEntry(Map<String, ?> source, String key, boolean fallback) {
        return source.containsKey(key) ? flag(source.get(key)) : fallback;
    }

    private static boolean flag(Object value) {
        return TrainingReportValues.booleanValue(value);
    }
}
