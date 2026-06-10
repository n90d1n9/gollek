package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Actionable explanation for a checkpoint resume preflight decision.
 */
public record DiscreteTokenDatasetCheckpointResumeExplanation(
        String status,
        boolean ready,
        String summary,
        List<Finding> findings,
        List<String> nextSteps) {

    public DiscreteTokenDatasetCheckpointResumeExplanation {
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        summary = DiscreteTokenDatasetMetadataSupport.requireText(summary, "summary");
        findings = immutableFindings(findings);
        nextSteps = DiscreteTokenDatasetMetadataSupport.optionalTextList(nextSteps, "nextSteps");
        String expectedStatus = ready ? "ready" : "blocked";
        if (!expectedStatus.equals(status)) {
            throw new IllegalArgumentException("status must be " + expectedStatus + " when ready=" + ready);
        }
    }

    public static DiscreteTokenDatasetCheckpointResumeExplanation from(
            DiscreteTokenDatasetCheckpointResumeReport report) {
        Objects.requireNonNull(report, "report must not be null");
        List<Finding> findings = new ArrayList<>();
        if (report.ready()) {
            findings.add(new Finding(
                    "resume-ready",
                    "ready",
                    "info",
                    "checkpoint can resume with the current dataset plan",
                    "Continue training from checkpoint "
                            + report.checkpoint().runId()
                            + " step "
                            + report.checkpoint().checkpointStep()
                            + ".",
                    readyDetails(report)));
            addCompatibilityWarnings(report, findings);
        } else {
            addSchemaFinding(report, findings);
            addCheckpointDatasetFinding(report, findings);
            addFingerprintFinding(report, findings);
            addExpectationFindings(report, findings);
            addCurrentPlanFinding(report, findings);
        }

        return new DiscreteTokenDatasetCheckpointResumeExplanation(
                report.status(),
                report.ready(),
                explanationSummary(report, findings),
                findings,
                nextSteps(findings));
    }

    public static DiscreteTokenDatasetCheckpointResumeExplanation fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeExplanation(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "ready"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "summary"),
                findingsFromMetadata(metadata, "findings"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "nextSteps"))
                .requireMetadataMatch(metadata);
    }

    public String primaryCode() {
        return findings.isEmpty() ? "none" : findings.get(0).code();
    }

    public String primaryCategory() {
        return findings.isEmpty() ? "none" : findings.get(0).category();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status);
        metadata.put("ready", ready);
        metadata.put("summary", summary);
        metadata.put("primaryCode", primaryCode());
        metadata.put("primaryCategory", primaryCategory());
        metadata.put("findingCount", findings.size());
        metadata.put("findings", findings.stream()
                .map(Finding::toMetadata)
                .toList());
        metadata.put("nextSteps", nextSteps);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public DiscreteTokenDatasetCheckpointResumeExplanation requireMetadataMatch(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Map<String, Object> expected = toMetadata();
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            Object actual = DiscreteTokenDatasetMetadataSupport.required(metadata, entry.getKey());
            if (!DiscreteTokenDatasetMetadataSupport.metadataValueMatches(entry.getValue(), actual)) {
                throw new IllegalArgumentException(
                        "resume explanation field '" + entry.getKey() + "' does not match findings");
            }
        }
        return this;
    }

    private static void addSchemaFinding(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.schemaAccepted()) {
            return;
        }
        findings.add(new Finding(
                "checkpoint-schema-mismatch",
                "manifest",
                "error",
                "checkpoint manifest schema is "
                        + report.checkpoint().schemaVersion()
                        + " but expected "
                        + DiscreteTokenDatasetCheckpointManifestSnapshot.SCHEMA_VERSION,
                "Regenerate or migrate the checkpoint manifest with the current Gollek trainer schema.",
                Map.of(
                        "actualSchemaVersion", report.checkpoint().schemaVersion(),
                        "expectedSchemaVersion", DiscreteTokenDatasetCheckpointManifestSnapshot.SCHEMA_VERSION)));
    }

    private static void addCheckpointDatasetFinding(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.datasetAccepted()) {
            return;
        }
        findings.add(new Finding(
                "checkpoint-dataset-rejected",
                "checkpoint-dataset",
                "error",
                "checkpoint dataset was not accepted by the checkpoint readiness gate: "
                        + report.checkpoint().datasetGateStatus(),
                "Resume from a checkpoint built from an accepted dataset plan, or rebuild this checkpoint after fixing the dataset plan.",
                Map.of(
                        "checkpointGateStatus", report.checkpoint().datasetGateStatus(),
                        "checkpointAccepted", report.checkpoint().datasetAccepted())));
    }

    private static void addFingerprintFinding(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.fingerprintMatched()) {
            return;
        }
        DiscreteTokenDatasetFingerprintMatch match = report.fingerprintMatch();
        findings.add(new Finding(
                "dataset-fingerprint-mismatch",
                "dataset-fingerprint",
                "error",
                match.summary(),
                "Use the original dataset, tokenizer, split, and planning configuration for this checkpoint, or start a fresh checkpoint lineage for the changed dataset.",
                Map.of(
                        "expected", match.expected().toMetadata(),
                        "actual", match.actual().toMetadata(),
                        "mismatchReasons", match.mismatchReasons())));
    }

    private static void addExpectationFindings(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        for (String reason : report.expectation().rejectionReasons(report.checkpoint())) {
            findings.add(new Finding(
                    expectationCode(reason),
                    "resume-expectation",
                    "error",
                    reason,
                    "Select a checkpoint that matches the active resume expectation, or deliberately relax that expectation in the resume policy.",
                    Map.of("reason", reason)));
        }
    }

    private static void addCurrentPlanFinding(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.currentPlanAccepted()) {
            return;
        }
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("currentPlanGateStatus", report.currentPlanGateStatus());
        details.put("currentPlanChecked", report.currentPlanChecked());
        if (report.currentPlanReport() != null) {
            details.put("currentPlanFingerprint", report.currentPlanReport().fingerprint().toMetadata());
            details.put("currentPlanRejectionReasons", report.currentPlanReport().readiness().rejectionReasons());
            details.put("currentPlanWarnings", report.currentPlanReport().warnings());
        }
        findings.add(new Finding(
                "current-plan-rejected",
                "current-dataset",
                "error",
                "current dataset plan was not accepted: " + report.currentPlanGateStatus(),
                "Fix the current dataset readiness issues or use a resume policy whose gate intentionally accepts this plan.",
                details));
    }

    private static Map<String, Object> readyDetails(DiscreteTokenDatasetCheckpointResumeReport report) {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("runId", report.checkpoint().runId());
        details.put("checkpointStep", report.checkpoint().checkpointStep());
        details.put("datasetFingerprint", report.checkpoint().fingerprint().toMetadata());
        details.put("currentPlanChecked", report.currentPlanChecked());
        details.put("currentPlanGateStatus", report.currentPlanGateStatus());
        details.put("compatibilityMode", report.compatibilityMode().id());
        details.put("forceAccepted", report.forceAccepted());
        return details;
    }

    private static void addCompatibilityWarnings(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.compatibilityWarnings().isEmpty()) {
            return;
        }
        String modeId = report.compatibilityMode().id();
        findings.add(new Finding(
                report.compatibilityMode().force() ? "force-resume-override" : "compatible-resume-warning",
                "resume-compatibility",
                "warning",
                "checkpoint resume was accepted by " + modeId + " mode with compatibility warnings",
                "Audit the compatibility warnings before continuing, and prefer a strict resume when possible.",
                Map.of(
                        "compatibilityMode", modeId,
                        "warnings", report.compatibilityWarnings())));
    }

    private static String expectationCode(String reason) {
        if (reason.startsWith("experimentName expected ")) {
            return "resume-expectation-experiment-name";
        }
        if (reason.startsWith("runId expected ")) {
            return "resume-expectation-run-id";
        }
        if (reason.startsWith("modelFamily expected ")) {
            return "resume-expectation-model-family";
        }
        if (reason.startsWith("seed expected ")) {
            return "resume-expectation-seed";
        }
        if (reason.startsWith("checkpointStep expected >= ")) {
            return "resume-expectation-minimum-step";
        }
        if (reason.startsWith("checkpointStep expected ")) {
            return "resume-expectation-checkpoint-step";
        }
        return "resume-expectation-mismatch";
    }

    private static String explanationSummary(
            DiscreteTokenDatasetCheckpointResumeReport report,
            List<Finding> findings) {
        if (report.ready()) {
            return "resume ready for "
                    + report.checkpoint().runId()
                    + " step "
                    + report.checkpoint().checkpointStep();
        }
        return "resume blocked by "
                + findings.size()
                + " finding(s): "
                + String.join(", ", findings.stream().map(Finding::code).toList());
    }

    private static List<String> nextSteps(List<Finding> findings) {
        List<String> steps = new ArrayList<>();
        for (Finding finding : findings) {
            if (!steps.contains(finding.suggestion())) {
                steps.add(finding.suggestion());
            }
        }
        return List.copyOf(steps);
    }

    private static List<Finding> immutableFindings(List<Finding> findings) {
        if (findings == null || findings.isEmpty()) {
            return List.of();
        }
        return findings.stream()
                .map(finding -> Objects.requireNonNull(finding, "findings entries must not be null"))
                .toList();
    }

    private static List<Finding> findingsFromMetadata(Map<?, ?> metadata, String key) {
        Object value = DiscreteTokenDatasetMetadataSupport.required(metadata, key);
        if (!(value instanceof List<?> values)) {
            throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
        }
        return values.stream()
                .map(entry -> {
                    if (entry instanceof Map<?, ?> map) {
                        return Finding.fromMetadata(map);
                    }
                    throw new IllegalArgumentException("metadata field '" + key + "' entries must be maps");
                })
                .toList();
    }

    public record Finding(
            String code,
            String category,
            String severity,
            String message,
            String suggestion,
            Map<String, Object> details) {

        public Finding {
            code = DiscreteTokenDatasetMetadataSupport.requireText(code, "code");
            category = DiscreteTokenDatasetMetadataSupport.requireText(category, "category");
            severity = DiscreteTokenDatasetMetadataSupport.requireText(severity, "severity");
            message = DiscreteTokenDatasetMetadataSupport.requireText(message, "message");
            suggestion = DiscreteTokenDatasetMetadataSupport.requireText(suggestion, "suggestion");
            details = DiscreteTokenDatasetMetadataSupport.immutableJsonMetadataMap(details, "details");
        }

        public static Finding fromMetadata(Map<?, ?> metadata) {
            Objects.requireNonNull(metadata, "metadata must not be null");
            return new Finding(
                    DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "code"),
                    DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "category"),
                    DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "severity"),
                    DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "message"),
                    DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "suggestion"),
                    DiscreteTokenDatasetMetadataSupport.optionalJsonMetadataMap(metadata, "details"));
        }

        public Map<String, Object> toMetadata() {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("code", code);
            metadata.put("category", category);
            metadata.put("severity", severity);
            metadata.put("message", message);
            metadata.put("suggestion", suggestion);
            if (!details.isEmpty()) {
                metadata.put("details", details);
            }
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }

}
