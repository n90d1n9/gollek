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
        status = requireText(status, "status");
        summary = requireText(summary, "summary");
        findings = immutableFindings(findings);
        nextSteps = immutableTextList(nextSteps, "nextSteps");
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

    private static List<String> immutableTextList(List<?> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof CharSequence text) {
                        return requireText(text.toString(), name + " entry");
                    }
                    throw new IllegalArgumentException(name + " entries must be strings");
                })
                .toList();
    }

    private static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    public record Finding(
            String code,
            String category,
            String severity,
            String message,
            String suggestion,
            Map<String, Object> details) {

        public Finding {
            code = requireText(code, "code");
            category = requireText(category, "category");
            severity = requireText(severity, "severity");
            message = requireText(message, "message");
            suggestion = requireText(suggestion, "suggestion");
            details = immutableMetadataMap(details, "details");
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

    private static Map<String, Object> immutableMetadataMap(Map<String, Object> metadata, String name) {
        if (metadata == null || metadata.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            String key = requireText(entry.getKey(), name + " key");
            Object value = Objects.requireNonNull(entry.getValue(), name + " field '" + key + "' must not be null");
            copy.put(key, value);
        }
        return Collections.unmodifiableMap(copy);
    }
}
