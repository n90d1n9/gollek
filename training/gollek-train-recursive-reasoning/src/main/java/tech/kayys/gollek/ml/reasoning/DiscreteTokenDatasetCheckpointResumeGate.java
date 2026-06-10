package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Stable checkpoint-resume preflight gate for automation, CI, and extensions.
 */
public record DiscreteTokenDatasetCheckpointResumeGate(
        String id,
        String category,
        boolean accepted,
        String severity,
        String summary,
        Map<String, Object> details) {
    public static final String CHECKPOINT_SCHEMA = "checkpoint-manifest-schema";
    public static final String CHECKPOINT_DATASET = "checkpoint-dataset-readiness";
    public static final String DATASET_FINGERPRINT = "dataset-fingerprint";
    public static final String RESUME_EXPECTATION = "resume-expectation";
    public static final String CURRENT_PLAN = "current-dataset-plan";

    public DiscreteTokenDatasetCheckpointResumeGate {
        id = DiscreteTokenDatasetMetadataSupport.requireText(id, "id");
        category = DiscreteTokenDatasetMetadataSupport.requireText(category, "category");
        severity = requireSeverity(severity);
        summary = DiscreteTokenDatasetMetadataSupport.requireText(summary, "summary");
        details = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(details, "details");
    }

    public static DiscreteTokenDatasetCheckpointResumeGate fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        DiscreteTokenDatasetCheckpointResumeGate gate = new DiscreteTokenDatasetCheckpointResumeGate(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "id"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "category"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "accepted"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "severity"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "summary"),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "details"));
        if (metadata.containsKey("status") && metadata.get("status") != null) {
            String status = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status");
            if (!gate.status().equals(status)) {
                throw new IllegalArgumentException("metadata field 'status' does not match gate status");
            }
        }
        if (metadata.containsKey("attentionRequired") && metadata.get("attentionRequired") != null) {
            boolean attentionRequired = DiscreteTokenDatasetMetadataSupport.requiredBoolean(
                    metadata,
                    "attentionRequired");
            if (gate.attentionRequired() != attentionRequired) {
                throw new IllegalArgumentException(
                        "metadata field 'attentionRequired' does not match gate attention state");
            }
        }
        if (metadata.containsKey("actionRequired") && metadata.get("actionRequired") != null) {
            boolean actionRequired = DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "actionRequired");
            if (gate.actionRequired() != actionRequired) {
                throw new IllegalArgumentException("metadata field 'actionRequired' does not match gate action state");
            }
        }
        if (metadata.containsKey("actionCode") && metadata.get("actionCode") != null) {
            String actionCode = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "actionCode");
            if (!gate.actionCode().equals(actionCode)) {
                throw new IllegalArgumentException("metadata field 'actionCode' does not match gate action code");
            }
        }
        if (metadata.containsKey("actionHint") && metadata.get("actionHint") != null) {
            String actionHint = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "actionHint");
            if (!gate.actionHint().equals(actionHint)) {
                throw new IllegalArgumentException("metadata field 'actionHint' does not match gate action hint");
            }
        }
        return gate;
    }

    public String status() {
        if (!accepted) {
            return "blocked";
        }
        return "warning".equals(severity) ? "warning" : "accepted";
    }

    public boolean blocked() {
        return !accepted;
    }

    public boolean warning() {
        return accepted && "warning".equals(severity);
    }

    public boolean attentionRequired() {
        return blocked() || warning();
    }

    public boolean actionRequired() {
        return blocked();
    }

    public String actionCode() {
        if (!attentionRequired()) {
            return "continue";
        }
        if (blocked()) {
            return switch (id) {
                case CHECKPOINT_SCHEMA -> "refresh-checkpoint-manifest";
                case CHECKPOINT_DATASET -> "inspect-checkpoint-dataset-readiness";
                case DATASET_FINGERPRINT -> "rebuild-or-select-matching-dataset";
                case RESUME_EXPECTATION -> "adjust-resume-expectation";
                case CURRENT_PLAN -> "fix-current-dataset-plan";
                default -> "resolve-resume-gate";
            };
        }
        return switch (id) {
            case CHECKPOINT_DATASET -> "review-forced-checkpoint-dataset";
            case DATASET_FINGERPRINT -> "review-forced-dataset-fingerprint";
            case CURRENT_PLAN -> "review-current-dataset-warning";
            default -> "review-resume-gate-warning";
        };
    }

    public String actionHint() {
        if (!attentionRequired()) {
            return "Continue training from the selected checkpoint.";
        }
        if (blocked()) {
            return switch (id) {
                case CHECKPOINT_SCHEMA -> "Refresh or migrate the checkpoint manifest before resuming.";
                case CHECKPOINT_DATASET ->
                        "Inspect the checkpoint dataset readiness report and regenerate the checkpoint if needed.";
                case DATASET_FINGERPRINT ->
                        "Use the checkpoint dataset plan or rebuild a checkpoint for the current dataset fingerprint.";
                case RESUME_EXPECTATION ->
                        "Update the resume expectation or select a checkpoint that matches it.";
                case CURRENT_PLAN -> "Fix the current dataset readiness issues or use a compatible resume mode.";
                default -> "Resolve the blocked resume gate before continuing.";
            };
        }
        return switch (id) {
            case CHECKPOINT_DATASET -> "Review the forced checkpoint dataset readiness warning before continuing.";
            case DATASET_FINGERPRINT -> "Review the forced dataset fingerprint mismatch before continuing.";
            case CURRENT_PLAN -> "Review current dataset warnings before continuing.";
            default -> "Review the resume warning before continuing.";
        };
    }

    public DiscreteTokenDatasetCheckpointResumeGate requireAccepted() {
        if (!accepted) {
            throw new IllegalStateException("checkpoint resume gate blocked: " + id + " (" + summary + ")");
        }
        return this;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("id", id);
        metadata.put("category", category);
        metadata.put("status", status());
        metadata.put("accepted", accepted);
        metadata.put("attentionRequired", attentionRequired());
        metadata.put("actionRequired", actionRequired());
        metadata.put("severity", severity);
        metadata.put("actionCode", actionCode());
        metadata.put("actionHint", actionHint());
        metadata.put("summary", summary);
        metadata.put("details", details);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static String requireSeverity(String value) {
        value = DiscreteTokenDatasetMetadataSupport.requireText(value, "severity");
        if (!"info".equals(value) && !"warning".equals(value) && !"error".equals(value)) {
            throw new IllegalArgumentException("severity must be info, warning, or error");
        }
        return value;
    }

}
