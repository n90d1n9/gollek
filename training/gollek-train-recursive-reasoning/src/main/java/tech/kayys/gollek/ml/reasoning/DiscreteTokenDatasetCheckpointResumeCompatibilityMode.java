package tech.kayys.gollek.ml.reasoning;

import java.util.Locale;
import java.util.Objects;

/**
 * Compatibility level for checkpoint resume preflight decisions.
 */
public enum DiscreteTokenDatasetCheckpointResumeCompatibilityMode {
    STRICT("strict", false, false, false, false),
    COMPATIBLE("compatible", false, false, true, false),
    FORCE("force", true, true, true, true);

    private final String id;
    private final boolean allowRejectedCheckpointDataset;
    private final boolean allowFingerprintMismatch;
    private final boolean allowWarningBlockedCurrentPlan;
    private final boolean allowRejectedCurrentPlan;

    DiscreteTokenDatasetCheckpointResumeCompatibilityMode(
            String id,
            boolean allowRejectedCheckpointDataset,
            boolean allowFingerprintMismatch,
            boolean allowWarningBlockedCurrentPlan,
            boolean allowRejectedCurrentPlan) {
        this.id = id;
        this.allowRejectedCheckpointDataset = allowRejectedCheckpointDataset;
        this.allowFingerprintMismatch = allowFingerprintMismatch;
        this.allowWarningBlockedCurrentPlan = allowWarningBlockedCurrentPlan;
        this.allowRejectedCurrentPlan = allowRejectedCurrentPlan;
    }

    public String id() {
        return id;
    }

    public boolean allowRejectedCheckpointDataset() {
        return allowRejectedCheckpointDataset;
    }

    public boolean allowFingerprintMismatch() {
        return allowFingerprintMismatch;
    }

    public boolean allowRejectedCurrentPlan() {
        return allowRejectedCurrentPlan;
    }

    public boolean allowCurrentPlanGateStatus(String gateStatus) {
        return allowRejectedCurrentPlan
                || (allowWarningBlockedCurrentPlan && "warning-blocked".equals(gateStatus));
    }

    public boolean force() {
        return this == FORCE;
    }

    public static DiscreteTokenDatasetCheckpointResumeCompatibilityMode fromId(String id) {
        id = Objects.requireNonNull(id, "id must not be null").trim().toLowerCase(Locale.ROOT);
        return switch (id) {
            case "strict" -> STRICT;
            case "compatible", "compat" -> COMPATIBLE;
            case "force", "forced", "unsafe" -> FORCE;
            default -> throw new IllegalArgumentException("unknown checkpoint resume compatibility mode: " + id);
        };
    }

    public static DiscreteTokenDatasetCheckpointResumeCompatibilityMode fromMetadataValue(Object value) {
        if (value == null) {
            return STRICT;
        }
        if (value instanceof DiscreteTokenDatasetCheckpointResumeCompatibilityMode mode) {
            return mode;
        }
        if (value instanceof CharSequence text) {
            return fromId(text.toString());
        }
        throw new IllegalArgumentException("checkpoint resume compatibility mode must be a string");
    }
}
