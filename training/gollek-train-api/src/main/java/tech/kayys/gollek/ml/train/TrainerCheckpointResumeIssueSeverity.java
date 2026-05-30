package tech.kayys.gollek.ml.train;

/**
 * Stable checkpoint resume issue severity values with machine-readable blocking semantics.
 */
enum TrainerCheckpointResumeIssueSeverity {
    WARNING("warning", false),
    ERROR("error", true);

    private final String value;
    private final boolean blocking;

    TrainerCheckpointResumeIssueSeverity(String value, boolean blocking) {
        this.value = value;
        this.blocking = blocking;
    }

    String value() {
        return value;
    }

    boolean blocking() {
        return blocking;
    }
}
