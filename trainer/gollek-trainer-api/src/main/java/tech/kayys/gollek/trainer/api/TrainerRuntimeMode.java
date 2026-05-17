package tech.kayys.gollek.trainer.api;

/**
 * Canonical trainer runtime mode selected by the trainer facade.
 */
public enum TrainerRuntimeMode {
    LEGACY_BRIDGE("legacy-bridge"),
    CANONICAL_FALLBACK("canonical-fallback");

    private final String value;

    TrainerRuntimeMode(String value) {
        this.value = value;
    }

    public String value() {
        return value;
    }
}
