package tech.kayys.gollek.ml.train;

/**
 * Result of checkpoint compatibility and integrity validation.
 */
record TrainerCheckpointCompatibilityReport(boolean compatible, String error) {
    static TrainerCheckpointCompatibilityReport ok() {
        return new TrainerCheckpointCompatibilityReport(true, null);
    }

    static TrainerCheckpointCompatibilityReport incompatible(String error) {
        return new TrainerCheckpointCompatibilityReport(false, error);
    }
}
