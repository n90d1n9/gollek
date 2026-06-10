package tech.kayys.gollek.ml.train;

/**
 * Controls expensive exact parameter-update diagnostics.
 */
record TrainerParameterUpdateDiagnosticsPolicy(boolean enabled, int intervalSteps) {
    static TrainerParameterUpdateDiagnosticsPolicy disabled() {
        return new TrainerParameterUpdateDiagnosticsPolicy(false, 1);
    }

    static TrainerParameterUpdateDiagnosticsPolicy everyStep() {
        return new TrainerParameterUpdateDiagnosticsPolicy(true, 1);
    }

    static TrainerParameterUpdateDiagnosticsPolicy of(boolean enabled, int intervalSteps) {
        return new TrainerParameterUpdateDiagnosticsPolicy(enabled, intervalSteps);
    }

    TrainerParameterUpdateDiagnosticsPolicy {
        intervalSteps = Math.max(1, intervalSteps);
    }

    boolean shouldCapture(int optimizerStepIndex) {
        return enabled && optimizerStepIndex > 0 && optimizerStepIndex % intervalSteps == 0;
    }
}
