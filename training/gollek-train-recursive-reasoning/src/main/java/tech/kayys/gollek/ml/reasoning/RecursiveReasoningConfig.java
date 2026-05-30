package tech.kayys.gollek.ml.reasoning;

import java.nio.file.Path;

/**
 * Lightweight rollout/trainer-facing configuration for recursive reasoning experiments.
 */
public record RecursiveReasoningConfig(
        int transitionsPerSupervisionStep,
        int supervisionSteps,
        int parallelSamples,
        long seed,
        Path checkpointDir,
        boolean resumeFromCheckpoint) {

    public RecursiveReasoningConfig {
        transitionsPerSupervisionStep = Math.max(1, transitionsPerSupervisionStep);
        supervisionSteps = Math.max(1, supervisionSteps);
        parallelSamples = Math.max(1, parallelSamples);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int transitionsPerSupervisionStep = 16;
        private int supervisionSteps = 1;
        private int parallelSamples = 1;
        private long seed = 0L;
        private Path checkpointDir;
        private boolean resumeFromCheckpoint;

        private Builder() {
        }

        public Builder transitionsPerSupervisionStep(int transitionsPerSupervisionStep) {
            this.transitionsPerSupervisionStep = transitionsPerSupervisionStep;
            return this;
        }

        public Builder supervisionSteps(int supervisionSteps) {
            this.supervisionSteps = supervisionSteps;
            return this;
        }

        public Builder parallelSamples(int parallelSamples) {
            this.parallelSamples = parallelSamples;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder checkpointDir(Path checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        public Builder resumeFromCheckpoint(boolean resumeFromCheckpoint) {
            this.resumeFromCheckpoint = resumeFromCheckpoint;
            return this;
        }

        public RecursiveReasoningConfig build() {
            return new RecursiveReasoningConfig(
                    transitionsPerSupervisionStep,
                    supervisionSteps,
                    parallelSamples,
                    seed,
                    checkpointDir,
                    resumeFromCheckpoint);
        }
    }
}
