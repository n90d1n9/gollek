package tech.kayys.gollek.ml.bytelatent;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Lightweight trainer-facing configuration for byte-latent language modeling.
 */
public record ByteLatentTrainerConfig(
        ByteLatentModelSpec modelSpec,
        int batchSize,
        int windowLength,
        int padTokenId,
        int epochs,
        boolean shuffle,
        long seed,
        Path checkpointDir,
        boolean resumeFromCheckpoint) {

    public ByteLatentTrainerConfig {
        modelSpec = Objects.requireNonNull(modelSpec, "modelSpec must not be null");
        batchSize = Math.max(1, batchSize);
        windowLength = Math.max(1, windowLength);
        if (padTokenId < 0) {
            throw new IllegalArgumentException("padTokenId must be >= 0 but was " + padTokenId);
        }
        epochs = Math.max(1, epochs);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private ByteLatentModelSpec modelSpec;
        private int batchSize = 32;
        private int windowLength = 1024;
        private int padTokenId = 0;
        private int epochs = 1;
        private boolean shuffle = true;
        private long seed = 0L;
        private Path checkpointDir;
        private boolean resumeFromCheckpoint = false;

        private Builder() {
        }

        public Builder modelSpec(ByteLatentModelSpec modelSpec) {
            this.modelSpec = modelSpec;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder windowLength(int windowLength) {
            this.windowLength = windowLength;
            return this;
        }

        public Builder padTokenId(int padTokenId) {
            this.padTokenId = padTokenId;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
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

        public ByteLatentTrainerConfig build() {
            return new ByteLatentTrainerConfig(
                    modelSpec,
                    batchSize,
                    windowLength,
                    padTokenId,
                    epochs,
                    shuffle,
                    seed,
                    checkpointDir,
                    resumeFromCheckpoint);
        }
    }
}
