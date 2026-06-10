package tech.kayys.suling.audio;

public record AudioProcessingOptions(
        boolean enabled,
        boolean removeDcOffset,
        double fadeInSeconds,
        double fadeOutSeconds,
        double gainDb,
        Double peakNormalizeDbfs,
        double maxNormalizeGainDb,
        boolean trimSilence,
        double trimSilenceThresholdDbfs,
        double trimSilencePaddingSeconds) {

    public static AudioProcessingOptions none() {
        return new AudioProcessingOptions(false, false, 0.0, 0.0, 0.0, null, 0.0, false, -48.0, 0.0);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private boolean removeDcOffset = true;
        private double fadeInSeconds;
        private double fadeOutSeconds;
        private double gainDb;
        private Double peakNormalizeDbfs;
        private double maxNormalizeGainDb = 9.0;
        private boolean trimSilence;
        private double trimSilenceThresholdDbfs = -48.0;
        private double trimSilencePaddingSeconds;

        public Builder removeDcOffset(boolean removeDcOffset) {
            this.removeDcOffset = removeDcOffset;
            return this;
        }

        public Builder fadeInSeconds(double fadeInSeconds) {
            this.fadeInSeconds = Math.max(0.0, fadeInSeconds);
            return this;
        }

        public Builder fadeOutSeconds(double fadeOutSeconds) {
            this.fadeOutSeconds = Math.max(0.0, fadeOutSeconds);
            return this;
        }

        public Builder gainDb(double gainDb) {
            this.gainDb = gainDb;
            return this;
        }

        public Builder peakNormalizeDbfs(double peakNormalizeDbfs) {
            this.peakNormalizeDbfs = peakNormalizeDbfs;
            return this;
        }

        public Builder maxNormalizeGainDb(double maxNormalizeGainDb) {
            this.maxNormalizeGainDb = Math.max(0.0, maxNormalizeGainDb);
            return this;
        }

        public Builder trimSilence(boolean trimSilence) {
            this.trimSilence = trimSilence;
            return this;
        }

        public Builder trimSilenceThresholdDbfs(double trimSilenceThresholdDbfs) {
            this.trimSilenceThresholdDbfs = Math.min(-0.1, trimSilenceThresholdDbfs);
            return this;
        }

        public Builder trimSilencePaddingSeconds(double trimSilencePaddingSeconds) {
            this.trimSilencePaddingSeconds = Math.max(0.0, trimSilencePaddingSeconds);
            return this;
        }

        public AudioProcessingOptions build() {
            return new AudioProcessingOptions(
                    true,
                    removeDcOffset,
                    fadeInSeconds,
                    fadeOutSeconds,
                    gainDb,
                    peakNormalizeDbfs,
                    maxNormalizeGainDb,
                    trimSilence,
                    trimSilenceThresholdDbfs,
                    trimSilencePaddingSeconds);
        }
    }
}
