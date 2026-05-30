package tech.kayys.gollek.ml.train;

/**
 * Trainer-level gradient clipping policy.
 *
 * <p>Norm clipping rescales the whole gradient vector, while value clipping
 * clamps individual elements after any norm rescale has been applied.</p>
 */
record TrainerGradientClipConfig(double normThreshold, double valueThreshold) {
    static final TrainerGradientClipConfig DISABLED = new TrainerGradientClipConfig(0.0, 0.0);

    TrainerGradientClipConfig {
        normThreshold = normalize(normThreshold);
        valueThreshold = normalize(valueThreshold);
    }

    static TrainerGradientClipConfig of(double normThreshold, double valueThreshold) {
        TrainerGradientClipConfig config = new TrainerGradientClipConfig(normThreshold, valueThreshold);
        return config.enabled() ? config : DISABLED;
    }

    static TrainerGradientClipConfig norm(double threshold) {
        return of(threshold, 0.0);
    }

    static TrainerGradientClipConfig value(double threshold) {
        return of(0.0, threshold);
    }

    boolean normEnabled() {
        return normThreshold > 0.0;
    }

    boolean valueEnabled() {
        return valueThreshold > 0.0;
    }

    boolean enabled() {
        return normEnabled() || valueEnabled();
    }

    private static double normalize(double threshold) {
        return Double.isFinite(threshold) && threshold > 0.0 ? threshold : 0.0;
    }
}
