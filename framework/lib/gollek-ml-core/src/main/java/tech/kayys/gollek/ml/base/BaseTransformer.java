package tech.kayys.gollek.ml.base;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for all transformers (preprocessing, feature extraction).
 * Extends BaseEstimator with transformation capabilities.
 */
public abstract class BaseTransformer extends BaseEstimator {

    private boolean isFitted = false;

    /**
     * Fit transformer to data (computes statistics like mean, std).
     */
    public abstract void fit(float[][] X);

    /**
     * Transform data using fitted parameters.
     */
    public abstract float[][] transform(float[][] X);

    /**
     * Fit and transform in one step.
     */
    public float[][] fitTransform(float[][] X) {
        fit(X);
        return transform(X);
    }

    /**
     * Transform a single sample (for online prediction).
     */
    public float[] transformSingle(float[] x) {
        float[][] single = new float[][] { x };
        return transform(single)[0];
    }

    /**
     * Inverse transform (if possible).
     */
    public float[][] inverseTransform(float[][] X) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not implement inverse_transform");
    }

    /**
     * Get feature names after transformation.
     */
    public List<String> getFeatureNames(List<String> inputFeatures) {
        // Default implementation returns input features
        return new ArrayList<>(inputFeatures);
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }

    protected void setFitted(boolean fitted) {
        this.isFitted = fitted;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        // Most transformers don't use labels, but implement this for compatibility
        fit(X);
    }

    @Override
    public int[] predict(float[][] X) {
        throw new UnsupportedOperationException("Transformers are for transformations, not predictions");
    }
}