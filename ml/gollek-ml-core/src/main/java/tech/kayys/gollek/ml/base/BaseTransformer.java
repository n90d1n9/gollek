package tech.kayys.gollek.ml.base;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared transformer foundation for preprocessing and feature extraction.
 */
public abstract class BaseTransformer extends BaseEstimator {

    public abstract void fit(float[][] x);

    public abstract float[][] transform(float[][] x);

    public float[][] fitTransform(float[][] x) {
        fit(x);
        return transform(x);
    }

    public float[] transformSingle(float[] x) {
        return transform(new float[][] { x })[0];
    }

    public float[][] inverseTransform(float[][] x) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not implement inverse_transform");
    }

    public List<String> getFeatureNames(List<String> inputFeatures) {
        return new ArrayList<>(inputFeatures);
    }

    public int getNFeaturesOut() {
        return -1;
    }

    @Override
    public void fit(float[][] x, int[] y) {
        fit(x);
    }

    @Override
    public int[] predict(float[][] x) {
        throw new UnsupportedOperationException("Transformers are for transformations, not predictions");
    }
}
