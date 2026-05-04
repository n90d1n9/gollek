package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * Pipeline that chains multiple transformers and a final estimator.
 * Implements the scikit-learn Pipeline API.
 */
public class Pipeline extends BaseEstimator {

    private final List<BaseTransformer> transformers;
    private final BaseEstimator finalEstimator;
    private List<String> featureNames;
    private boolean isFitted = false;

    public Pipeline(List<BaseTransformer> transformers, BaseEstimator finalEstimator) {
        this.transformers = new ArrayList<>(transformers);
        this.finalEstimator = finalEstimator;
    }

    public Pipeline(BaseTransformer[] transformers, BaseEstimator finalEstimator) {
        this(Arrays.asList(transformers), finalEstimator);
    }

    @Override
    public void fit(float[][] X, int[] y) {
        float[][] current = X;

        // Apply all transformers
        for (int i = 0; i < transformers.size(); i++) {
            BaseTransformer transformer = transformers.get(i);
            transformer.fit(current);
            current = transformer.transform(current);

            // Update feature names if transformer provides them
            if (i == transformers.size() - 1 && transformer.getFeatureNames(null) != null) {
                this.featureNames = transformer.getFeatureNames(this.featureNames);
            }
        }

        // Fit final estimator
        finalEstimator.fit(current, y);
        isFitted = true;
    }

    @Override
    public int[] predict(float[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("Pipeline must be fitted before prediction");
        }

        float[][] current = X;
        for (BaseTransformer transformer : transformers) {
            current = transformer.transform(current);
        }

        return finalEstimator.predict(current);
    }

    @Override
    public double[][] predictProba(float[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("Pipeline must be fitted before prediction");
        }

        float[][] current = X;
        for (BaseTransformer transformer : transformers) {
            current = transformer.transform(current);
        }

        return finalEstimator.predictProba(current);
    }

    /**
     * Get transformer by name or index.
     */
    public BaseTransformer getTransformer(int index) {
        if (index < 0 || index >= transformers.size()) {
            throw new IndexOutOfBoundsException("Transformer index out of range");
        }
        return transformers.get(index);
    }

    /**
     * Get the final estimator.
     */
    public BaseEstimator getFinalEstimator() {
        return finalEstimator;
    }

    /**
     * Get feature names after all transformations.
     */
    public List<String> getFeatureNames() {
        if (featureNames == null) {
            featureNames = new ArrayList<>();
            if (transformers.isEmpty()) {
                return featureNames;
            }
            // Try to get from last transformer
            BaseTransformer last = transformers.get(transformers.size() - 1);
            featureNames = last.getFeatureNames(featureNames);
        }
        return Collections.unmodifiableList(featureNames);
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Pipeline(steps=[");
        for (int i = 0; i < transformers.size(); i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(transformers.get(i).getClass().getSimpleName());
        }
        sb.append(", ").append(finalEstimator.getClass().getSimpleName()).append("])");
        return sb.toString();
    }
}