package tech.kayys.gollek.ml.dummy;

import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * Dummy classifiers for baselines and simple fallbacks.
 */
public class DummyClassifier extends BaseEstimator {

    public enum Strategy {
        MOST_FREQUENT,
        STRATIFIED,
        UNIFORM,
        CONSTANT
    }

    private final Strategy strategy;
    private final Integer constantLabel;
    private int mostFrequentClass;
    private double[] classProbabilities;
    private boolean isFitted = false;

    public DummyClassifier(Strategy strategy) {
        this(strategy, null);
    }

    public DummyClassifier(Strategy strategy, Integer constantLabel) {
        this.strategy = strategy;
        this.constantLabel = constantLabel;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        if (strategy == Strategy.CONSTANT) {
            if (constantLabel == null) {
                throw new IllegalArgumentException("Constant label required for constant strategy");
            }
            mostFrequentClass = constantLabel;
        } else if (strategy == Strategy.MOST_FREQUENT) {
            Map<Integer, Integer> counts = new HashMap<>();
            for (int label : y) {
                counts.put(label, counts.getOrDefault(label, 0) + 1);
            }
            mostFrequentClass = Collections.max(counts.entrySet(), Map.Entry.comparingByValue()).getKey();
        } else if (strategy == Strategy.STRATIFIED || strategy == Strategy.UNIFORM) {
            Map<Integer, Integer> counts = new HashMap<>();
            for (int label : y) {
                counts.put(label, counts.getOrDefault(label, 0) + 1);
            }
            classProbabilities = new double[counts.size()];
            double total = y.length;
            if (strategy == Strategy.STRATIFIED) {
                for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
                    classProbabilities[entry.getKey()] = entry.getValue() / total;
                }
            } else {
                // UNIFORM
                double prob = 1.0 / counts.size();
                Arrays.fill(classProbabilities, prob);
            }
        }
        isFitted = true;
    }

    @Override
    public int[] predict(float[][] X) {
        if (!isFitted)
            throw new IllegalStateException("Not fitted");

        int[] predictions = new int[X.length];
        if (strategy == Strategy.STRATIFIED) {
            Random rng = new Random();
            for (int i = 0; i < X.length; i++) {
                double r = rng.nextDouble();
                double cumsum = 0;
                for (int c = 0; c < classProbabilities.length; c++) {
                    cumsum += classProbabilities[c];
                    if (r <= cumsum) {
                        predictions[i] = c;
                        break;
                    }
                }
            }
        } else {
            Arrays.fill(predictions, mostFrequentClass);
        }
        return predictions;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        if (!isFitted)
            throw new IllegalStateException("Not fitted");

        double[][] probs = new double[X.length][classProbabilities.length];
        for (int i = 0; i < X.length; i++) {
            probs[i] = classProbabilities.clone();
        }
        return probs;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }
}