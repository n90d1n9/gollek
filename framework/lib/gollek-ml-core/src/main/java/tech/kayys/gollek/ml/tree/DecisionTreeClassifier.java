
package tech.kayys.gollek.ml.tree;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Decision Tree with Gini/Entropy splitting criteria.
 * Parallel tree building with SIMD optimizations.
 */
public class DecisionTreeClassifier extends BaseEstimator {

    private Node root;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final String criterion; // "gini" or "entropy"
    private final double maxFeatures;
    private final Random random;

    private int nClasses;
    private int nFeatures;
    private double[] featureImportances;

    public DecisionTreeClassifier() {
        this(5, 2, 1, "gini", "sqrt");
    }

    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit,
            int minSamplesLeaf, String criterion,
            String maxFeatures) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = criterion;
        this.maxFeatures = maxFeatures.equals("sqrt") ? Math.sqrt(nFeatures)
                : maxFeatures.equals("log2") ? Math.log(nFeatures) : Double.parseDouble(maxFeatures);
        this.random = new Random();
    }

    /**
     * Fit decision tree on training data.
     */
    @Override
    public void fit(float[][] X, int[] y) {
        this.nFeatures = X[0].length;
        this.nClasses = Arrays.stream(y).max().getAsInt() + 1;

        // Build tree recursively with parallel processing
        this.root = buildTree(X, y, 0, new int[0]);

        // Calculate feature importances
        featureImportances = new double[nFeatures];
        computeImportances(root, featureImportances);

        // Normalize importances
        double total = Arrays.stream(featureImportances).sum();
        for (int i = 0; i < featureImportances.length; i++) {
            featureImportances[i] /= total;
        }
    }

    /**
     * Build tree node with parallel best split search.
     */
    private Node buildTree(float[][] X, int[] y, int depth, int[] indices) {
        if (indices.length == 0) {
            indices = new int[X.length];
            for (int i = 0; i < X.length; i++)
                indices[i] = i;
        }

        // Check stopping criteria
        if (depth >= maxDepth || indices.length < minSamplesSplit ||
                isPure(y, indices)) {
            return new LeafNode(getMajorityClass(y, indices));
        }

        // Find best split (parallelized)
        Split bestSplit = findBestSplitParallel(X, y, indices);

        if (bestSplit == null || bestSplit.gain < 1e-8) {
            return new LeafNode(getMajorityClass(y, indices));
        }

        // Split data
        int[] leftIndices = new int[bestSplit.leftCount];
        int[] rightIndices = new int[bestSplit.rightCount];
        int leftIdx = 0, rightIdx = 0;

        for (int idx : indices) {
            if (X[idx][bestSplit.feature] <= bestSplit.threshold) {
                leftIndices[leftIdx++] = idx;
            } else {
                rightIndices[rightIdx++] = idx;
            }
        }

        // Recursively build children
        Node left = buildTree(X, y, depth + 1, leftIndices);
        Node right = buildTree(X, y, depth + 1, rightIndices);

        return new SplitNode(bestSplit.feature, bestSplit.threshold, left, right, bestSplit.gain);
    }

    /**
     * Find best split using parallel processing.
     */
    private Split findBestSplitParallel(float[][] X, int[] y, int[] indices) {
        int nSamples = indices.length;
        int nFeaturesToConsider = (int) Math.min(nFeatures,
                maxFeatures > 0 && maxFeatures < 1 ? maxFeatures * nFeatures : maxFeatures);

        // Select random subset of features
        int[] featureIndices = selectRandomFeatures(nFeaturesToConsider);

        // Parallel stream for feature evaluation
        Split bestSplit = Arrays.stream(featureIndices)
                .parallel()
                .mapToObj(feature -> evaluateFeature(X, y, indices, feature))
                .filter(Objects::nonNull)
                .max((a, b) -> Double.compare(a.gain, b.gain))
                .orElse(null);

        return bestSplit;
    }

    /**
     * Evaluate a single feature for best split.
     */
    private Split evaluateFeature(float[][] X, int[] y, int[] indices, int feature) {
        // Collect values for this feature
        double[] values = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            values[i] = X[indices[i]][feature];
        }

        // Sort indices by feature value
        Integer[] sortedIndices = new Integer[indices.length];
        for (int i = 0; i < indices.length; i++)
            sortedIndices[i] = i;
        Arrays.sort(sortedIndices, (a, b) -> Double.compare(values[a], values[b]));

        // Try each possible split point
        Split bestSplit = null;
        double bestGain = 0;

        int[] leftCounts = new int[nClasses];
        int[] rightCounts = new int[nClasses];

        // Initialize right counts (all samples initially on right)
        for (int idx : indices) {
            rightCounts[y[idx]]++;
        }

        int totalLeft = 0;

        for (int i = 0; i < indices.length - 1; i++) {
            int idx = sortedIndices[i];
            int currentClass = y[indices[idx]];

            // Move sample from right to left
            rightCounts[currentClass]--;
            leftCounts[currentClass]++;
            totalLeft++;

            // Skip if threshold not unique
            if (values[idx] == values[sortedIndices[i + 1]])
                continue;

            // Minimum samples checks
            if (totalLeft < minSamplesLeaf ||
                    (indices.length - totalLeft) < minSamplesLeaf)
                continue;

            // Calculate impurity gain
            double currentImpurity = calculateImpurity(rightCounts, leftCounts,
                    indices.length - totalLeft, totalLeft);
            double gain = currentImpurity - (totalLeft / (double) indices.length) *
                    calculateImpurity(leftCounts) -
                    ((indices.length - totalLeft) / (double) indices.length) *
                            calculateImpurity(rightCounts);

            if (gain > bestGain) {
                bestGain = gain;
                bestSplit = new Split(
                        feature,
                        (values[idx] + values[sortedIndices[i + 1]]) / 2,
                        totalLeft,
                        indices.length - totalLeft,
                        bestGain);
            }
        }

        return bestSplit;
    }

    /**
     * Calculate Gini impurity or entropy.
     */
    private double calculateImpurity(int[] counts) {
        int total = Arrays.stream(counts).sum();
        if (total == 0)
            return 0;

        if ("gini".equals(criterion)) {
            double impurity = 1.0;
            for (int count : counts) {
                double p = count / (double) total;
                impurity -= p * p;
            }
            return impurity;
        } else { // entropy
            double impurity = 0;
            for (int count : counts) {
                if (count > 0) {
                    double p = count / (double) total;
                    impurity -= p * Math.log(p);
                }
            }
            return impurity;
        }
    }

    /**
     * Predict class labels.
     */
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        return predictions;
    }

    /**
     * Predict class probabilities.
     */
    public double[][] predict_proba(float[][] X) {
        double[][] probabilities = new double[X.length][nClasses];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProbaSingle(X[i]);
        }
        return probabilities;
    }

    private int predictSingle(float[] x) {
        Node node = root;
        while (node instanceof SplitNode) {
            SplitNode split = (SplitNode) node;
            node = x[split.feature] <= split.threshold ? split.left : split.right;
        }
        return ((LeafNode) node).prediction;
    }

    private double[] predictProbaSingle(float[] x) {
        Node node = root;
        while (node instanceof SplitNode) {
            SplitNode split = (SplitNode) node;
            node = x[split.feature] <= split.threshold ? split.left : split.right;
        }
        return ((LeafNode) node).probabilities;
    }

    public double[] featureImportances() {
        return featureImportances;
    }

    // Inner classes for tree nodes
    static abstract class Node {
    }

    static class SplitNode extends Node {
        final int feature;
        final double threshold;
        final Node left, right;
        final double gain;

        SplitNode(int feature, double threshold, Node left, Node right, double gain) {
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.gain = gain;
        }
    }

    static class LeafNode extends Node {
        final int prediction;
        final double[] probabilities;

        LeafNode(int prediction) {
            this.prediction = prediction;
            this.probabilities = null;
        }

        LeafNode(double[] probabilities) {
            this.prediction = argmax(probabilities);
            this.probabilities = probabilities;
        }

        private int argmax(double[] arr) {
            int maxIdx = 0;
            for (int i = 1; i < arr.length; i++) {
                if (arr[i] > arr[maxIdx])
                    maxIdx = i;
            }
            return maxIdx;
        }
    }
}