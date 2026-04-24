package tech.kayys.gollek.ml.naive_bayes;

import java.util.*;

/**
 * Gaussian Naive Bayes for continuous features.
 */
public class GaussianNB extends BaseEstimator {
    private double[][] means;
    private double[][] variances;
    private double[] classPriors;
    private int nClasses;
    private double varSmoothing = 1e-9;

    @Override
    public void fit(float[][] X, int[] y) {
        nClasses = Arrays.stream(y).max().getAsInt() + 1;
        int nFeatures = X[0].length;

        means = new double[nClasses][nFeatures];
        variances = new double[nClasses][nFeatures];
        classPriors = new double[nClasses];

        // Collect samples per class
        List<List<float[]>> classSamples = new ArrayList<>(nClasses);
        List<List<Integer>> classIndices = new ArrayList<>(nClasses);
        for (int c = 0; c < nClasses; c++) {
            classSamples.add(new ArrayList<>());
            classIndices.add(new ArrayList<>());
        }

        for (int i = 0; i < X.length; i++) {
            classSamples.get(y[i]).add(X[i]);
            classIndices.get(y[i]).add(i);
        }

        // Compute statistics for each class
        for (int c = 0; c < nClasses; c++) {
            List<float[]> samples = classSamples.get(c);
            classPriors[c] = (double) samples.size() / X.length;

            if (samples.isEmpty())
                continue;

            // Compute means
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (float[] sample : samples) {
                    sum += sample[j];
                }
                means[c][j] = sum / samples.size();
            }

            // Compute variances
            for (int j = 0; j < nFeatures; j++) {
                double sumSq = 0;
                for (float[] sample : samples) {
                    double diff = sample[j] - means[c][j];
                    sumSq += diff * diff;
                }
                variances[c][j] = sumSq / samples.size() + varSmoothing;
            }
        }
    }

    @Override
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        return predictions;
    }

    public int predictSingle(float[] x) {
        double[] logProbs = predictLogProba(x);
        int maxIdx = 0;
        for (int i = 1; i < logProbs.length; i++) {
            if (logProbs[i] > logProbs[maxIdx])
                maxIdx = i;
        }
        return maxIdx;
    }

    public double[][] predictProba(float[][] X) {
        double[][] probabilities = new double[X.length][nClasses];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProbaSingle(X[i]);
        }
        return probabilities;
    }

    public double[] predictLogProba(float[] x) {
        double[] logProbs = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            logProbs[c] = Math.log(classPriors[c]);
            for (int j = 0; j < x.length; j++) {
                double diff = x[j] - means[c][j];
                logProbs[c] -= 0.5 * Math.log(2 * Math.PI * variances[c][j]);
                logProbs[c] -= 0.5 * diff * diff / variances[c][j];
            }
        }
        return logProbs;
    }

    public double[] predictProbaSingle(float[] x) {
        double[] logProbs = predictLogProba(x);
        double maxLog = Arrays.stream(logProbs).max().getAsDouble();

        double[] probs = new double[nClasses];
        double sum = 0;
        for (int c = 0; c < nClasses; c++) {
            probs[c] = Math.exp(logProbs[c] - maxLog);
            sum += probs[c];
        }
        for (int c = 0; c < nClasses; c++) {
            probs[c] /= sum;
        }
        return probs;
    }
}

/**
 * Multinomial Naive Bayes for discrete features (e.g., text counts).
 */
public class MultinomialNB extends BaseEstimator {
    private double[][] classLogProbs;
    private double[] classPriors;
    private double alpha = 1.0; // Laplace smoothing
    private int nClasses;

    @Override
    public void fit(float[][] X, int[] y) {
        nClasses = Arrays.stream(y).max().getAsInt() + 1;
        int nFeatures = X[0].length;

        classLogProbs = new double[nClasses][nFeatures];
        classPriors = new double[nClasses];

        double[] featureSumPerClass = new double[nClasses];
        double[][] featureCounts = new double[nClasses][nFeatures];

        // Count features per class
        for (int i = 0; i < X.length; i++) {
            int c = y[i];
            classPriors[c]++;
            for (int j = 0; j < nFeatures; j++) {
                featureCounts[c][j] += X[i][j];
                featureSumPerClass[c] += X[i][j];
            }
        }

        // Normalize to probabilities with Laplace smoothing
        for (int c = 0; c < nClasses; c++) {
            classPriors[c] /= X.length;
            double denominator = featureSumPerClass[c] + alpha * nFeatures;

            for (int j = 0; j < nFeatures; j++) {
                double prob = (featureCounts[c][j] + alpha) / denominator;
                classLogProbs[c][j] = Math.log(prob);
            }
        }
    }

    @Override
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        return predictions;
    }

    private int predictSingle(float[] x) {
        double[] logProbs = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            logProbs[c] = Math.log(classPriors[c]);
            for (int j = 0; j < x.length; j++) {
                if (x[j] > 0) {
                    logProbs[c] += x[j] * classLogProbs[c][j];
                }
            }
        }
        int maxIdx = 0;
        for (int i = 1; i < nClasses; i++) {
            if (logProbs[i] > logProbs[maxIdx])
                maxIdx = i;
        }
        return maxIdx;
    }
}