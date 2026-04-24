package tech.kayys.gollek.ml.feature_selection;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.metrics.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Mutual Information for feature selection.
 */
public class MutualInfo extends BaseTransformer {
    private double[] scores;
    private int[] selectedIndices;
    private int nFeaturesToSelect;

    public MutualInfo(int nFeaturesToSelect) {
        this.nFeaturesToSelect = nFeaturesToSelect;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        int nFeatures = X[0].length;
        scores = new double[nFeatures];

        // Calculate mutual information for each feature
        IntStream.range(0, nFeatures).parallel().forEach(i -> {
            scores[i] = mutualInformation(X, y, i);
        });

        // Select top features
        Integer[] indices = new Integer[nFeatures];
        for (int i = 0; i < nFeatures; i++)
            indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        selectedIndices = Arrays.copyOf(indices, Math.min(nFeaturesToSelect, nFeatures));
        setFitted(true);
    }

    private double mutualInformation(float[][] X, int[] y, int featureIdx) {
        // Discretize continuous features
        float[] values = new float[X.length];
        for (int i = 0; i < X.length; i++) {
            values[i] = X[i][featureIdx];
        }

        // Use 10 bins for discretization
        int nBins = 10;
        float min = Arrays.stream(values).min().getAsFloat();
        float max = Arrays.stream(values).max().getAsFloat();
        float step = (max - min) / nBins;

        int[][] joint = new int[nBins][];
        int[] marginalX = new int[nBins];
        int[] marginalY = new int[Arrays.stream(y).max().getAsInt() + 1];

        for (int i = 0; i < nBins; i++) {
            joint[i] = new int[marginalY.length];
        }

        for (int i = 0; i < X.length; i++) {
            int binX = Math.min(nBins - 1, (int) ((values[i] - min) / step));
            int binY = y[i];
            joint[binX][binY]++;
            marginalX[binX]++;
            marginalY[binY]++;
        }

        double mi = 0;
        int total = X.length;
        for (int i = 0; i < nBins; i++) {
            for (int j = 0; j < marginalY.length; j++) {
                if (joint[i][j] > 0) {
                    double pxy = joint[i][j] / (double) total;
                    double px = marginalX[i] / (double) total;
                    double py = marginalY[j] / (double) total;
                    mi += pxy * Math.log(pxy / (px * py) + 1e-10);
                }
            }
        }

        return mi;
    }

    @Override
    public float[][] transform(float[][] X) {
        float[][] transformed = new float[X.length][selectedIndices.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                transformed[i][j] = X[i][selectedIndices[j]];
            }
        }
        return transformed;
    }

    public double[] getScores() {
        return scores;
    }

    public int[] getSelectedIndices() {
        return selectedIndices;
    }
}

/**
 * Recursive Feature Elimination (RFE).
 */
public class RFE extends BaseTransformer {
    private final BaseEstimator estimator;
    private final int nFeaturesToSelect;
    private final double step;
    private int[] selectedIndices;
    private List<Integer> ranking;

    public RFE(BaseEstimator estimator, int nFeaturesToSelect) {
        this(estimator, nFeaturesToSelect, 1.0);
    }

    public RFE(BaseEstimator estimator, int nFeaturesToSelect, double step) {
        this.estimator = estimator;
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.step = step;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        int nFeatures = X[0].length;
        List<Integer> remaining = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++)
            remaining.add(i);

        ranking = new ArrayList<>(Collections.nCopies(nFeatures, 0));
        int rank = 1;

        while (remaining.size() > nFeaturesToSelect) {
            int nToRemove = Math.max(1, (int) (remaining.size() * step));

            // Get feature importances from the estimator
            estimator.fit(X, y);
            double[] importances = getFeatureImportances();

            // Find least important features
            List<Integer> sorted = new ArrayList<>(remaining);
            sorted.sort((a, b) -> Double.compare(importances[a], importances[b]));

            for (int i = 0; i < nToRemove && i < sorted.size(); i++) {
                int feature = sorted.get(i);
                ranking.set(feature, rank);
                remaining.remove((Integer) feature);
            }

            rank++;

            // Remove features and continue
            float[][] XReduced = new float[X.length][remaining.size()];
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < remaining.size(); j++) {
                    XReduced[i][j] = X[i][remaining.get(j)];
                }
            }
            X = XReduced;
        }

        // Set final selected indices
        selectedIndices = remaining.stream().mapToInt(i -> i).toArray();

        // Final fit on selected features
        float[][] XFinal = new float[X.length][selectedIndices.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                XFinal[i][j] = X[i][j];
            }
        }
        estimator.fit(XFinal, y);
        setFitted(true);
    }

    private double[] getFeatureImportances() {
        if (estimator instanceof RandomForestClassifier) {
            return ((RandomForestClassifier) estimator).featureImportances();
        } else if (estimator instanceof LinearModel) {
            double[] coef = ((LinearModel) estimator).getCoefficients();
            double[] absCoef = new double[coef.length];
            for (int i = 0; i < coef.length; i++) {
                absCoef[i] = Math.abs(coef[i]);
            }
            return absCoef;
        } else if (estimator instanceof SVC) {
            // For linear SVM, use coefficients
            return new double[0];
        }
        throw new UnsupportedOperationException("Estimator does not provide feature importances");
    }

    @Override
    public float[][] transform(float[][] X) {
        float[][] transformed = new float[X.length][selectedIndices.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                transformed[i][j] = X[i][selectedIndices[j]];
            }
        }
        return transformed;
    }

    public int[] getRanking() {
        return ranking.stream().mapToInt(i -> i).toArray();
    }

    public int[] getSupport() {
        int[] support = new int[ranking.size()];
        for (int i = 0; i < ranking.size(); i++) {
            support[i] = ranking.get(i) == 0 ? 1 : 0;
        }
        return support;
    }
}