package tech.kayys.gollek.ml.ensemble;

/**
 * Soft/Hard voting ensemble that combines multiple estimators.
 */
public class VotingClassifier extends BaseEstimator {
    private final List<BaseEstimator> estimators;
    private final List<String> weights; // "hard" or "soft"
    private final String voting; // "hard" or "soft"
    private final double[] weights; // Weight for each estimator

    public VotingClassifier(List<BaseEstimator> estimators, String voting, double[] weights) {
        this.estimators = estimators;
        this.voting = voting;
        this.weights = weights;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        for (BaseEstimator estimator : estimators) {
            estimator.fit(X, y);
        }
    }

    @Override
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];

        if ("hard".equals(voting)) {
            // Hard voting: majority rule
            for (int i = 0; i < X.length; i++) {
                int[] votes = new int[nClasses];
                for (int e = 0; e < estimators.size(); e++) {
                    int pred = estimators.get(e).predictSingle(X[i]);
                    votes[pred] += weights[e];
                }
                predictions[i] = argmax(votes);
            }
        } else {
            // Soft voting: average probabilities
            double[][] allProbs = new double[nClasses][X.length];

            for (int e = 0; e < estimators.size(); e++) {
                double[][] probs = estimators.get(e).predictProba(X);
                double w = weights[e];
                for (int i = 0; i < X.length; i++) {
                    for (int c = 0; c < nClasses; c++) {
                        allProbs[c][i] += probs[i][c] * w;
                    }
                }
            }

            for (int i = 0; i < X.length; i++) {
                int bestClass = 0;
                double bestProb = allProbs[0][i];
                for (int c = 1; c < nClasses; c++) {
                    if (allProbs[c][i] > bestProb) {
                        bestProb = allProbs[c][i];
                        bestClass = c;
                    }
                }
                predictions[i] = bestClass;
            }
        }

        return predictions;
    }
}