package tech.kayys.gollek.ml.ensemble;

/**
 * Gradient Boosting Machine - sequential ensemble of weak learners.
 * Supports multiple loss functions and early stopping.
 */
public class GradientBoostingClassifier extends BaseEstimator {

    private final List<DecisionTreeRegressor> trees = new ArrayList<>();
    private final List<Double> treeWeights = new ArrayList<>();
    private final int nEstimators;
    private final double learningRate;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final String loss; // deviance, exponential
    private final double subsample;
    private final int earlyStoppingRounds;

    private double[] initPrediction;
    private double[] trainingLoss;
    private double[] validationLoss;

    public GradientBoostingClassifier() {
        this(100, 0.1, 3, 2, "deviance", 1.0, 10);
    }

    public GradientBoostingClassifier(int nEstimators, double learningRate,
            int maxDepth, int minSamplesSplit,
            String loss, double subsample,
            int earlyStoppingRounds) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.loss = loss;
        this.subsample = subsample;
        this.earlyStoppingRounds = earlyStoppingRounds;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        int nSamples = X.length;
        nClasses = Arrays.stream(y).max().getAsInt() + 1;

        // Initialize with log-odds for binary, class probabilities for multi
        initPrediction = initializePredictions(y);

        // Current predictions (raw scores)
        double[] currentPredictions = initPrediction.clone();

        trainingLoss = new double[nEstimators];

        int bestRound = 0;
        double bestLoss = Double.MAX_VALUE;
        int roundsNoImprove = 0;

        for (int round = 0; round < nEstimators; round++) {
            // Calculate residuals (negative gradient)
            double[] residuals = computeResiduals(y, currentPredictions);

            // Subsample if needed
            int[] indices;
            if (subsample < 1.0) {
                indices = randomSubsample(nSamples, (int) (nSamples * subsample), round);
            } else {
                indices = new int[nSamples];
                for (int i = 0; i < nSamples; i++)
                    indices[i] = i;
            }

            // Train tree on residuals
            float[][] Xsub = new float[indices.length][];
            double[] rsub = new double[indices.length];
            for (int i = 0; i < indices.length; i++) {
                Xsub[i] = X[indices[i]];
                rsub[i] = residuals[indices[i]];
            }

            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit);
            tree.fit(Xsub, rsub);

            // Update predictions with learning rate
            for (int i = 0; i < nSamples; i++) {
                currentPredictions[i] += learningRate * tree.predictSingle(X[i]);
            }

            trees.add(tree);
            treeWeights.add(learningRate);

            // Calculate training loss
            trainingLoss[round] = computeLoss(y, currentPredictions);

            // Early stopping
            if (earlyStoppingRounds > 0) {
                if (trainingLoss[round] < bestLoss - 1e-7) {
                    bestLoss = trainingLoss[round];
                    bestRound = round;
                    roundsNoImprove = 0;
                } else {
                    roundsNoImprove++;
                    if (roundsNoImprove >= earlyStoppingRounds) {
                        // Truncate trees
                        while (trees.size() > bestRound + 1) {
                            trees.remove(trees.size() - 1);
                            treeWeights.remove(treeWeights.size() - 1);
                        }
                        break;
                    }
                }
            }
        }
    }

    /**
     * Initialize predictions based on loss function.
     */
    private double[] initializePredictions(int[] y) {
        double[] init = new double[y.length];

        if (nClasses == 2) { // Binary classification
            double mean = Arrays.stream(y).average().getAsDouble();
            double logOdds = Math.log(mean / (1 - mean));
            Arrays.fill(init, logOdds);
        } else { // Multi-class
            double[] classProbs = new double[nClasses];
            for (int label : y)
                classProbs[label]++;
            for (int i = 0; i < nClasses; i++)
                classProbs[i] /= y.length;
            // One-vs-rest initialization
            for (int i = 0; i < y.length; i++) {
                init[i] = Math.log(classProbs[y[i]] / (1 - classProbs[y[i]]));
            }
        }
        return init;
    }

    /**
     * Compute residuals (negative gradient of loss function).
     */
    private double[] computeResiduals(int[] y, double[] predictions) {
        double[] residuals = new double[y.length];

        if (nClasses == 2) {
            for (int i = 0; i < y.length; i++) {
                double p = 1.0 / (1 + Math.exp(-predictions[i]));
                residuals[i] = y[i] - p;
            }
        } else {
            // Multi-class: one-vs-rest approach
            for (int i = 0; i < y.length; i++) {
                double sumExp = 0;
                double[] probs = new double[nClasses];
                for (int c = 0; c < nClasses; c++) {
                    probs[c] = Math.exp(predictions[i]);
                    sumExp += probs[c];
                }
                for (int c = 0; c < nClasses; c++) {
                    probs[c] /= sumExp;
                }
                residuals[i] = (y[i] == argmax(probs)) ? 1 - probs[y[i]] : -probs[y[i]];
            }
        }

        return residuals;
    }

    private double computeLoss(int[] y, double[] predictions) {
        if (nClasses == 2) {
            double loss = 0;
            for (int i = 0; i < y.length; i++) {
                double p = 1.0 / (1 + Math.exp(-predictions[i]));
                loss -= y[i] * Math.log(p + 1e-15) + (1 - y[i]) * Math.log(1 - p + 1e-15);
            }
            return loss / y.length;
        } else {
            // Multi-class log loss
            double loss = 0;
            for (int i = 0; i < y.length; i++) {
                double sumExp = 0;
                double[] probs = new double[nClasses];
                for (int c = 0; c < nClasses; c++) {
                    probs[c] = Math.exp(predictions[i]);
                    sumExp += probs[c];
                }
                for (int c = 0; c < nClasses; c++) {
                    probs[c] /= sumExp;
                }
                loss -= Math.log(probs[y[i]] + 1e-15);
            }
            return loss / y.length;
        }
    }

    @Override
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];

        for (int i = 0; i < X.length; i++) {
            double pred = initPrediction[i];
            for (int t = 0; t < trees.size(); t++) {
                pred += treeWeights.get(t) * trees.get(t).predictSingle(X[i]);
            }

            if (nClasses == 2) {
                predictions[i] = sigmoid(pred) > 0.5 ? 1 : 0;
            } else {
                predictions[i] = (int) Math.round(pred); // Simplified
            }
        }

        return predictions;
    }

    public double[] stagedPredict(float[][] X, int stage) {
        double[] predictions = initPrediction.clone();
        for (int i = 0; i <= stage && i < trees.size(); i++) {
            for (int j = 0; j < X.length; j++) {
                predictions[j] += treeWeights.get(i) * trees.get(i).predictSingle(X[j]);
            }
        }
        return predictions;
    }

    private double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }
}