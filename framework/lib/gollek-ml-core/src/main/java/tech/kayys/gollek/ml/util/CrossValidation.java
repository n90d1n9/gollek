package tech.kayys.gollek.ml.util;

/**
 * Cross-validation utilities.
 */
public class CrossValidation {

    /**
     * K-Fold cross-validation.
     */
    public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y,
            int nFolds, String scoring) {
        List<Fold> folds = kFoldSplit(X.length, nFolds, 42);
        double[] scores = new double[nFolds];

        // Parallel execution of folds
        IntStream.range(0, nFolds).parallel().forEach(fold -> {
            Fold foldData = folds.get(fold);

            // Create a fresh copy of the estimator
            BaseEstimator foldEstimator = cloneEstimator(estimator);

            // Extract training and validation sets
            float[][] XTrain = new float[foldData.trainIndices.length][];
            int[] yTrain = new int[foldData.trainIndices.length];
            for (int i = 0; i < foldData.trainIndices.length; i++) {
                XTrain[i] = X[foldData.trainIndices[i]];
                yTrain[i] = y[foldData.trainIndices[i]];
            }

            float[][] XVal = new float[foldData.valIndices.length][];
            int[] yVal = new int[foldData.valIndices.length];
            for (int i = 0; i < foldData.valIndices.length; i++) {
                XVal[i] = X[foldData.valIndices[i]];
                yVal[i] = y[foldData.valIndices[i]];
            }

            // Train and evaluate
            foldEstimator.fit(XTrain, yTrain);
            int[] predictions = foldEstimator.predict(XVal);

            if ("accuracy".equals(scoring)) {
                int correct = 0;
                for (int i = 0; i < predictions.length; i++) {
                    if (predictions[i] == yVal[i])
                        correct++;
                }
                scores[fold] = (double) correct / predictions.length;
            } else if ("f1".equals(scoring)) {
                scores[fold] = f1Score(yVal, predictions);
            }
        });

        return Arrays.stream(scores).average().orElse(0.0);
    }

    /**
     * Grid search for hyperparameter tuning.
     */
    public static GridSearchResult gridSearch(BaseEstimator estimator,
            Map<String, Object[]> paramGrid,
            float[][] X, int[] y, int nFolds) {
        List<ParameterSet> paramSets = generateParameterSets(paramGrid);

        // Evaluate each parameter set in parallel
        List<GridSearchResult> results = paramSets.parallelStream()
                .map(params -> {
                    BaseEstimator clone = cloneEstimator(estimator);
                    setParameters(clone, params);
                    double score = crossValScore(clone, X, y, nFolds, "accuracy");
                    return new GridSearchResult(params, score);
                })
                .collect(Collectors.toList());

        // Find best parameters
        GridSearchResult best = results.stream()
                .max((a, b) -> Double.compare(a.score, b.score))
                .orElse(null);

        // Train best model on all data
        if (best != null) {
            setParameters(estimator, best.params);
            estimator.fit(X, y);
        }

        return best;
    }

    /**
     * Learning curve analysis.
     */
    public static LearningCurve learningCurve(BaseEstimator estimator, float[][] X, int[] y,
            int[] trainSizes, int nFolds) {
        LearningCurve curve = new LearningCurve();

        for (int trainSize : trainSizes) {
            double[] scores = new double[nFolds];

            for (int fold = 0; fold < nFolds; fold++) {
                // Sample subset of training data
                int[] indices = randomSample(X.length, trainSize, fold);
                float[][] XSub = new float[trainSize][];
                int[] ySub = new int[trainSize];
                for (int i = 0; i < trainSize; i++) {
                    XSub[i] = X[indices[i]];
                    ySub[i] = y[indices[i]];
                }

                BaseEstimator foldEstimator = cloneEstimator(estimator);
                foldEstimator.fit(XSub, ySub);

                // Cross-validation on subset
                double cvScore = crossValScore(foldEstimator, X, y, 3, "accuracy");
                scores[fold] = cvScore;
            }

            curve.trainSizes.add(trainSize);
            double meanScore = Arrays.stream(scores).average().orElse(0);
            double stdScore = Math.sqrt(Arrays.stream(scores)
                    .map(s -> Math.pow(s - meanScore, 2))
                    .average().orElse(0));
            curve.trainScores.add(meanScore);
            curve.trainStd.add(stdScore);
        }

        return curve;
    }

    // Helper classes
    static class Fold {
        int[] trainIndices;
        int[] valIndices;
    }

    static class GridSearchResult {
        ParameterSet params;
        double score;

        GridSearchResult(ParameterSet params, double score) {
            this.params = params;
            this.score = score;
        }
    }

    static class LearningCurve {
        List<Integer> trainSizes = new ArrayList<>();
        List<Double> trainScores = new ArrayList<>();
        List<Double> trainStd = new ArrayList<>();
    }
}