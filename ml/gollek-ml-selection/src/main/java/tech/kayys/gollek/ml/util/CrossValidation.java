package tech.kayys.gollek.ml.util;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.metrics.Metrics;
import tech.kayys.gollek.ml.model_selection.ModelSelection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Cross-validation utilities.
 */
public final class CrossValidation {

    private CrossValidation() {
    }

    /**
     * K-Fold cross-validation score.
     */
    public static double crossValScore(BaseEstimator estimator, float[][] x, int[] y, int nFolds, String scoring) {
        ModelSelection.KFold kfold = new ModelSelection.KFold(nFolds);
        List<ModelSelection.Fold> folds = kfold.split(x.length);
        double[] scores = new double[nFolds];

        IntStream.range(0, nFolds).parallel().forEach(foldIdx -> {
            ModelSelection.Fold foldData = folds.get(foldIdx);
            BaseEstimator foldEstimator = estimator.clone();

            float[][] xTrain = new float[foldData.trainIndices.length][];
            int[] yTrain = new int[foldData.trainIndices.length];
            for (int i = 0; i < foldData.trainIndices.length; i++) {
                xTrain[i] = x[foldData.trainIndices[i]];
                yTrain[i] = y[foldData.trainIndices[i]];
            }

            float[][] xVal = new float[foldData.valIndices.length][];
            int[] yVal = new int[foldData.valIndices.length];
            for (int i = 0; i < foldData.valIndices.length; i++) {
                xVal[i] = x[foldData.valIndices[i]];
                yVal[i] = y[foldData.valIndices[i]];
            }

            foldEstimator.fit(xTrain, yTrain);

            if ("accuracy".equals(scoring)) {
                scores[foldIdx] = foldEstimator.score(xVal, yVal);
            } else if ("f1".equals(scoring)) {
                int[] predictions = foldEstimator.predict(xVal);
                scores[foldIdx] = Metrics.f1Score(yVal, predictions, 1);
            }
        });

        return Arrays.stream(scores).average().orElse(0.0);
    }

    /**
     * Grid search for hyperparameter tuning.
     */
    public static GridSearchResult gridSearch(BaseEstimator estimator, Map<String, Object[]> paramGrid,
            float[][] x, int[] y, int nFolds) {
        List<Map<String, Object>> paramSets = generateParameterSets(paramGrid);

        List<GridSearchResult> results = paramSets.parallelStream()
                .map(params -> {
                    BaseEstimator clone = estimator.clone();
                    clone.setParams(params);
                    double score = crossValScore(clone, x, y, nFolds, "accuracy");
                    return new GridSearchResult(params, score);
                })
                .collect(Collectors.toList());

        GridSearchResult best = results.stream()
                .max((a, b) -> Double.compare(a.score, b.score))
                .orElse(null);

        if (best != null) {
            estimator.setParams(best.params);
            estimator.fit(x, y);
        }

        return best;
    }

    private static List<Map<String, Object>> generateParameterSets(Map<String, Object[]> paramGrid) {
        List<Map<String, Object>> combinations = new ArrayList<>();
        combinations.add(new LinkedHashMap<>());

        for (Map.Entry<String, Object[]> entry : paramGrid.entrySet()) {
            String key = entry.getKey();
            Object[] values = entry.getValue();
            List<Map<String, Object>> nextCombinations = new ArrayList<>();

            for (Map<String, Object> combination : combinations) {
                for (Object value : values) {
                    Map<String, Object> next = new LinkedHashMap<>(combination);
                    next.put(key, value);
                    nextCombinations.add(next);
                }
            }
            combinations = nextCombinations;
        }

        return combinations;
    }

    public static final class GridSearchResult {
        public Map<String, Object> params;
        public double score;

        public GridSearchResult(Map<String, Object> params, double score) {
            this.params = params;
            this.score = score;
        }
    }

    public static final class LearningCurve {
        public List<Integer> trainSizes = new ArrayList<>();
        public List<Double> trainScores = new ArrayList<>();
        public List<Double> trainStd = new ArrayList<>();
    }
}
