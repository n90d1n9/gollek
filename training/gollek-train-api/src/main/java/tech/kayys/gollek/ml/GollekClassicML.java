package tech.kayys.gollek.ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.base.BaseTransformer;
import tech.kayys.gollek.ml.clustering.DBSCAN;
import tech.kayys.gollek.ml.clustering.KMeans;
import tech.kayys.gollek.ml.ensemble.GradientBoostingClassifier;
import tech.kayys.gollek.ml.ensemble.RandomForestClassifier;
import tech.kayys.gollek.ml.ensemble.VotingClassifier;
import tech.kayys.gollek.ml.export.Benchmark;
import tech.kayys.gollek.ml.export.ModelExporter;
import tech.kayys.gollek.ml.feature.PolynomialFeatures;
import tech.kayys.gollek.ml.hub.HubConfig;
import tech.kayys.gollek.ml.hub.ModelHub;
import tech.kayys.gollek.ml.linear_model.LinearModel;
import tech.kayys.gollek.ml.model_selection.ModelSelection;
import tech.kayys.gollek.ml.naive_bayes.GaussianNB;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.pipeline.PCA;
import tech.kayys.gollek.ml.pipeline.Pipeline;
import tech.kayys.gollek.ml.pipeline.StandardScaler;
import tech.kayys.gollek.ml.svm.SVC;
import tech.kayys.gollek.ml.util.CrossValidation;

/**
 * Shared implementation for the scikit-learn style Gollek facade namespace.
 */
class GollekClassicML {
    public static RandomForestClassifier randomForest() {
        return new RandomForestClassifier();
    }

    public static GradientBoostingClassifier gradientBoosting() {
        return new GradientBoostingClassifier();
    }

    public static SVC svm() {
        return new SVC();
    }

    public static GaussianNB naiveBayes() {
        return new GaussianNB();
    }

    public static VotingClassifier voting(List<BaseEstimator> estimators, String voting) {
        return new VotingClassifier(estimators, voting, null);
    }

    public static LinearModel linearRegression() {
        return new LinearModel("none", 0, 0, 1e-4, 1000, 0.01);
    }

    public static LinearModel ridge(double alpha) {
        return new LinearModel("l2", alpha, 0, 1e-4, 1000, 0.01);
    }

    public static LinearModel lasso(double alpha) {
        return new LinearModel("l1", alpha, 1, 1e-4, 1000, 0.01);
    }

    public static KMeans kMeans(int nClusters) {
        return new KMeans(nClusters);
    }

    public static DBSCAN dbscan(double eps, int minSamples) {
        return new DBSCAN(eps, minSamples, "euclidean", -1);
    }

    public static StandardScaler standardScaler() {
        return new StandardScaler();
    }

    public static PCA pca(int nComponents) {
        return new PCA(nComponents);
    }

    public static PolynomialFeatures polynomialFeatures(int degree) {
        return new PolynomialFeatures(degree);
    }

    public static Pipeline pipeline(BaseEstimator... steps) {
        if (steps.length == 0) {
            throw new IllegalArgumentException("Pipeline must have at least one step");
        }
        List<BaseTransformer> transformers = new ArrayList<>();
        for (int i = 0; i < steps.length - 1; i++) {
            if (!(steps[i] instanceof BaseTransformer)) {
                throw new IllegalArgumentException("All steps except the last must be transformers");
            }
            transformers.add((BaseTransformer) steps[i]);
        }
        return new Pipeline(transformers, steps[steps.length - 1]);
    }
}

class GollekModelSelectionFacade {
    public static double crossValScore(BaseEstimator estimator, float[][] x, int[] y, int nFolds) {
        return CrossValidation.crossValScore(estimator, x, y, nFolds, "accuracy");
    }

    public static CrossValidation.GridSearchResult gridSearch(
            BaseEstimator estimator,
            Map<String, Object[]> paramGrid,
            float[][] x,
            int[] y) {
        return CrossValidation.gridSearch(estimator, paramGrid, x, y, 5);
    }

    public static ModelSelection.TrainTestSplit trainTestSplit(
            float[][] x,
            int[] y,
            double testSize,
            int randomState) {
        return ModelSelection.trainTestSplit(x, y, testSize, randomState);
    }
}

class GollekModelHubFacade {
    public static HubConfig.Builder config() {
        return HubConfig.builder();
    }

    public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
        return ModelHub.loadWeights(modelId);
    }

    public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) throws IOException {
        return ModelHub.loadWeights(modelId, config);
    }

    public static void loadInto(NNModule model, String modelId) throws IOException {
        ModelHub.loadInto(model, modelId);
    }

    public static void loadInto(NNModule model, String modelId, HubConfig config) throws IOException {
        ModelHub.loadInto(model, modelId, config);
    }
}

class GollekModelExportFacade {
    public static ModelExporter.Builder model(NNModule model) {
        return ModelExporter.builder().model(model);
    }

    public static Benchmark benchmark(Object model) {
        return new Benchmark(model);
    }
}
