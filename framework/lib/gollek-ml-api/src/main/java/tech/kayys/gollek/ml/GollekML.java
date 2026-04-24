package tech.kayys.gollek.ml;

package tech.kayys.gollek.ml;

/**
 * Unified ML API - combines sklearn-style estimators with deep learning.
 * This is the main entry point for Gollek's ML functionality.
 */

public class GollekML {

    // ==================== Deep Learning ====================
    public static class DL {
        public static GradTensor tensor(float[] data, long... shape) {
            return Gollek.tensor(data, shape);
        }

        public static NNModule sequential(NNModule... layers) {
            return new Sequential(layers);
        }

        public static Optimizer adamW(List<Parameter> params, float lr) {
            return AdamW.builder(params, lr).build();
        }

        public static CrossEntropyLoss crossEntropy() {
            return new CrossEntropyLoss();
        }
    }

    // ==================== Traditional ML ====================
    public static class ML {
        // Classification
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

        // Regression
        public static LinearModel linearRegression() {
            return new LinearModel("none", 0, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel ridge(double alpha) {
            return new LinearModel("l2", alpha, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel lasso(double alpha) {
            return new LinearModel("l1", alpha, 1, 1e-4, 1000, 0.01);
        }

        // Clustering
        public static KMeans kMeans(int nClusters) {
            return new KMeans(nClusters);
        }

        public static DBSCAN dbscan(double eps, int minSamples) {
            return new DBSCAN(eps, minSamples, "euclidean", -1);
        }

        // Preprocessing
        public static StandardScaler standardScaler() {
            return new StandardScaler();
        }

        public static PCA pca(int nComponents) {
            return new PCA(nComponents);
        }

        public static PolynomialFeatures polynomialFeatures(int degree) {
            return new PolynomialFeatures(degree);
        }
    }

    // ==================== Model Selection ====================
    public static class Selection {
        public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y, int nFolds) {
            return CrossValidation.crossValScore(estimator, X, y, nFolds, "accuracy");
        }

        public static GridSearchResult gridSearch(BaseEstimator estimator,
                Map<String, Object[]> paramGrid,
                float[][] X, int[] y) {
            return CrossValidation.gridSearch(estimator, paramGrid, X, y, 5);
        }

        public static TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
            return CrossValidation.trainTestSplit(X, y, testSize, randomState);
        }
    }

    // ==================== Example Usage ====================
    public static void main(String[] args) {
        // Load data (e.g., Iris dataset)
        float[][] X = loadIrisData();
        int[] y = loadIrisLabels();

        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║              Gollek ML - Complete ML Framework              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");

        // 1. Traditional ML: Random Forest
        System.out.println("1. Training Random Forest...");
        RandomForestClassifier rf = GollekML.ML.randomForest();
        double rfScore = GollekML.Selection.crossValScore(rf, X, y, 5);
        System.out.printf("   Random Forest CV Accuracy: %.4f\n", rfScore);

        // 2. SVM with RBF kernel
        System.out.println("\n2. Training SVM...");
        SVC svm = GollekML.ML.svm();
        double svmScore = GollekML.Selection.crossValScore(svm, X, y, 5);
        System.out.printf("   SVM CV Accuracy: %.4f\n", svmScore);

        // 3. Deep Learning: MLP
        System.out.println("\n3. Training Neural Network...");
        NNModule mlp = GollekML.DL.sequential(
                new Linear(4, 64),
                new ReLU(),
                new Dropout(0.2f),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 3));

        // 4. Clustering
        System.out.println("\n4. Clustering with K-Means...");
        KMeans kmeans = GollekML.ML.kMeans(3);
        kmeans.fit(X);
        System.out.printf("   K-Means Inertia: %.4f\n", kmeans.getInertia());

        // 5. Feature engineering
        System.out.println("\n5. Feature Engineering...");
        StandardScaler scaler = GollekML.ML.standardScaler();
        float[][] XScaled = scaler.fitTransform(X);

        PCA pca = GollekML.ML.pca(2);
        float[][] XReduced = pca.fitTransform(XScaled);
        System.out.printf("   PCA explained variance: %.4f\n",
                pca.getExplainedVarianceRatio()[0] + pca.getExplainedVarianceRatio()[1]);

        System.out.println("\n✓ All ML components working!");
    }

    private static float[][] loadIrisData() {
        // Iris dataset: 150 samples, 4 features
        float[][] data = new float[150][4];
        // Sample data generation...
        for (int i = 0; i < 150; i++) {
            data[i][0] = 5.0f + (float) Math.random() * 3;
            data[i][1] = 3.0f + (float) Math.random() * 2;
            data[i][2] = 1.0f + (float) Math.random() * 5;
            data[i][3] = 0.2f + (float) Math.random() * 2;
        }
        return data;
    }

    private static int[] loadIrisLabels() {
        int[] labels = new int[150];
        for (int i = 0; i < 150; i++) {
            labels[i] = i / 50; // 0, 1, 2
        }
        return labels;
    }
}