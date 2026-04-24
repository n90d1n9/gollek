package tech.kayys.gollek.ml.ensemble;

/**
 * Random Forest - bagging ensemble of decision trees.
 * Supports parallel tree building and out-of-bag scoring.
 */
public class RandomForestClassifier extends BaseEstimator {

    private final List<DecisionTreeClassifier> trees = new ArrayList<>();
    private final int nEstimators;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final String criterion;
    private final double maxFeatures;
    private final boolean bootstrap;
    private final int nJobs;
    private final boolean oobScore;

    private double oobScore_;
    private double[][] oobPredictions;
    private int[] oobCounts;

    public RandomForestClassifier() {
        this(100, 10, 2, "gini", "sqrt", true, -1, false);
    }

    public RandomForestClassifier(int nEstimators, int maxDepth, int minSamplesSplit,
            String criterion, String maxFeatures, boolean bootstrap,
            int nJobs, boolean oobScore) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.criterion = criterion;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.nJobs = nJobs > 0 ? nJobs : Runtime.getRuntime().availableProcessors();
        this.oobScore = oobScore;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        int nSamples = X.length;

        // Initialize OOB tracking if needed
        if (oobScore) {
            oobPredictions = new double[nSamples][];
            oobCounts = new int[nSamples];
            for (int i = 0; i < nSamples; i++) {
                oobPredictions[i] = new int[nClasses];
            }
        }

        // Build trees in parallel
        ExecutorService executor = Executors.newFixedThreadPool(nJobs);
        List<Future<DecisionTreeClassifier>> futures = new ArrayList<>();

        for (int i = 0; i < nEstimators; i++) {
            final int seed = i;
            futures.add(executor.submit(() -> {
                DecisionTreeClassifier tree = new DecisionTreeClassifier(
                        maxDepth, minSamplesSplit, 1, criterion, maxFeatures);

                // Bootstrap sampling
                int[] indices;
                int[] oobIndices = null;

                if (bootstrap) {
                    indices = bootstrapSample(nSamples, seed);
                    if (oobScore) {
                        oobIndices = findOobIndices(indices, nSamples);
                    }
                } else {
                    indices = new int[nSamples];
                    for (int j = 0; j < nSamples; j++)
                        indices[j] = j;
                }

                // Extract bootstrap sample
                float[][] Xboot = new float[indices.length][];
                int[] yboot = new int[indices.length];
                for (int j = 0; j < indices.length; j++) {
                    Xboot[j] = X[indices[j]];
                    yboot[j] = y[indices[j]];
                }

                // Train tree
                tree.fit(Xboot, yboot);

                // Update OOB predictions
                if (oobScore && oobIndices != null) {
                    for (int idx : oobIndices) {
                        int pred = tree.predictSingle(X[idx]);
                        synchronized (oobPredictions[idx]) {
                            ((int[]) oobPredictions[idx])[pred]++;
                            oobCounts[idx]++;
                        }
                    }
                }

                return tree;
            }));
        }

        // Collect results
        for (Future<DecisionTreeClassifier> future : futures) {
            try {
                trees.add(future.get());
            } catch (Exception e) {
                throw new RuntimeException("Tree building failed", e);
            }
        }
        executor.shutdown();

        // Calculate OOB score if requested
        if (oobScore) {
            int correct = 0;
            int total = 0;
            for (int i = 0; i < nSamples; i++) {
                if (oobCounts[i] > 0) {
                    int majority = argmax((int[]) oobPredictions[i]);
                    if (majority == y[i])
                        correct++;
                    total++;
                }
            }
            oobScore_ = total > 0 ? (double) correct / total : 0.0;
        }
    }

    /**
     * Bootstrap sampling with replacement.
     */
    private int[] bootstrapSample(int nSamples, int seed) {
        Random rng = new Random(seed);
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = rng.nextInt(nSamples);
        }
        return indices;
    }

    /**
     * Find OOB indices (samples not in bootstrap sample).
     */
    private int[] findOobIndices(int[] bootstrapIndices, int nSamples) {
        boolean[] selected = new boolean[nSamples];
        for (int idx : bootstrapIndices) {
            selected[idx] = true;
        }

        List<Integer> oobList = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            if (!selected[i])
                oobList.add(i);
        }

        return oobList.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Predict by majority voting.
     */
    @Override
    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];

        // Parallel prediction
        Arrays.parallelSetAll(predictions, i -> {
            int[] votes = new int[nClasses];
            for (DecisionTreeClassifier tree : trees) {
                votes[tree.predictSingle(X[i])]++;
            }
            return argmax(votes);
        });

        return predictions;
    }

    /**
     * Predict probabilities (averaged across trees).
     */
    public double[][] predict_proba(float[][] X) {
        double[][] probabilities = new double[X.length][nClasses];

        Arrays.parallelSetAll(probabilities, i -> {
            double[] probs = new double[nClasses];
            for (DecisionTreeClassifier tree : trees) {
                double[] treeProbs = tree.predictProbaSingle(X[i]);
                for (int c = 0; c < nClasses; c++) {
                    probs[c] += treeProbs[c];
                }
            }
            for (int c = 0; c < nClasses; c++) {
                probs[c] /= trees.size();
            }
            return probs;
        });

        return probabilities;
    }

    public double oobScore() {
        return oobScore_;
    }

    public double[] featureImportances() {
        double[] importances = new double[nFeatures];
        for (DecisionTreeClassifier tree : trees) {
            double[] treeImportances = tree.featureImportances();
            for (int i = 0; i < nFeatures; i++) {
                importances[i] += treeImportances[i];
            }
        }
        for (int i = 0; i < nFeatures; i++) {
            importances[i] /= trees.size();
        }
        return importances;
    }
}