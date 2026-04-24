package tech.kayys.gollek.ml.model_selection;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Cross-validation and model selection utilities.
 */
public class ModelSelection {

    /**
     * K-Fold cross-validation splitter.
     */
    public static class KFold {
        private final int nSplits;
        private final boolean shuffle;
        private final int randomState;

        public KFold(int nSplits) {
            this(nSplits, false, 42);
        }

        public KFold(int nSplits, boolean shuffle, int randomState) {
            this.nSplits = nSplits;
            this.shuffle = shuffle;
            this.randomState = randomState;
        }

        public List<Fold> split(int nSamples) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < nSamples; i++)
                indices.add(i);

            if (shuffle) {
                Collections.shuffle(indices, new Random(randomState));
            }

            List<Fold> folds = new ArrayList<>();
            int foldSize = nSamples / nSplits;

            for (int fold = 0; fold < nSplits; fold++) {
                int start = fold * foldSize;
                int end = (fold == nSplits - 1) ? nSamples : start + foldSize;

                List<Integer> trainIdx = new ArrayList<>();
                List<Integer> valIdx = new ArrayList<>();

                for (int i = 0; i < nSamples; i++) {
                    if (i >= start && i < end) {
                        valIdx.add(indices.get(i));
                    } else {
                        trainIdx.add(indices.get(i));
                    }
                }

                folds.add(new Fold(
                        trainIdx.stream().mapToInt(i -> i).toArray(),
                        valIdx.stream().mapToInt(i -> i).toArray()));
            }

            return folds;
        }
    }

    /**
     * Stratified K-Fold - preserves class distribution.
     */
    public static class StratifiedKFold {
        private final int nSplits;
        private final int randomState;

        public StratifiedKFold(int nSplits, int randomState) {
            this.nSplits = nSplits;
            this.randomState = randomState;
        }

        public List<Fold> split(float[][] X, int[] y) {
            // Count samples per class
            int nClasses = Arrays.stream(y).max().getAsInt() + 1;
            List<List<Integer>> classIndices = new ArrayList<>();
            for (int c = 0; c < nClasses; c++) {
                classIndices.add(new ArrayList<>());
            }

            for (int i = 0; i < y.length; i++) {
                classIndices.get(y[i]).add(i);
            }

            // Shuffle each class
            Random rng = new Random(randomState);
            for (List<Integer> indices : classIndices) {
                Collections.shuffle(indices, rng);
            }

            // Split each class proportionally
            List<Fold> folds = new ArrayList<>();
            for (int i = 0; i < nSplits; i++) {
                folds.add(new Fold(new int[0], new int[0]));
            }

            for (List<Integer> classIdx : classIndices) {
                int foldSize = classIdx.size() / nSplits;
                for (int fold = 0; fold < nSplits; fold++) {
                    int start = fold * foldSize;
                    int end = (fold == nSplits - 1) ? classIdx.size() : start + foldSize;

                    Fold current = folds.get(fold);
                    for (int j = start; j < end; j++) {
                        current.valIndices.add(classIdx.get(j));
                    }
                    for (int j = 0; j < start; j++) {
                        current.trainIndices.add(classIdx.get(j));
                    }
                    for (int j = end; j < classIdx.size(); j++) {
                        current.trainIndices.add(classIdx.get(j));
                    }
                }
            }

            // Convert to arrays
            return folds.stream()
                    .map(f -> new Fold(
                            f.trainIndices.stream().mapToInt(i -> i).toArray(),
                            f.valIndices.stream().mapToInt(i -> i).toArray()))
                    .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        }
    }

    /**
     * Train-test split utility.
     */
    public static class TrainTestSplit {
        public final float[][] XTrain;
        public final float[][] XTest;
        public final int[] yTrain;
        public final int[] yTest;

        public TrainTestSplit(float[][] XTrain, float[][] XTest, int[] yTrain, int[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    public static TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
        int nSamples = X.length;
        int nTest = (int) (nSamples * testSize);

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++)
            indices.add(i);
        Collections.shuffle(indices, new Random(randomState));

        float[][] XTest = new float[nTest][];
        int[] yTest = new int[nTest];
        float[][] XTrain = new float[nSamples - nTest][];
        int[] yTrain = new int[nSamples - nTest];

        for (int i = 0; i < nSamples; i++) {
            int idx = indices.get(i);
            if (i < nTest) {
                XTest[i] = X[idx];
                yTest[i] = y[idx];
            } else {
                XTrain[i - nTest] = X[idx];
                yTrain[i - nTest] = y[idx];
            }
        }

        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }

    public static class Fold {
        public final int[] trainIndices;
        public final int[] valIndices;

        // For building
        private List<Integer> trainIndicesList = new ArrayList<>();
        private List<Integer> valIndicesList = new ArrayList<>();

        public Fold(int[] trainIndices, int[] valIndices) {
            this.trainIndices = trainIndices;
            this.valIndices = valIndices;
        }

        private Fold() {
            this.trainIndices = null;
            this.valIndices = null;
        }
    }
}