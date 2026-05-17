package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * DataLoader — batches, shuffles, and yields data in mini-batches.
 *
 * <p>Supports:
 * <ul>
 *   <li>Batching with configurable batch size</li>
 *   <li>Shuffling (on/off)</li>
 *   <li>Drop last incomplete batch</li>
 *   <li>Custom collate function</li>
 *   <li>Element mapping via {@link #map(Function)}</li>
 * </ul>
 *
 * <h3>Generic usage</h3>
 * <pre>{@code
 * Dataset<Integer> dataset = ...;
 * var loader = new DataLoader<>(dataset, 32);
 * for (List<Integer> batch : loader) {
 *     // process batch
 * }
 * }</pre>
 *
 * <h3>Tensor (training) usage</h3>
 * <pre>{@code
 * var tensorDataset = new TensorDataset(inputs, targets);
 * var loader = DataLoader.tensorBuilder(tensorDataset)
 *     .batchSize(32)
 *     .shuffle(true)
 *     .dropLast(true)
 *     .build();
 * for (List<GradTensor> batch : loader) {
 *     var input = batch.get(0);
 *     var target = batch.get(1);
 * }
 * }</pre>
 *
 * @param <T> the type of elements produced by this data loader
 * @author Gollek Team
 * @version 0.2.0
 */
public class DataLoader<T> implements Iterable<List<T>> {

    private final Dataset<? extends T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;

    // ── Constructors ─────────────────────────────────────────────────────

    /**
     * Create a DataLoader with the given batch size (no shuffle, no drop-last).
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize) {
        this(dataset, batchSize, false, false);
    }

    /**
     * Create a DataLoader with full configuration.
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     * @param shuffle   whether to shuffle indices each epoch
     * @param dropLast  whether to drop the last incomplete batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
    }

    // ── Iteration ────────────────────────────────────────────────────────

    @Override
    public Iterator<List<T>> iterator() {
        int n = dataset.size();
        // Create a mutable copy — toList() returns an unmodifiable list
        List<Integer> indices = new ArrayList<>(IntStream.range(0, n).boxed().toList());

        if (shuffle) {
            Collections.shuffle(indices, ThreadLocalRandom.current());
        }

        int numBatches = numBatches();

        return new Iterator<>() {
            private int current = 0;

            @Override
            public boolean hasNext() {
                return current < numBatches;
            }

            @Override
            public List<T> next() {
                if (current >= numBatches) throw new NoSuchElementException();

                int start = current * batchSize;
                int end = Math.min(start + batchSize, n);

                List<T> batch = new ArrayList<>(end - start);
                for (int i = start; i < end; i++) {
                    batch.add(dataset.get(indices.get(i)));
                }

                current++;
                return batch;
            }
        };
    }

    /** Get number of batches per epoch. */
    public int numBatches() {
        int n = dataset.size();
        return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
    }

    // ── Mapping ──────────────────────────────────────────────────────────

    /**
     * Create a new DataLoader that applies a mapping function to each element.
     *
     * <pre>{@code
     * DataLoader<String> mapped = new DataLoader<>(intDataset, 5)
     *     .map(i -> "val_" + i);
     * }</pre>
     *
     * @param mapper function to apply to each element
     * @param <R>    the output type
     * @return a new DataLoader producing mapped elements
     */
    public <R> DataLoader<R> map(Function<? super T, ? extends R> mapper) {
        Dataset<? extends T> srcDataset = this.dataset;
        Dataset<R> mappedDataset = new Dataset<>() {
            @Override
            public R get(int index) {
                return mapper.apply(srcDataset.get(index));
            }

            @Override
            public int size() {
                return srcDataset.size();
            }
        };
        return new DataLoader<>(mappedDataset, this.batchSize, this.shuffle, this.dropLast);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TensorDataLoader — GradTensor-specific builder with collation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a builder for a tensor-specific DataLoader that supports
     * custom collation and stacking of {@link GradTensor} batches.
     *
     * @param dataset the tensor dataset
     * @return a new builder
     */
    public static TensorBuilder tensorBuilder(TensorDatasetAdapter dataset) {
        return new TensorBuilder(dataset);
    }

    public static TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return new TensorDataset(inputs, labels);
    }

    public static GradTensor classLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length);
    }

    public static float[] classWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int maxLabel = -1;
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            maxLabel = Math.max(maxLabel, label);
        }
        return classWeightsFor(maxLabel + 1, labels);
    }

    public static float[] classWeightsFor(int numClasses, int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (numClasses <= 0) {
            throw new IllegalArgumentException("numClasses must be positive, got: " + numClasses);
        }
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int[] counts = new int[numClasses];
        for (int label : labels) {
            if (label < 0 || label >= numClasses) {
                throw new IllegalArgumentException(
                        "class label " + label + " out of range [0, " + (numClasses - 1) + "]");
            }
            counts[label]++;
        }
        return balancedClassWeights(counts, labels.length);
    }

    public static GradTensor binaryLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] != 0 && labels[i] != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            values[i] = labels[i] ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            float value = labels[i];
            if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
            }
            values[i] = value >= 0.5f ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(int[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static GradTensor binaryLabels(boolean[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static GradTensor binaryLabels(float[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static float binaryPositiveWeight(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (int label : labels) {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + label);
            }
            positives += label;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float binaryPositiveWeight(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (boolean label : labels) {
            positives += label ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float binaryPositiveWeight(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (float label : labels) {
            if (Math.abs(label) > 1e-6f && Math.abs(label - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + label);
            }
            positives += label >= 0.5f ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float[] multiLabelPositiveWeights(int[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static float[] multiLabelPositiveWeights(boolean[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static float[] multiLabelPositiveWeights(float[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.length) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.length);
        }
        return tensorDataset(inputs, classLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.length) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.length);
        }
        return tensorDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    private static TensorDataset binaryDataset(GradTensor inputs, GradTensor labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.shape()[0]) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.shape()[0]);
        }
        return tensorDataset(inputs, labels);
    }

    public static TensorDataLoader tensors(GradTensor inputs, GradTensor labels, int batchSize) {
        return tensorBuilder(tensorDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader classification(GradTensor inputs, int[] labels, int batchSize) {
        return tensorBuilder(classificationDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, int[] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, int[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, boolean[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, float[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDatasetSplit split(GradTensor inputs, GradTensor labels, double trainFraction, long seed) {
        return tensorDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return classificationDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return stratifiedSplit(classificationDataset(inputs, labels), labels, trainFraction, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return stratifiedSplit(binaryDataset(inputs, labels), labels, trainFraction, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
                matrix,
                trainFraction,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
                matrix,
                trainFraction,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
                matrix,
                trainFraction,
                seed);
    }

    private static TensorDatasetSplit stratifiedSplit(
            TensorDataset dataset,
            int[] labels,
            double trainFraction,
            long seed) {
        requireTrainFraction(trainFraction);
        if (labels.length != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.length + " vs " + dataset.size());
        }
        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            groups.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        }

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, entry.getKey())));
            int trainCount = stratifiedTrainCount(group.size(), trainFraction);
            trainIndices.addAll(group.subList(0, trainCount));
            validationIndices.addAll(group.subList(trainCount, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x6A09E667F3BCC909L));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xBB67AE8584CAA73BL));
        return splitByIndices(dataset, trainIndices, validationIndices);
    }

    private static TensorDatasetSplit multiLabelStratifiedSplit(
            TensorDataset dataset,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        requireTrainFraction(trainFraction);
        if (labels.rows() != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.rows() + " vs " + dataset.size());
        }

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        int[] trainTargets = new int[columns];
        for (int column = 0; column < columns; column++) {
            trainTargets[column] = stratifiedTrainCount(totalPositives[column], trainFraction);
        }

        int trainTargetSize = stratifiedTrainCount(rows, trainFraction);
        int validationTargetSize = rows - trainTargetSize;
        int[] trainCounts = new int[columns];
        List<Integer> order = multiLabelStratificationOrder(flatLabels, rows, columns, totalPositives, seed);
        List<Integer> trainIndices = new ArrayList<>(trainTargetSize);
        List<Integer> validationIndices = new ArrayList<>(validationTargetSize);

        for (int row : order) {
            if (trainIndices.size() >= trainTargetSize) {
                validationIndices.add(row);
                continue;
            }
            if (validationIndices.size() >= validationTargetSize) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
                continue;
            }

            double trainPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size() + 1,
                    trainTargetSize, true);
            double validationPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size(),
                    trainTargetSize, false);
            if (trainPenalty <= validationPenalty) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
            } else {
                validationIndices.add(row);
            }
        }

        improveMultiLabelAssignment(
                flatLabels, columns, totalPositives, trainTargets, trainIndices, validationIndices, trainCounts);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x3C6EF372FE94F82AL));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xA54FF53A5F1D36F1L));
        return splitByIndices(dataset, trainIndices, validationIndices);
    }

    private static int[] binaryLabelValues(BinaryLabelMatrix labels) {
        int[] values = new int[labels.values().length];
        for (int i = 0; i < labels.values().length; i++) {
            values[i] = labels.values()[i] >= 0.5f ? 1 : 0;
        }
        return values;
    }

    private static int[] columnPositiveCounts(int[] flatLabels, int rows, int columns) {
        int[] counts = new int[columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                counts[column] += flatLabels[row * columns + column];
            }
        }
        return counts;
    }

    private static List<Integer> multiLabelStratificationOrder(
            int[] flatLabels,
            int rows,
            int columns,
            int[] totalPositives,
            long seed) {
        List<Integer> order = new ArrayList<>(IntStream.range(0, rows).boxed().toList());
        Map<Integer, Integer> tieBreakers = new HashMap<>();
        Collections.shuffle(order, new Random(seed));
        for (int i = 0; i < order.size(); i++) {
            tieBreakers.put(order.get(i), i);
        }
        order.sort((left, right) -> {
            int rarity = Double.compare(
                    rowRarityScore(flatLabels, right, columns, totalPositives),
                    rowRarityScore(flatLabels, left, columns, totalPositives));
            if (rarity != 0) {
                return rarity;
            }
            int cardinality = Integer.compare(
                    rowCardinality(flatLabels, right, columns),
                    rowCardinality(flatLabels, left, columns));
            if (cardinality != 0) {
                return cardinality;
            }
            return Integer.compare(tieBreakers.get(left), tieBreakers.get(right));
        });
        return order;
    }

    private static double rowRarityScore(int[] flatLabels, int row, int columns, int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < columns; column++) {
            if (flatLabels[row * columns + column] == 1) {
                score += totalPositives[column] == 0 ? 0.0 : 1.0 / totalPositives[column];
            }
        }
        return score;
    }

    private static int rowCardinality(int[] flatLabels, int row, int columns) {
        int cardinality = 0;
        for (int column = 0; column < columns; column++) {
            cardinality += flatLabels[row * columns + column];
        }
        return cardinality;
    }

    private static double multiLabelSplitPenalty(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[] trainCounts,
            int[] trainTargets,
            int trainSize,
            int trainTargetSize,
            boolean assignToTrain) {
        double penalty = Math.pow(trainSize - trainTargetSize, 2) * columns;
        for (int column = 0; column < columns; column++) {
            int after = trainCounts[column];
            if (assignToTrain) {
                after += flatLabels[row * columns + column];
            }
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            penalty += Math.pow(after - trainTargets[column], 2) * labelWeight;
        }
        return penalty;
    }

    private static void addRowCounts(int[] flatLabels, int row, int columns, int[] counts, int delta) {
        for (int column = 0; column < columns; column++) {
            counts[column] += flatLabels[row * columns + column] * delta;
        }
    }

    private static void improveMultiLabelAssignment(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[] trainTargets,
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            int[] trainCounts) {
        double bestScore = multiLabelAssignmentScore(trainCounts, trainTargets, totalPositives);
        boolean improved;
        do {
            improved = false;
            int bestTrainPosition = -1;
            int bestValidationPosition = -1;
            int[] bestCounts = null;

            for (int trainPosition = 0; trainPosition < trainIndices.size(); trainPosition++) {
                int trainRow = trainIndices.get(trainPosition);
                for (int validationPosition = 0; validationPosition < validationIndices.size(); validationPosition++) {
                    int validationRow = validationIndices.get(validationPosition);
                    int[] candidateCounts = swappedTrainCounts(
                            flatLabels, columns, trainCounts, trainRow, validationRow);
                    double candidateScore = multiLabelAssignmentScore(candidateCounts, trainTargets, totalPositives);
                    if (candidateScore + 1e-9 < bestScore) {
                        bestScore = candidateScore;
                        bestTrainPosition = trainPosition;
                        bestValidationPosition = validationPosition;
                        bestCounts = candidateCounts;
                        improved = true;
                    }
                }
            }

            if (improved) {
                int trainRow = trainIndices.get(bestTrainPosition);
                trainIndices.set(bestTrainPosition, validationIndices.get(bestValidationPosition));
                validationIndices.set(bestValidationPosition, trainRow);
                System.arraycopy(bestCounts, 0, trainCounts, 0, trainCounts.length);
            }
        } while (improved);
    }

    private static int[] swappedTrainCounts(
            int[] flatLabels,
            int columns,
            int[] trainCounts,
            int trainRow,
            int validationRow) {
        int[] candidate = trainCounts.clone();
        addRowCounts(flatLabels, trainRow, columns, candidate, -1);
        addRowCounts(flatLabels, validationRow, columns, candidate, 1);
        return candidate;
    }

    private static double multiLabelAssignmentScore(
            int[] trainCounts,
            int[] trainTargets,
            int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < trainTargets.length; column++) {
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            score += Math.pow(trainCounts[column] - trainTargets[column], 2) * labelWeight;
        }
        return score;
    }

    private static int stratifiedTrainCount(int groupSize, double trainFraction) {
        if (groupSize <= 0) {
            return 0;
        }
        if (groupSize == 1) {
            return 1;
        }
        int trainCount = (int) Math.round(groupSize * trainFraction);
        return Math.max(1, Math.min(groupSize - 1, trainCount));
    }

    private static void rebalanceNonEmptySplit(List<Integer> trainIndices, List<Integer> validationIndices) {
        if (trainIndices.isEmpty() && !validationIndices.isEmpty()) {
            trainIndices.add(validationIndices.remove(validationIndices.size() - 1));
        }
        if (validationIndices.isEmpty() && trainIndices.size() > 1) {
            validationIndices.add(trainIndices.remove(trainIndices.size() - 1));
        }
    }

    private static TensorDatasetSplit splitByIndices(
            TensorDataset dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        List<GradTensor[]> train = new ArrayList<>(trainIndices.size());
        List<GradTensor[]> validation = new ArrayList<>(validationIndices.size());
        for (int index : trainIndices) {
            train.add(dataset.get(index));
        }
        for (int index : validationIndices) {
            validation.add(dataset.get(index));
        }
        return new TensorDatasetSplit(new TensorDataset(train), new TensorDataset(validation));
    }

    private static void requireTrainFraction(double trainFraction) {
        if (trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be between 0 and 1");
        }
    }

    private static long mixSeed(long seed, int label) {
        long value = seed ^ (0x9E3779B97F4A7C15L * (label + 0x632BE5ABL));
        value ^= (value >>> 33);
        value *= 0xff51afd7ed558ccdL;
        value ^= (value >>> 33);
        return value;
    }

    private static float[] positiveWeights(BinaryLabelMatrix matrix) {
        int[] positives = new int[matrix.columns()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {
                if (matrix.values()[row * matrix.columns() + column] >= 0.5f) {
                    positives[column]++;
                }
            }
        }
        float[] weights = new float[matrix.columns()];
        for (int column = 0; column < matrix.columns(); column++) {
            weights[column] = positiveWeight(positives[column], matrix.rows());
        }
        return weights;
    }

    private static float positiveWeight(int positives, int total) {
        if (total <= 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int negatives = total - positives;
        if (positives == 0 || negatives == 0) {
            return 1.0f;
        }
        return negatives / (float) positives;
    }

    private static float[] balancedClassWeights(int[] counts, int total) {
        float[] weights = new float[counts.length];
        for (int i = 0; i < counts.length; i++) {
            weights[i] = counts[i] == 0 ? 1.0f : total / (float) (counts.length * counts[i]);
        }
        return weights;
    }

    private static BinaryLabelMatrix binaryLabelMatrix(int[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            int[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                int value = labelRow[column];
                if (value != 0 && value != 1) {
                    throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + value);
                }
                values[row * columns + column] = value;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static BinaryLabelMatrix binaryLabelMatrix(boolean[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            boolean[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                values[row * columns + column] = labelRow[column] ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static BinaryLabelMatrix binaryLabelMatrix(float[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            float[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                float value = labelRow[column];
                if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                    throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
                }
                values[row * columns + column] = value >= 0.5f ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static int matrixColumns(int rows, int columns) {
        if (rows == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one row");
        }
        if (columns == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one column");
        }
        return columns;
    }

    private static int rowLength(int[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(boolean[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(float[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static void requireRowWidth(int row, int actual, int expected) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "binary label matrix must be rectangular; row "
                            + row + " has " + actual + " columns, expected " + expected);
        }
    }

    private record BinaryLabelMatrix(float[] values, int rows, int columns) {
    }

    /**
     * Adapter interface for tensor datasets that return arrays of GradTensors.
     * This is the specialized interface for training workflows where each
     * sample is a tuple of tensors (e.g., [input, target]).
     */
    public interface TensorDatasetAdapter {
        /** Total number of samples. */
        int size();

        /** Get number of samples for given indices. */
        default int size(List<Integer> indices) {
            return indices.size();
        }

        /** Get sample at given index. Returns array of tensors (e.g., [input, target]). */
        GradTensor[] get(int index);
    }

    /**
     * Simple dataset from arrays of GradTensor.
     */
    public static class TensorDataset implements TensorDatasetAdapter {
        private final List<GradTensor[]> samples;

        public TensorDataset(GradTensor[]... samples) {
            this.samples = List.copyOf(Arrays.asList(samples));
        }

        private TensorDataset(List<GradTensor[]> samples) {
            this.samples = List.copyOf(samples);
        }

        /**
         * Create from separate input and target tensors.
         * Each input[i] and target[i] form a sample pair.
         */
        public TensorDataset(GradTensor inputs, GradTensor targets) {
            if (inputs.shape().length == 0 || targets.shape().length == 0) {
                throw new IllegalArgumentException("inputs and targets must include a batch dimension");
            }
            if (inputs.shape()[0] != targets.shape()[0]) {
                throw new IllegalArgumentException("inputs and targets must have same batch dimension");
            }
            int n = (int) inputs.shape()[0];
            long[] inputShape = inputs.shape();
            long[] targetShape = targets.shape();
            samples = new ArrayList<>(n);

            for (int i = 0; i < n; i++) {
                // Extract individual samples
                int inputStride = 1;
                for (int d = 1; d < inputShape.length; d++) inputStride *= inputShape[d];
                int targetStride = 1;
                for (int d = 1; d < targetShape.length; d++) targetStride *= targetShape[d];

                float[] inputData = inputs.data();
                float[] targetData = targets.data();
                int inputOffset = i * inputStride;
                int targetOffset = i * targetStride;

                long[] inputSampleShape = Arrays.copyOfRange(inputShape, 1, inputShape.length);
                long[] targetSampleShape = Arrays.copyOfRange(targetShape, 1, targetShape.length);

                float[] inputSample = Arrays.copyOfRange(inputData, inputOffset, inputOffset + inputStride);
                float[] targetSample = Arrays.copyOfRange(targetData, targetOffset, targetOffset + targetStride);

                samples.add(new GradTensor[]{
                    GradTensor.of(inputSample, inputSampleShape),
                    GradTensor.of(targetSample, targetSampleShape)
                });
            }
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public GradTensor[] get(int index) {
            return samples.get(index);
        }

        public TensorDatasetSplit split(double trainFraction, long seed) {
            if (trainFraction <= 0.0 || trainFraction >= 1.0) {
                throw new IllegalArgumentException("trainFraction must be between 0 and 1");
            }
            List<Integer> indices = new ArrayList<>(IntStream.range(0, size()).boxed().toList());
            Collections.shuffle(indices, new Random(seed));
            int trainSize = (int) Math.round(size() * trainFraction);
            trainSize = Math.max(1, Math.min(size() - 1, trainSize));
            List<GradTensor[]> train = new ArrayList<>(trainSize);
            List<GradTensor[]> validation = new ArrayList<>(size() - trainSize);
            for (int i = 0; i < indices.size(); i++) {
                GradTensor[] sample = samples.get(indices.get(i));
                if (i < trainSize) {
                    train.add(sample);
                } else {
                    validation.add(sample);
                }
            }
            return new TensorDatasetSplit(new TensorDataset(train), new TensorDataset(validation));
        }
    }

    public record TensorDatasetSplit(TensorDataset train, TensorDataset validation) {
        public TensorDatasetSplit {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }

        public TensorBuilder trainLoader() {
            return tensorBuilder(train);
        }

        public TensorBuilder validationLoader() {
            return tensorBuilder(validation);
        }

        public TensorDataLoader trainLoader(int batchSize) {
            return trainLoader().batchSize(batchSize).build();
        }

        public TensorDataLoader trainLoader(int batchSize, boolean shuffle, long seed) {
            return trainLoader().batchSize(batchSize).shuffle(shuffle).seed(seed).build();
        }

        public TensorDataLoader validationLoader(int batchSize) {
            return validationLoader().batchSize(batchSize).build();
        }
    }

    /**
     * Builder for tensor-specific DataLoader with collation support.
     */
    public static class TensorBuilder {
        private final TensorDatasetAdapter dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private boolean dropLast = false;
        private Long shuffleSeed;
        private CollateFn collateFn;

        TensorBuilder(TensorDatasetAdapter dataset) {
            this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        }

        public TensorBuilder batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
            }
            this.batchSize = batchSize;
            return this;
        }

        public TensorBuilder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }

        public TensorBuilder dropLast(boolean dropLast) {
            this.dropLast = dropLast;
            return this;
        }

        public TensorBuilder seed(long seed) {
            this.shuffleSeed = seed;
            return this;
        }

        public TensorBuilder randomSeed(long seed) {
            return seed(seed);
        }

        public TensorBuilder collateFn(CollateFn collateFn) {
            this.collateFn = collateFn;
            return this;
        }

        public TensorDataLoader build() {
            return new TensorDataLoader(this);
        }
    }

    /**
     * Tensor-specific DataLoader that handles collation and stacking
     * of {@link GradTensor} batches.
     */
    public static class TensorDataLoader implements Iterable<Batch> {
        private final TensorDatasetAdapter dataset;
        private final int batchSize;
        private final boolean shuffle;
        private final boolean dropLast;
        private final Long shuffleSeed;
        private final CollateFn collateFn;

        private TensorDataLoader(TensorBuilder builder) {
            this.dataset = Objects.requireNonNull(builder.dataset, "dataset must not be null");
            this.batchSize = builder.batchSize;
            this.shuffle = builder.shuffle;
            this.dropLast = builder.dropLast;
            this.shuffleSeed = builder.shuffleSeed;
            this.collateFn = builder.collateFn != null ? builder.collateFn : defaultCollate();
        }

        @Override
        public Iterator<Batch> iterator() {
            int n = dataset.size();
            List<Integer> indices = new ArrayList<>(IntStream.range(0, n).boxed().toList());

            if (shuffle) {
                if (shuffleSeed == null) {
                    Collections.shuffle(indices, ThreadLocalRandom.current());
                } else {
                    Collections.shuffle(indices, new Random(shuffleSeed));
                }
            }

            int numBatches = numBatches();

            return new Iterator<>() {
                private int current = 0;

                @Override
                public boolean hasNext() {
                    return current < numBatches;
                }

                @Override
                public Batch next() {
                    if (current >= numBatches) throw new NoSuchElementException();

                    int start = current * batchSize;
                    int end = Math.min(start + batchSize, n);
                    List<Integer> batchIndices = indices.subList(start, end);

                    current++;
                    return collateFn.collate(batchIndices, dataset);
                }
            };
        }

        /** Get number of batches per epoch. */
        public int numBatches() {
            int n = dataset.size();
            return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
        }

        public int size() {
            return dataset.size();
        }

        public int batchSize() {
            return batchSize;
        }

        public boolean shuffle() {
            return shuffle;
        }

        public boolean dropLast() {
            return dropLast;
        }

        /** Default collate function: stacks tensors into batched tensors. */
        private static CollateFn defaultCollate() {
            return (indices, ds) -> {
                int numSamples = ds.size(indices);
                int numTensors = ds.get(0).length;
                if (numTensors < 2) {
                    throw new IllegalStateException("TensorDataset must have at least 2 tensors (input and label)");
                }

                GradTensor[] inputSamples = new GradTensor[numSamples];
                GradTensor[] labelSamples = new GradTensor[numSamples];

                for (int i = 0; i < numSamples; i++) {
                    GradTensor[] sample = ds.get(indices.get(i));
                    inputSamples[i] = sample[0];
                    labelSamples[i] = sample[1];
                }

                return new Batch(
                    GradTensor.stack(0, inputSamples),
                    GradTensor.stack(0, labelSamples)
                );
            };
        }
    }

    /**
     * Collate function interface for tensor datasets.
     */
    @FunctionalInterface
    public interface CollateFn {
        Batch collate(List<Integer> indices, TensorDatasetAdapter dataset);
    }

    /**
     * A record representing a mini-batch of data.
     *
     * @param inputs input tensor (usually [N, ...])
     * @param labels target/label tensor (usually [N, ...])
     */
    public record Batch(GradTensor inputs, GradTensor labels) {
        /** @return inputs tensor */
        @Override public GradTensor inputs() { return inputs; }
        /** @return labels tensor */
        @Override public GradTensor labels() { return labels; }
    }
}
