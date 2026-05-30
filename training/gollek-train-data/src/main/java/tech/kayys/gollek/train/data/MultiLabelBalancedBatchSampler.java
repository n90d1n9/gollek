package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Produces an index stream arranged as label-balanced multi-label mini-batches.
 *
 * <p>Each row may carry more than one positive label. The sampler therefore
 * balances target label coverage, not mutually-exclusive classes. Replacement
 * mode can oversample rare labels for BCE-style training while keeping the
 * underlying dataset unchanged.</p>
 */
public final class MultiLabelBalancedBatchSampler implements IndexSampler {

    private final float[] labels;
    private final int rows;
    private final int columns;
    private final int batchSize;
    private final int batchesPerEpoch;
    private final boolean replacement;
    private final long seed;
    private final int[] positiveCounts;
    private final List<Integer> activeColumns;

    public MultiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, true, seed);
    }

    public MultiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, replacement, seed);
    }

    public MultiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, true, seed);
    }

    public MultiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, replacement, seed);
    }

    public MultiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, true, seed);
    }

    public MultiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this(validateLabels(labels), batchSize, batchesPerEpoch, replacement, seed);
    }

    MultiLabelBalancedBatchSampler(
            float[] labels,
            int rows,
            int columns,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this(new LabelMatrix(labels, rows, columns), batchSize, batchesPerEpoch, replacement, seed);
    }

    private MultiLabelBalancedBatchSampler(
            LabelMatrix matrix,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        if (batchesPerEpoch <= 0) {
            throw new IllegalArgumentException("batchesPerEpoch must be positive, got: " + batchesPerEpoch);
        }
        int requestedSamples = Math.multiplyExact(batchSize, batchesPerEpoch);
        if (!replacement && requestedSamples > matrix.rows()) {
            throw new IllegalArgumentException(
                    "requested samples cannot exceed label row count without replacement, got: "
                            + requestedSamples + " > " + matrix.rows());
        }

        this.labels = Arrays.copyOf(matrix.values(), matrix.values().length);
        this.rows = matrix.rows();
        this.columns = matrix.columns();
        this.batchSize = batchSize;
        this.batchesPerEpoch = batchesPerEpoch;
        this.replacement = replacement;
        this.seed = seed;
        this.positiveCounts = countPositives(this.labels, rows, columns);
        this.activeColumns = activeColumns(this.positiveCounts);
        if (this.activeColumns.isEmpty()) {
            throw new IllegalArgumentException("multi-label sampler requires at least one positive label");
        }
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireDatasetSize(datasetSize);
        PositiveIndexState state = positiveIndexState();
        return replacement ? sampleWithReplacement(state) : sampleWithoutReplacement(state);
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        return Math.multiplyExact(batchSize, batchesPerEpoch);
    }

    public float[] labels() {
        return Arrays.copyOf(labels, labels.length);
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public int batchSize() {
        return batchSize;
    }

    public int batchesPerEpoch() {
        return batchesPerEpoch;
    }

    public boolean replacement() {
        return replacement;
    }

    public long seed() {
        return seed;
    }

    public int[] positiveCounts() {
        return Arrays.copyOf(positiveCounts, positiveCounts.length);
    }

    public List<Integer> activeColumns() {
        return List.copyOf(activeColumns);
    }

    private List<Integer> sampleWithReplacement(PositiveIndexState state) {
        int count = sampleCount(rows);
        List<Integer> selected = new ArrayList<>(count);
        int[] cursors = new int[columns];
        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            boolean[] batchUsed = new boolean[rows];
            int[] batchCounts = new int[columns];
            List<Integer> batchIndices = new ArrayList<>(batchSize);

            while (batchIndices.size() < batchSize) {
                int column = chooseTargetColumn(batchCounts, batch, batchIndices.size());
                int index = drawPositiveIndex(state.positiveByColumn().get(column), cursors, batchUsed, true,
                        column, batch, batchIndices.size());
                batchIndices.add(index);
                batchUsed[index] = true;
                updateCounts(batchCounts, index);
            }

            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0xCBF29CE484222325L)));
            selected.addAll(batchIndices);
        }
        return selected;
    }

    private List<Integer> sampleWithoutReplacement(PositiveIndexState state) {
        int count = sampleCount(rows);
        List<Integer> selected = new ArrayList<>(count);
        boolean[] used = new boolean[rows];
        int[] cursors = new int[columns];
        int[] allCursor = new int[] {0};

        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            int[] batchCounts = new int[columns];
            List<Integer> batchIndices = new ArrayList<>(batchSize);

            while (batchIndices.size() < batchSize) {
                int column = chooseTargetColumn(batchCounts, batch, batchIndices.size());
                int index = drawPositiveIndex(state.positiveByColumn().get(column), cursors, used, false,
                        column, batch, batchIndices.size());
                if (index < 0) {
                    index = drawAnyUnused(state.allIndices(), allCursor, used);
                }
                if (index < 0) {
                    throw new IllegalStateException("could not produce requested samples without replacement");
                }

                batchIndices.add(index);
                used[index] = true;
                updateCounts(batchCounts, index);
            }

            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0x100000001B3L)));
            selected.addAll(batchIndices);
        }
        return selected;
    }

    private int chooseTargetColumn(int[] batchCounts, int batch, int slot) {
        List<Integer> candidates = new ArrayList<>(activeColumns);
        Collections.shuffle(candidates, new Random(mix(seed, batch, slot, 0xA0761D6478BD642FL)));
        candidates.sort((left, right) -> {
            int byBatchCount = Integer.compare(batchCounts[left], batchCounts[right]);
            if (byBatchCount != 0) {
                return byBatchCount;
            }
            return Integer.compare(positiveCounts[left], positiveCounts[right]);
        });
        return candidates.get(0);
    }

    private int drawPositiveIndex(
            List<Integer> indices,
            int[] cursors,
            boolean[] excluded,
            boolean allowExcluded,
            int column,
            int batch,
            int slot) {
        int size = indices.size();
        for (int attempts = 0; attempts < size; attempts++) {
            if (cursors[column] >= size) {
                Collections.shuffle(indices, new Random(mix(seed, column, batch, slot + cursors[column])));
                cursors[column] = 0;
            }
            int index = indices.get(cursors[column]);
            cursors[column]++;
            if (!excluded[index]) {
                return index;
            }
        }
        if (!allowExcluded) {
            return -1;
        }
        if (cursors[column] >= size) {
            Collections.shuffle(indices, new Random(mix(seed, column, batch, slot + cursors[column])));
            cursors[column] = 0;
        }
        int index = indices.get(cursors[column]);
        cursors[column]++;
        return index;
    }

    private int drawAnyUnused(List<Integer> indices, int[] cursor, boolean[] used) {
        while (cursor[0] < indices.size()) {
            int index = indices.get(cursor[0]);
            cursor[0]++;
            if (!used[index]) {
                return index;
            }
        }
        return -1;
    }

    private void updateCounts(int[] counts, int row) {
        int offset = row * columns;
        for (int column = 0; column < columns; column++) {
            if (labels[offset + column] >= 0.5f) {
                counts[column]++;
            }
        }
    }

    private PositiveIndexState positiveIndexState() {
        List<List<Integer>> byColumn = new ArrayList<>(columns);
        for (int column = 0; column < columns; column++) {
            byColumn.add(new ArrayList<>());
        }
        List<Integer> all = new ArrayList<>(rows);
        for (int row = 0; row < rows; row++) {
            all.add(row);
            int offset = row * columns;
            for (int column = 0; column < columns; column++) {
                if (labels[offset + column] >= 0.5f) {
                    byColumn.get(column).add(row);
                }
            }
        }
        for (int column = 0; column < columns; column++) {
            Collections.shuffle(byColumn.get(column), new Random(mix(seed, column, rows, 0)));
        }
        Collections.shuffle(all, new Random(mix(seed, rows, columns, 0x9E3779B97F4A7C15L)));
        return new PositiveIndexState(byColumn, all);
    }

    private void requireDatasetSize(int datasetSize) {
        if (datasetSize != rows) {
            throw new IllegalArgumentException(
                    "label row count must match dataset size, got: " + rows + " vs " + datasetSize);
        }
    }

    private static int[] countPositives(float[] labels, int rows, int columns) {
        int[] counts = new int[columns];
        for (int row = 0; row < rows; row++) {
            int offset = row * columns;
            for (int column = 0; column < columns; column++) {
                if (labels[offset + column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }

    private static List<Integer> activeColumns(int[] positiveCounts) {
        List<Integer> active = new ArrayList<>();
        for (int column = 0; column < positiveCounts.length; column++) {
            if (positiveCounts[column] > 0) {
                active.add(column);
            }
        }
        return active;
    }

    private static LabelMatrix validateLabels(int[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int rows = labels.length;
        int columns = requireMatrixShape(rows, rows == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
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
        return new LabelMatrix(values, rows, columns);
    }

    private static LabelMatrix validateLabels(boolean[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int rows = labels.length;
        int columns = requireMatrixShape(rows, rows == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            boolean[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                values[row * columns + column] = labelRow[column] ? 1f : 0f;
            }
        }
        return new LabelMatrix(values, rows, columns);
    }

    private static LabelMatrix validateLabels(float[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int rows = labels.length;
        int columns = requireMatrixShape(rows, rows == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            float[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                float value = labelRow[column];
                if (!Float.isFinite(value)
                        || (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f)) {
                    throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
                }
                values[row * columns + column] = value >= 0.5f ? 1f : 0f;
            }
        }
        return new LabelMatrix(values, rows, columns);
    }

    private static int requireMatrixShape(int rows, int columns) {
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

    private static long mix(long seed, long a, long b, long c) {
        long value = seed ^ 0x9E3779B97F4A7C15L;
        value = (value ^ a) * 0xBF58476D1CE4E5B9L;
        value = (value ^ b) * 0x94D049BB133111EBL;
        return value ^ c ^ (value >>> 31);
    }

    private record LabelMatrix(float[] values, int rows, int columns) {
        private LabelMatrix {
            Objects.requireNonNull(values, "labels must not be null");
            if (rows <= 0) {
                throw new IllegalArgumentException("rows must be positive, got: " + rows);
            }
            if (columns <= 0) {
                throw new IllegalArgumentException("columns must be positive, got: " + columns);
            }
            if (values.length != Math.multiplyExact(rows, columns)) {
                throw new IllegalArgumentException(
                        "label value count must equal rows * columns, got: " + values.length);
            }
            values = Arrays.copyOf(values, values.length);
            for (float value : values) {
                if (!Float.isFinite(value)
                        || (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f)) {
                    throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
                }
            }
        }
    }

    private record PositiveIndexState(List<List<Integer>> positiveByColumn, List<Integer> allIndices) {
    }
}
