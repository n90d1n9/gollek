package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Internal implementation for dataset label distribution diagnostics.
 *
 * <p>Keeping these concerns out of {@link DataLoader} keeps the public loader
 * facade small while preserving the existing {@code DataLoader.*Report} API.
 */
final class DistributionDiagnostics {

    private DistributionDiagnostics() {
    }

    static DataLoader.ClassificationDistributionReport classificationDistribution(
            DataLoader.TensorDataLoader loader,
            int numClasses) {
        return classificationDistribution((Iterable<DataLoader.Batch>) loader, numClasses);
    }

    static DataLoader.ClassificationDistributionReport classificationDistribution(
            Iterable<DataLoader.Batch> loader,
            int numClasses) {
        Objects.requireNonNull(loader, "loader must not be null");
        if (numClasses <= 0) {
            throw new IllegalArgumentException("numClasses must be positive, got: " + numClasses);
        }
        int[] totals = new int[numClasses];
        List<int[]> batchCounts = new ArrayList<>();
        int sampleCount = 0;
        for (DataLoader.Batch batch : loader) {
            int[] counts = new int[numClasses];
            for (float label : batch.labels().data()) {
                int classIndex = classIndex(label, numClasses);
                counts[classIndex]++;
                totals[classIndex]++;
                sampleCount++;
            }
            batchCounts.add(counts);
        }
        return new DataLoader.ClassificationDistributionReport(
                sampleCount,
                batchCounts.size(),
                numClasses,
                totals,
                batchCounts);
    }

    static DataLoader.ClassificationDistributionReport classificationDistribution(DataLoader.TensorDataLoader loader) {
        return classificationDistribution((Iterable<DataLoader.Batch>) loader);
    }

    static DataLoader.ClassificationDistributionReport classificationDistribution(Iterable<DataLoader.Batch> loader) {
        Objects.requireNonNull(loader, "loader must not be null");
        int[] totals = new int[0];
        List<int[]> batchCounts = new ArrayList<>();
        int sampleCount = 0;
        for (DataLoader.Batch batch : loader) {
            int[] counts = new int[totals.length];
            for (float label : batch.labels().data()) {
                int classIndex = inferredClassIndex(label);
                if (classIndex >= totals.length) {
                    int newWidth = classIndex + 1;
                    totals = Arrays.copyOf(totals, newWidth);
                    counts = Arrays.copyOf(counts, newWidth);
                }
                counts[classIndex]++;
                totals[classIndex]++;
                sampleCount++;
            }
            batchCounts.add(counts);
        }
        if (totals.length == 0) {
            throw new IllegalArgumentException("loader must contain at least one classification label");
        }
        int numClasses = totals.length;
        List<int[]> normalizedBatchCounts = new ArrayList<>(batchCounts.size());
        for (int[] counts : batchCounts) {
            normalizedBatchCounts.add(Arrays.copyOf(counts, numClasses));
        }
        return new DataLoader.ClassificationDistributionReport(
                sampleCount,
                normalizedBatchCounts.size(),
                numClasses,
                totals,
                normalizedBatchCounts);
    }

    static DataLoader.MultiLabelDistributionReport multiLabelDistribution(
            DataLoader.TensorDataLoader loader,
            int labelCount) {
        return multiLabelDistribution((Iterable<DataLoader.Batch>) loader, labelCount);
    }

    static DataLoader.MultiLabelDistributionReport multiLabelDistribution(
            Iterable<DataLoader.Batch> loader,
            int labelCount) {
        Objects.requireNonNull(loader, "loader must not be null");
        if (labelCount <= 0) {
            throw new IllegalArgumentException("labelCount must be positive, got: " + labelCount);
        }
        int[] positiveTotals = new int[labelCount];
        int[] negativeTotals = new int[labelCount];
        List<int[]> batchPositiveCounts = new ArrayList<>();
        int sampleCount = 0;
        for (DataLoader.Batch batch : loader) {
            float[] labels = batch.labels().data();
            if (labels.length % labelCount != 0) {
                throw new IllegalArgumentException(
                        "batch label data length must be divisible by labelCount, got: "
                                + labels.length + " vs " + labelCount);
            }
            int rows = labels.length / labelCount;
            int[] batchCounts = new int[labelCount];
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < labelCount; column++) {
                    float label = labels[row * labelCount + column];
                    if (isBinaryOne(label)) {
                        batchCounts[column]++;
                        positiveTotals[column]++;
                    } else if (isBinaryZero(label)) {
                        negativeTotals[column]++;
                    } else {
                        throw new IllegalArgumentException("multi-label values must be 0 or 1, got: " + label);
                    }
                }
            }
            sampleCount += rows;
            batchPositiveCounts.add(batchCounts);
        }
        return new DataLoader.MultiLabelDistributionReport(
                sampleCount,
                batchPositiveCounts.size(),
                labelCount,
                positiveTotals,
                negativeTotals,
                batchPositiveCounts);
    }

    static DataLoader.MultiLabelDistributionReport multiLabelDistribution(DataLoader.TensorDataLoader loader) {
        return multiLabelDistribution((Iterable<DataLoader.Batch>) loader);
    }

    static DataLoader.MultiLabelDistributionReport multiLabelDistribution(Iterable<DataLoader.Batch> loader) {
        Objects.requireNonNull(loader, "loader must not be null");
        int labelCount = -1;
        int[] positiveTotals = null;
        int[] negativeTotals = null;
        List<int[]> batchPositiveCounts = new ArrayList<>();
        int sampleCount = 0;
        for (DataLoader.Batch batch : loader) {
            int batchLabelCount = inferMultiLabelWidth(batch.labels());
            if (labelCount < 0) {
                labelCount = batchLabelCount;
                positiveTotals = new int[labelCount];
                negativeTotals = new int[labelCount];
            } else if (batchLabelCount != labelCount) {
                throw new IllegalArgumentException(
                        "all multi-label batches must have label width "
                                + labelCount + ", got: " + batchLabelCount);
            }
            float[] labels = batch.labels().data();
            if (labels.length % labelCount != 0) {
                throw new IllegalArgumentException(
                        "batch label data length must be divisible by label width, got: "
                                + labels.length + " vs " + labelCount);
            }
            int rows = labels.length / labelCount;
            int[] batchCounts = new int[labelCount];
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < labelCount; column++) {
                    float label = labels[row * labelCount + column];
                    if (isBinaryOne(label)) {
                        batchCounts[column]++;
                        positiveTotals[column]++;
                    } else if (isBinaryZero(label)) {
                        negativeTotals[column]++;
                    } else {
                        throw new IllegalArgumentException("multi-label values must be 0 or 1, got: " + label);
                    }
                }
            }
            sampleCount += rows;
            batchPositiveCounts.add(batchCounts);
        }
        if (labelCount < 0) {
            throw new IllegalArgumentException("loader must contain at least one multi-label batch");
        }
        return new DataLoader.MultiLabelDistributionReport(
                sampleCount,
                batchPositiveCounts.size(),
                labelCount,
                positiveTotals,
                negativeTotals,
                batchPositiveCounts);
    }

    static DataLoader.ClassificationDistributionDriftReport classificationDistributionDrift(
            ClassificationDistributionView reference,
            ClassificationDistributionView candidate) {
        Objects.requireNonNull(reference, "reference must not be null");
        Objects.requireNonNull(candidate, "candidate must not be null");
        requireSameWidth(reference.numClasses(), candidate.numClasses(), "numClasses");
        double[] referenceFractions = reference.classFractions();
        double[] candidateFractions = candidate.classFractions();
        return new DataLoader.ClassificationDistributionDriftReport(
                reference.numClasses(),
                referenceFractions,
                candidateFractions,
                fractionDeltas(referenceFractions, candidateFractions),
                missingIndices(reference.classCounts()),
                missingIndices(candidate.classCounts()));
    }

    static DataLoader.MultiLabelDistributionDriftReport multiLabelDistributionDrift(
            MultiLabelDistributionView reference,
            MultiLabelDistributionView candidate) {
        Objects.requireNonNull(reference, "reference must not be null");
        Objects.requireNonNull(candidate, "candidate must not be null");
        requireSameWidth(reference.labelCount(), candidate.labelCount(), "labelCount");
        double[] referenceFractions = reference.positiveFractions();
        double[] candidateFractions = candidate.positiveFractions();
        return new DataLoader.MultiLabelDistributionDriftReport(
                reference.labelCount(),
                referenceFractions,
                candidateFractions,
                fractionDeltas(referenceFractions, candidateFractions),
                reference.labelCardinality(),
                candidate.labelCardinality(),
                missingIndices(reference.positiveCounts()),
                missingIndices(candidate.positiveCounts()));
    }

    static Map<String, Object> classificationMetadata(
            String prefix,
            ClassificationDistributionView report) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(metadataKey(prefix, "sampleCount"), report.sampleCount());
        metadata.put(metadataKey(prefix, "batchCount"), report.batchCount());
        metadata.put(metadataKey(prefix, "numClasses"), report.numClasses());
        metadata.put(metadataKey(prefix, "classCounts"), countVectorAsList(report.classCounts()));
        metadata.put(metadataKey(prefix, "classFractions"), doubleVectorAsList(report.classFractions()));
        metadata.put(metadataKey(prefix, "batchClassCounts"), countVectorsAsLists(report.batchClassCounts()));
        metadata.put(metadataKey(prefix, "majorityClassIndex"), report.majorityClassIndex());
        metadata.put(metadataKey(prefix, "majorityClassCount"), report.majorityClassCount());
        metadata.put(metadataKey(prefix, "minorityClassIndex"), report.minorityClassIndex());
        metadata.put(metadataKey(prefix, "minorityClassCount"), report.minorityClassCount());
        metadata.put(metadataKey(prefix, "missingClassCount"), report.missingClassCount());
        metadata.put(metadataKey(prefix, "imbalanceRatio"), report.imbalanceRatio());
        metadata.put(metadataKey(prefix, "normalizedEntropy"), report.normalizedEntropy());
        metadata.put(metadataKey(prefix, "balancedClassWeights"), floatVectorAsList(report.balancedClassWeights()));
        return Collections.unmodifiableMap(metadata);
    }

    static Map<String, Object> multiLabelMetadata(
            String prefix,
            MultiLabelDistributionView report) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(metadataKey(prefix, "sampleCount"), report.sampleCount());
        metadata.put(metadataKey(prefix, "batchCount"), report.batchCount());
        metadata.put(metadataKey(prefix, "labelCount"), report.labelCount());
        metadata.put(metadataKey(prefix, "positiveCounts"), countVectorAsList(report.positiveCounts()));
        metadata.put(metadataKey(prefix, "negativeCounts"), countVectorAsList(report.negativeCounts()));
        metadata.put(metadataKey(prefix, "positiveFractions"), doubleVectorAsList(report.positiveFractions()));
        metadata.put(metadataKey(prefix, "batchPositiveCounts"), countVectorsAsLists(report.batchPositiveCounts()));
        metadata.put(metadataKey(prefix, "maxPositiveLabelIndex"), report.maxPositiveLabelIndex());
        metadata.put(metadataKey(prefix, "maxPositiveCount"), report.maxPositiveCount());
        metadata.put(metadataKey(prefix, "maxPositiveFraction"), report.maxPositiveFraction());
        metadata.put(metadataKey(prefix, "minPositiveLabelIndex"), report.minPositiveLabelIndex());
        metadata.put(metadataKey(prefix, "minPositiveCount"), report.minPositiveCount());
        metadata.put(metadataKey(prefix, "minPositiveFraction"), report.minPositiveFraction());
        metadata.put(metadataKey(prefix, "zeroPositiveLabelCount"), report.zeroPositiveLabelCount());
        metadata.put(metadataKey(prefix, "allPositiveLabelCount"), report.allPositiveLabelCount());
        metadata.put(metadataKey(prefix, "positiveImbalanceRatio"), report.positiveImbalanceRatio());
        metadata.put(metadataKey(prefix, "labelCardinality"), report.labelCardinality());
        metadata.put(metadataKey(prefix, "labelDensity"), report.labelDensity());
        metadata.put(metadataKey(prefix, "positiveWeights"), floatVectorAsList(report.positiveWeights()));
        return Collections.unmodifiableMap(metadata);
    }

    static Map<String, Object> classificationDriftMetadata(
            String prefix,
            ClassificationDistributionDriftView report) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(metadataKey(prefix, "available"), Boolean.TRUE);
        metadata.put(metadataKey(prefix, "kind"), "classification");
        metadata.put(metadataKey(prefix, "numClasses"), report.numClasses());
        metadata.put(metadataKey(prefix, "referenceFractions"), doubleVectorAsList(report.referenceFractions()));
        metadata.put(metadataKey(prefix, "candidateFractions"), doubleVectorAsList(report.candidateFractions()));
        metadata.put(metadataKey(prefix, "fractionDeltas"), doubleVectorAsList(report.fractionDeltas()));
        metadata.put(metadataKey(prefix, "totalVariationDistance"), report.totalVariationDistance());
        metadata.put(metadataKey(prefix, "maxDeltaClassIndex"), report.maxDeltaClassIndex());
        metadata.put(metadataKey(prefix, "maxAbsoluteFractionDelta"), report.maxAbsoluteFractionDelta());
        metadata.put(metadataKey(prefix, "referenceMissingClasses"), report.referenceMissingClasses());
        metadata.put(metadataKey(prefix, "candidateMissingClasses"), report.candidateMissingClasses());
        return Collections.unmodifiableMap(metadata);
    }

    static Map<String, Object> multiLabelDriftMetadata(
            String prefix,
            MultiLabelDistributionDriftView report) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(metadataKey(prefix, "available"), Boolean.TRUE);
        metadata.put(metadataKey(prefix, "kind"), "multi-label");
        metadata.put(metadataKey(prefix, "labelCount"), report.labelCount());
        metadata.put(metadataKey(prefix, "referencePositiveFractions"),
                doubleVectorAsList(report.referencePositiveFractions()));
        metadata.put(metadataKey(prefix, "candidatePositiveFractions"),
                doubleVectorAsList(report.candidatePositiveFractions()));
        metadata.put(metadataKey(prefix, "positiveFractionDeltas"),
                doubleVectorAsList(report.positiveFractionDeltas()));
        metadata.put(metadataKey(prefix, "meanAbsolutePositiveFractionDelta"),
                report.meanAbsolutePositiveFractionDelta());
        metadata.put(metadataKey(prefix, "maxDeltaLabelIndex"), report.maxDeltaLabelIndex());
        metadata.put(metadataKey(prefix, "maxAbsolutePositiveFractionDelta"),
                report.maxAbsolutePositiveFractionDelta());
        metadata.put(metadataKey(prefix, "referenceLabelCardinality"), report.referenceLabelCardinality());
        metadata.put(metadataKey(prefix, "candidateLabelCardinality"), report.candidateLabelCardinality());
        metadata.put(metadataKey(prefix, "labelCardinalityDelta"), report.labelCardinalityDelta());
        metadata.put(metadataKey(prefix, "referenceZeroPositiveLabels"), report.referenceZeroPositiveLabels());
        metadata.put(metadataKey(prefix, "candidateZeroPositiveLabels"), report.candidateZeroPositiveLabels());
        return Collections.unmodifiableMap(metadata);
    }

    static float positiveWeight(int positives, int total) {
        if (total <= 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int negatives = total - positives;
        if (positives == 0 || negatives == 0) {
            return 1.0f;
        }
        return negatives / (float) positives;
    }

    static float[] balancedClassWeights(int[] counts, int total) {
        float[] weights = new float[counts.length];
        for (int i = 0; i < counts.length; i++) {
            weights[i] = counts[i] == 0 ? 1.0f : total / (float) (counts.length * counts[i]);
        }
        return weights;
    }

    static int[] copyCountVector(int[] counts, String name, int expectedWidth) {
        Objects.requireNonNull(counts, name + " must not be null");
        if (counts.length != expectedWidth) {
            throw new IllegalArgumentException(
                    name + " length must be " + expectedWidth + ", got: " + counts.length);
        }
        int[] copy = Arrays.copyOf(counts, counts.length);
        for (int count : copy) {
            if (count < 0) {
                throw new IllegalArgumentException(name + " must contain non-negative counts, got: " + count);
            }
        }
        return copy;
    }

    static List<int[]> copyCountVectors(List<int[]> counts, String name, int expectedWidth) {
        Objects.requireNonNull(counts, name + " must not be null");
        List<int[]> copies = new ArrayList<>(counts.size());
        for (int i = 0; i < counts.size(); i++) {
            copies.add(copyCountVector(counts.get(i), name + "[" + i + "]", expectedWidth));
        }
        return List.copyOf(copies);
    }

    static double[] copyFractionVector(double[] values, String name, int expectedWidth) {
        double[] copy = copyFiniteVector(values, name, expectedWidth);
        for (double value : copy) {
            if (value < 0.0 || value > 1.0) {
                throw new IllegalArgumentException(name + " must contain values in [0, 1], got: " + value);
            }
        }
        return copy;
    }

    static double[] copyFiniteVector(double[] values, String name, int expectedWidth) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.length != expectedWidth) {
            throw new IllegalArgumentException(
                    name + " length must be " + expectedWidth + ", got: " + values.length);
        }
        double[] copy = Arrays.copyOf(values, values.length);
        for (double value : copy) {
            requireFinite(value, name);
        }
        return copy;
    }

    static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite, got: " + value);
        }
    }

    static int sumCounts(int[] counts) {
        int total = 0;
        for (int count : counts) {
            total += count;
        }
        return total;
    }

    static int maxCountIndex(int[] counts) {
        int index = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > counts[index]) {
                index = i;
            }
        }
        return index;
    }

    static int minCountIndex(int[] counts) {
        int index = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] < counts[index]) {
                index = i;
            }
        }
        return index;
    }

    static int zeroCount(int[] counts) {
        int zeros = 0;
        for (int count : counts) {
            if (count == 0) {
                zeros++;
            }
        }
        return zeros;
    }

    static double[] countFractions(int[] counts, int denominator) {
        double[] fractions = new double[counts.length];
        if (denominator == 0) {
            return fractions;
        }
        for (int i = 0; i < counts.length; i++) {
            fractions[i] = counts[i] / (double) denominator;
        }
        return fractions;
    }

    static double normalizedCountEntropy(int[] counts, int sampleCount) {
        if (sampleCount == 0 || counts.length <= 1) {
            return 0.0;
        }
        double entropy = 0.0;
        for (int count : counts) {
            if (count == 0) {
                continue;
            }
            double probability = count / (double) sampleCount;
            entropy -= probability * Math.log(probability);
        }
        return entropy / Math.log(counts.length);
    }

    static double absoluteSum(double[] values) {
        double total = 0.0;
        for (double value : values) {
            total += Math.abs(value);
        }
        return total;
    }

    static int maxAbsIndex(double[] values) {
        int index = 0;
        for (int i = 1; i < values.length; i++) {
            if (Math.abs(values[i]) > Math.abs(values[index])) {
                index = i;
            }
        }
        return index;
    }

    private static int classIndex(float label, int numClasses) {
        if (!Float.isFinite(label)) {
            throw new IllegalArgumentException("class label must be finite, got: " + label);
        }
        int classIndex = Math.round(label);
        if (Math.abs(label - classIndex) > 1e-6f || classIndex < 0 || classIndex >= numClasses) {
            throw new IllegalArgumentException(
                    "class label " + label + " out of range [0, " + (numClasses - 1) + "]");
        }
        return classIndex;
    }

    private static int inferredClassIndex(float label) {
        if (!Float.isFinite(label)) {
            throw new IllegalArgumentException("class label must be finite, got: " + label);
        }
        int classIndex = Math.round(label);
        if (Math.abs(label - classIndex) > 1e-6f || classIndex < 0) {
            throw new IllegalArgumentException("class label must be a non-negative integer, got: " + label);
        }
        return classIndex;
    }

    private static int inferMultiLabelWidth(GradTensor labels) {
        long[] shape = labels.shape();
        if (shape.length < 2) {
            throw new IllegalArgumentException(
                    "multi-label distribution requires labels with shape [N, C], got: "
                            + Arrays.toString(shape));
        }
        long width = shape[shape.length - 1];
        if (width <= 0 || width > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("multi-label label width must be positive, got: " + width);
        }
        return (int) width;
    }

    private static boolean isBinaryOne(float label) {
        return Float.isFinite(label) && Math.abs(label - 1.0f) <= 1e-6f;
    }

    private static boolean isBinaryZero(float label) {
        return Float.isFinite(label) && Math.abs(label) <= 1e-6f;
    }

    private static void requireSameWidth(int referenceWidth, int candidateWidth, String name) {
        if (referenceWidth != candidateWidth) {
            throw new IllegalArgumentException(
                    name + " must match, got: " + referenceWidth + " vs " + candidateWidth);
        }
    }

    private static double[] fractionDeltas(double[] referenceFractions, double[] candidateFractions) {
        requireSameWidth(referenceFractions.length, candidateFractions.length, "fraction vector width");
        double[] deltas = new double[referenceFractions.length];
        for (int i = 0; i < deltas.length; i++) {
            deltas[i] = candidateFractions[i] - referenceFractions[i];
        }
        return deltas;
    }

    private static List<Integer> missingIndices(int[] counts) {
        List<Integer> missing = new ArrayList<>();
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] == 0) {
                missing.add(i);
            }
        }
        return List.copyOf(missing);
    }

    private static String metadataKey(String prefix, String key) {
        if (prefix == null || prefix.isBlank()) {
            return key;
        }
        return prefix.endsWith(".") ? prefix + key : prefix + "." + key;
    }

    private static List<Integer> countVectorAsList(int[] counts) {
        return Arrays.stream(counts).boxed().toList();
    }

    private static List<List<Integer>> countVectorsAsLists(List<int[]> counts) {
        List<List<Integer>> rows = new ArrayList<>(counts.size());
        for (int[] row : counts) {
            rows.add(countVectorAsList(row));
        }
        return List.copyOf(rows);
    }

    private static List<Double> doubleVectorAsList(double[] values) {
        return Arrays.stream(values).boxed().toList();
    }

    private static List<Float> floatVectorAsList(float[] values) {
        List<Float> floats = new ArrayList<>(values.length);
        for (float value : values) {
            floats.add(value);
        }
        return List.copyOf(floats);
    }
}
