package tech.kayys.gollek.train.data;

import java.util.List;
import java.util.Objects;

final class DistributionReportStates {

    private DistributionReportStates() {
    }

    static ClassificationReportState classificationReport(
            int sampleCount,
            int batchCount,
            int numClasses,
            int[] classCounts,
            List<int[]> batchClassCounts) {
        requireNonNegative(sampleCount, "sampleCount");
        requireNonNegative(batchCount, "batchCount");
        requirePositive(numClasses, "numClasses");
        int[] safeClassCounts = DistributionDiagnostics.copyCountVector(classCounts, "classCounts", numClasses);
        List<int[]> safeBatchCounts = DistributionDiagnostics.copyCountVectors(
                batchClassCounts,
                "batchClassCounts",
                numClasses);
        if (safeBatchCounts.size() != batchCount) {
            throw new IllegalArgumentException(
                    "batchClassCounts size must match batchCount, got: "
                            + safeBatchCounts.size() + " vs " + batchCount);
        }
        if (DistributionDiagnostics.sumCounts(safeClassCounts) != sampleCount) {
            throw new IllegalArgumentException("classCounts must sum to sampleCount");
        }
        return new ClassificationReportState(safeClassCounts, safeBatchCounts);
    }

    static MultiLabelReportState multiLabelReport(
            int sampleCount,
            int batchCount,
            int labelCount,
            int[] positiveCounts,
            int[] negativeCounts,
            List<int[]> batchPositiveCounts) {
        requireNonNegative(sampleCount, "sampleCount");
        requireNonNegative(batchCount, "batchCount");
        requirePositive(labelCount, "labelCount");
        int[] safePositiveCounts = DistributionDiagnostics.copyCountVector(
                positiveCounts,
                "positiveCounts",
                labelCount);
        int[] safeNegativeCounts = DistributionDiagnostics.copyCountVector(
                negativeCounts,
                "negativeCounts",
                labelCount);
        List<int[]> safeBatchCounts = DistributionDiagnostics.copyCountVectors(
                batchPositiveCounts,
                "batchPositiveCounts",
                labelCount);
        if (safeBatchCounts.size() != batchCount) {
            throw new IllegalArgumentException(
                    "batchPositiveCounts size must match batchCount, got: "
                            + safeBatchCounts.size() + " vs " + batchCount);
        }
        for (int column = 0; column < labelCount; column++) {
            if (safePositiveCounts[column] + safeNegativeCounts[column] != sampleCount) {
                throw new IllegalArgumentException(
                        "positiveCounts + negativeCounts must equal sampleCount for label " + column);
            }
        }
        return new MultiLabelReportState(safePositiveCounts, safeNegativeCounts, safeBatchCounts);
    }

    static ClassificationDriftState classificationDrift(
            int numClasses,
            double[] referenceFractions,
            double[] candidateFractions,
            double[] fractionDeltas,
            List<Integer> referenceMissingClasses,
            List<Integer> candidateMissingClasses) {
        requirePositive(numClasses, "numClasses");
        return new ClassificationDriftState(
                DistributionDiagnostics.copyFractionVector(referenceFractions, "referenceFractions", numClasses),
                DistributionDiagnostics.copyFractionVector(candidateFractions, "candidateFractions", numClasses),
                DistributionDiagnostics.copyFiniteVector(fractionDeltas, "fractionDeltas", numClasses),
                immutableList(referenceMissingClasses, "referenceMissingClasses"),
                immutableList(candidateMissingClasses, "candidateMissingClasses"));
    }

    static MultiLabelDriftState multiLabelDrift(
            int labelCount,
            double[] referencePositiveFractions,
            double[] candidatePositiveFractions,
            double[] positiveFractionDeltas,
            double referenceLabelCardinality,
            double candidateLabelCardinality,
            List<Integer> referenceZeroPositiveLabels,
            List<Integer> candidateZeroPositiveLabels) {
        requirePositive(labelCount, "labelCount");
        DistributionDiagnostics.requireFinite(referenceLabelCardinality, "referenceLabelCardinality");
        DistributionDiagnostics.requireFinite(candidateLabelCardinality, "candidateLabelCardinality");
        return new MultiLabelDriftState(
                DistributionDiagnostics.copyFractionVector(
                        referencePositiveFractions,
                        "referencePositiveFractions",
                        labelCount),
                DistributionDiagnostics.copyFractionVector(
                        candidatePositiveFractions,
                        "candidatePositiveFractions",
                        labelCount),
                DistributionDiagnostics.copyFiniteVector(
                        positiveFractionDeltas,
                        "positiveFractionDeltas",
                        labelCount),
                immutableList(referenceZeroPositiveLabels, "referenceZeroPositiveLabels"),
                immutableList(candidateZeroPositiveLabels, "candidateZeroPositiveLabels"));
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got: " + value);
        }
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got: " + value);
        }
    }

    private static <T> List<T> immutableList(List<T> values, String name) {
        return List.copyOf(Objects.requireNonNull(values, name + " must not be null"));
    }

    record ClassificationReportState(int[] classCounts, List<int[]> batchClassCounts) {
    }

    record MultiLabelReportState(int[] positiveCounts, int[] negativeCounts, List<int[]> batchPositiveCounts) {
    }

    record ClassificationDriftState(
            double[] referenceFractions,
            double[] candidateFractions,
            double[] fractionDeltas,
            List<Integer> referenceMissingClasses,
            List<Integer> candidateMissingClasses) {
    }

    record MultiLabelDriftState(
            double[] referencePositiveFractions,
            double[] candidatePositiveFractions,
            double[] positiveFractionDeltas,
            List<Integer> referenceZeroPositiveLabels,
            List<Integer> candidateZeroPositiveLabels) {
    }
}
