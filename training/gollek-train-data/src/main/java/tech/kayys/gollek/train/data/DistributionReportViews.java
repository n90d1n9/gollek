package tech.kayys.gollek.train.data;

import java.util.List;
import java.util.Map;

interface ClassificationDistributionView {

    int sampleCount();

    int batchCount();

    int numClasses();

    int[] classCounts();

    List<int[]> batchClassCounts();

    default int classCount(int classIndex) {
        return classCounts()[classIndex];
    }

    default double[] classFractions() {
        return DistributionDiagnostics.countFractions(classCounts(), sampleCount());
    }

    default int majorityClassIndex() {
        return DistributionDiagnostics.maxCountIndex(classCounts());
    }

    default int minorityClassIndex() {
        return DistributionDiagnostics.minCountIndex(classCounts());
    }

    default int majorityClassCount() {
        int[] counts = classCounts();
        return counts[DistributionDiagnostics.maxCountIndex(counts)];
    }

    default int minorityClassCount() {
        int[] counts = classCounts();
        return counts[DistributionDiagnostics.minCountIndex(counts)];
    }

    default int missingClassCount() {
        return DistributionDiagnostics.zeroCount(classCounts());
    }

    default double imbalanceRatio() {
        if (sampleCount() == 0) {
            return 0.0;
        }
        int[] counts = classCounts();
        int majority = counts[DistributionDiagnostics.maxCountIndex(counts)];
        int minority = counts[DistributionDiagnostics.minCountIndex(counts)];
        return majority / (double) Math.max(1, minority);
    }

    default double normalizedEntropy() {
        return DistributionDiagnostics.normalizedCountEntropy(classCounts(), sampleCount());
    }

    default float[] balancedClassWeights() {
        return DistributionDiagnostics.balancedClassWeights(classCounts(), sampleCount());
    }

    default float balancedClassWeight(int classIndex) {
        return balancedClassWeights()[classIndex];
    }

    default DataLoader.ClassificationDistributionDriftReport driftTo(
            DataLoader.ClassificationDistributionReport candidate) {
        return DistributionDiagnostics.classificationDistributionDrift(this, candidate);
    }

    default Map<String, Object> toMetadata(String prefix) {
        return DistributionDiagnostics.classificationMetadata(prefix, this);
    }
}

interface MultiLabelDistributionView {

    int sampleCount();

    int batchCount();

    int labelCount();

    int[] positiveCounts();

    int[] negativeCounts();

    List<int[]> batchPositiveCounts();

    default int positiveCount(int labelIndex) {
        return positiveCounts()[labelIndex];
    }

    default int negativeCount(int labelIndex) {
        return negativeCounts()[labelIndex];
    }

    default double[] positiveFractions() {
        return DistributionDiagnostics.countFractions(positiveCounts(), sampleCount());
    }

    default int maxPositiveLabelIndex() {
        return DistributionDiagnostics.maxCountIndex(positiveCounts());
    }

    default int minPositiveLabelIndex() {
        return DistributionDiagnostics.minCountIndex(positiveCounts());
    }

    default int maxPositiveCount() {
        int[] counts = positiveCounts();
        return counts[DistributionDiagnostics.maxCountIndex(counts)];
    }

    default int minPositiveCount() {
        int[] counts = positiveCounts();
        return counts[DistributionDiagnostics.minCountIndex(counts)];
    }

    default int zeroPositiveLabelCount() {
        return DistributionDiagnostics.zeroCount(positiveCounts());
    }

    default int allPositiveLabelCount() {
        int count = 0;
        int samples = sampleCount();
        for (int positives : positiveCounts()) {
            if (positives == samples) {
                count++;
            }
        }
        return count;
    }

    default double positiveImbalanceRatio() {
        if (sampleCount() == 0) {
            return 0.0;
        }
        int[] counts = positiveCounts();
        int max = counts[DistributionDiagnostics.maxCountIndex(counts)];
        int min = counts[DistributionDiagnostics.minCountIndex(counts)];
        return max / (double) Math.max(1, min);
    }

    default double minPositiveFraction() {
        int samples = sampleCount();
        if (samples == 0) {
            return 0.0;
        }
        int[] counts = positiveCounts();
        return counts[DistributionDiagnostics.minCountIndex(counts)] / (double) samples;
    }

    default double maxPositiveFraction() {
        int samples = sampleCount();
        if (samples == 0) {
            return 0.0;
        }
        int[] counts = positiveCounts();
        return counts[DistributionDiagnostics.maxCountIndex(counts)] / (double) samples;
    }

    default double labelCardinality() {
        int samples = sampleCount();
        return samples == 0 ? 0.0 : DistributionDiagnostics.sumCounts(positiveCounts()) / (double) samples;
    }

    default double labelDensity() {
        return labelCardinality() / labelCount();
    }

    default float[] positiveWeights() {
        int[] counts = positiveCounts();
        float[] weights = new float[labelCount()];
        for (int label = 0; label < weights.length; label++) {
            weights[label] = DistributionDiagnostics.positiveWeight(counts[label], sampleCount());
        }
        return weights;
    }

    default float positiveWeight(int labelIndex) {
        return positiveWeights()[labelIndex];
    }

    default DataLoader.MultiLabelDistributionDriftReport driftTo(DataLoader.MultiLabelDistributionReport candidate) {
        return DistributionDiagnostics.multiLabelDistributionDrift(this, candidate);
    }

    default Map<String, Object> toMetadata(String prefix) {
        return DistributionDiagnostics.multiLabelMetadata(prefix, this);
    }
}

interface ClassificationDistributionDriftView {

    int numClasses();

    double[] referenceFractions();

    double[] candidateFractions();

    double[] fractionDeltas();

    List<Integer> referenceMissingClasses();

    List<Integer> candidateMissingClasses();

    default double totalVariationDistance() {
        return 0.5 * DistributionDiagnostics.absoluteSum(fractionDeltas());
    }

    default int maxDeltaClassIndex() {
        return DistributionDiagnostics.maxAbsIndex(fractionDeltas());
    }

    default double maxAbsoluteFractionDelta() {
        double[] deltas = fractionDeltas();
        return Math.abs(deltas[DistributionDiagnostics.maxAbsIndex(deltas)]);
    }

    default Map<String, Object> toMetadata(String prefix) {
        return DistributionDiagnostics.classificationDriftMetadata(prefix, this);
    }
}

interface MultiLabelDistributionDriftView {

    int labelCount();

    double[] referencePositiveFractions();

    double[] candidatePositiveFractions();

    double[] positiveFractionDeltas();

    double referenceLabelCardinality();

    double candidateLabelCardinality();

    List<Integer> referenceZeroPositiveLabels();

    List<Integer> candidateZeroPositiveLabels();

    default double labelCardinalityDelta() {
        return candidateLabelCardinality() - referenceLabelCardinality();
    }

    default double meanAbsolutePositiveFractionDelta() {
        return DistributionDiagnostics.absoluteSum(positiveFractionDeltas()) / labelCount();
    }

    default int maxDeltaLabelIndex() {
        return DistributionDiagnostics.maxAbsIndex(positiveFractionDeltas());
    }

    default double maxAbsolutePositiveFractionDelta() {
        double[] deltas = positiveFractionDeltas();
        return Math.abs(deltas[DistributionDiagnostics.maxAbsIndex(deltas)]);
    }

    default Map<String, Object> toMetadata(String prefix) {
        return DistributionDiagnostics.multiLabelDriftMetadata(prefix, this);
    }
}
