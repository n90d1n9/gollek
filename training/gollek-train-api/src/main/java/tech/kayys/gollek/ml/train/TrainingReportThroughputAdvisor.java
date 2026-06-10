package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds trainer performance recommendations from throughput counters.
 */
final class TrainingReportThroughputAdvisor {
    private static final double VALIDATION_TRAIN_AVG_BATCH_RATIO = 2.0;
    private static final double MIN_TRAIN_SAMPLES_PER_SECOND = 1.0e-9;

    private TrainingReportThroughputAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        TrainingReportThroughput throughput = TrainingReportReader.throughputView(report);
        if (!throughput.available()) {
            return List.of();
        }
        if (hasZeroTrainThroughput(throughput.train())) {
            return List.of(zeroTrainThroughputRecommendation(throughput));
        }
        if (validationMuchSlowerThanTrain(throughput)) {
            return List.of(validationThroughputRecommendation(throughput));
        }
        return List.of();
    }

    private static boolean hasZeroTrainThroughput(TrainingReportThroughput.Phase train) {
        return train.batchCount().orElse(0L) > 0L
                && train.sampleCount().orElse(0L) > 0L
                && train.samplesPerSecond().orElse(0.0) <= MIN_TRAIN_SAMPLES_PER_SECOND;
    }

    private static boolean validationMuchSlowerThanTrain(TrainingReportThroughput throughput) {
        double trainAverage = throughput.train().averageBatchMillis().orElse(0.0);
        double validationAverage = throughput.validation().averageBatchMillis().orElse(0.0);
        return trainAverage > 0.0
                && validationAverage / trainAverage >= VALIDATION_TRAIN_AVG_BATCH_RATIO
                && throughput.validation().batchCount().orElse(0L) > 0L;
    }

    private static TrainingReportRecommendation zeroTrainThroughputRecommendation(
            TrainingReportThroughput throughput) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.HIGH,
                TrainingReportRecommendation.Category.OPTIMIZATION,
                TrainingReportDiagnostics.Severity.WARNING,
                "throughput.train.zero_samples_per_second",
                "Investigate near-zero train throughput",
                "The throughput counters recorded train batches and samples, but the computed train samples/sec is near zero.",
                List.of(
                        "Inspect train batch compute time, backend placement, and CPU/GPU transfer boundaries before tuning model quality.",
                        "Run a short profiling job with runtime profile enabled to identify whether input, forward, backward, or optimizer work dominates.",
                        "Confirm batch tensors are not accidentally oversized or forcing synchronous host-device copies."),
                evidence(throughput));
    }

    private static TrainingReportRecommendation validationThroughputRecommendation(
            TrainingReportThroughput throughput) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.VALIDATION,
                TrainingReportDiagnostics.Severity.INFO,
                "throughput.validation.slower_than_train",
                "Review validation throughput before increasing validation frequency",
                "Validation average batch time is at least twice the train average batch time.",
                List.of(
                        "Inspect validation-only metrics, decoding, and preprocessing before changing model architecture.",
                        "Cache validation transforms or reduce validation frequency when validation throughput dominates wall-clock time.",
                        "Compare train and validation batch shapes to confirm validation is not using larger or irregular batches."),
                evidence(throughput));
    }

    private static Map<String, Object> evidence(TrainingReportThroughput throughput) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        addPhaseEvidence(evidence, "train", throughput.train());
        addPhaseEvidence(evidence, "validation", throughput.validation());
        double trainAverage = throughput.train().averageBatchMillis().orElse(0.0);
        double validationAverage = throughput.validation().averageBatchMillis().orElse(0.0);
        if (trainAverage > 0.0 && validationAverage > 0.0) {
            evidence.put("validationToTrainAverageBatchMillisRatio", validationAverage / trainAverage);
        }
        return Map.copyOf(evidence);
    }

    private static void addPhaseEvidence(
            Map<String, Object> evidence,
            String phase,
            TrainingReportThroughput.Phase throughput) {
        throughput.batchCount().ifPresent(value -> evidence.put(phase + "BatchCount", value));
        throughput.sampleCount().ifPresent(value -> evidence.put(phase + "SampleCount", value));
        throughput.computeMillis().ifPresent(value -> evidence.put(phase + "ComputeMillis", value));
        throughput.samplesPerSecond().ifPresent(value -> evidence.put(phase + "SamplesPerSecond", value));
        throughput.batchesPerSecond().ifPresent(value -> evidence.put(phase + "BatchesPerSecond", value));
        throughput.averageBatchMillis().ifPresent(value -> evidence.put(phase + "AverageBatchMillis", value));
    }
}
