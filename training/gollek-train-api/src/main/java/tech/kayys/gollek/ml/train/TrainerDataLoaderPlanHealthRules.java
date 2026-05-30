package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainerDataLoaderPlanHealthRules {
    private static final String KIND = "data-loader";

    private TrainerDataLoaderPlanHealthRules() {
    }

    static List<TrainerHealthIssue> evaluate(Map<String, Object> metadata) {
        List<TrainerHealthIssue> issues = new ArrayList<>();
        addLoaderIssues(metadata, issues, "trainLoaderPlan", "train", true);
        addLoaderIssues(metadata, issues, "validationLoaderPlan", "validation", false);
        return List.copyOf(issues);
    }

    private static void addLoaderIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues,
            String prefix,
            String phase,
            boolean required) {
        if (!booleanValue(metadata.get(prefix + ".available"))) {
            if (required || "empty-batch-collection".equals(metadata.get(prefix + ".skipReason"))) {
                issues.add(issue(
                        code(phase, "plan-unavailable"),
                        required ? TrainerHealthIssue.ERROR : TrainerHealthIssue.WARNING,
                        required,
                        phase,
                        phase + " loader plan is unavailable: "
                                + stringValue(metadata.get(prefix + ".skipReason"), "unknown"),
                        required
                                ? "use a DataLoader.TensorDataLoader or materialized Batch collection for training"
                                : "provide a validation DataLoader or materialized validation Batch collection",
                        evidence(prefix, metadata)));
            }
            return;
        }

        int sampleCount = intValue(metadata.get(prefix + ".sampleCount"), -1);
        int batchSize = intValue(metadata.get(prefix + ".batchSize"), -1);
        int batchCount = intValue(metadata.get(prefix + ".batchCount"), -1);
        boolean dropLast = booleanValue(metadata.get(prefix + ".dropLast"));
        boolean sampled = booleanValue(metadata.get(prefix + ".sampled"));
        boolean shuffle = booleanValue(metadata.get(prefix + ".shuffle"));
        boolean reshuffleEachEpoch = booleanValue(metadata.get(prefix + ".reshuffleEachEpoch"));
        boolean hasShuffleSeed = booleanValue(metadata.get(prefix + ".hasShuffleSeed"));

        if (sampleCount <= 0 || batchCount <= 0) {
            issues.add(issue(
                    code(phase, "empty"),
                    required ? TrainerHealthIssue.ERROR : TrainerHealthIssue.WARNING,
                    required,
                    phase,
                    phase + " loader has no usable batches or samples",
                    required
                            ? "provide at least one non-empty training batch before fitting"
                            : "provide validation batches or disable validation-dependent checks",
                    evidence(prefix, metadata)));
            return;
        }

        int droppedSamples = droppedSamples(sampleCount, batchSize, batchCount, dropLast);
        metadata.put(prefix + ".droppedSampleCount", droppedSamples);
        if (droppedSamples > 0) {
            issues.add(warning(
                    code(phase, "drop-last-discarded-samples"),
                    phase,
                    phase + " loader drops " + droppedSamples + " sample(s) because dropLast=true",
                    "disable dropLast or make dataset size divisible by batch size if every sample should train",
                    evidence(prefix, metadata)));
        }

        double coverage = optionalDouble(metadata.get(prefix + ".sampleCoverageRatio")).orElse(1.0);
        if (sampled && coverage < 0.95) {
            issues.add(warning(
                    code(phase, "low-sampler-coverage"),
                    phase,
                    phase + " loader samples only " + percent(coverage) + " of the dataset per epoch",
                    "increase sampler sample count or confirm that subset/weighted sampling is intentional",
                    evidence(prefix, metadata)));
        }

        if (shuffle && reshuffleEachEpoch && !hasShuffleSeed) {
            issues.add(warning(
                    code(phase, "reshuffle-without-seed"),
                    phase,
                    phase + " loader reshuffles every epoch without a recorded seed",
                    "set an explicit loader seed when reproducible training runs are required",
                    evidence(prefix, metadata)));
        }
    }

    private static TrainerHealthIssue issue(
            String code,
            String severity,
            boolean blocking,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return TrainerHealthIssue.issue(KIND, code, severity, blocking, artifact, message, action, evidence);
    }

    private static TrainerHealthIssue warning(
            String code,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return TrainerHealthIssue.warning(KIND, code, artifact, message, action, evidence);
    }

    private static Map<String, Object> evidence(String prefix, Map<String, Object> metadata) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        copy(metadata, evidence, prefix + ".kind");
        copy(metadata, evidence, prefix + ".available");
        copy(metadata, evidence, prefix + ".skipReason");
        copy(metadata, evidence, prefix + ".datasetSize");
        copy(metadata, evidence, prefix + ".sampleCount");
        copy(metadata, evidence, prefix + ".batchSize");
        copy(metadata, evidence, prefix + ".batchCount");
        copy(metadata, evidence, prefix + ".sampled");
        copy(metadata, evidence, prefix + ".dropLast");
        copy(metadata, evidence, prefix + ".sampleCoverageRatio");
        copy(metadata, evidence, prefix + ".droppedSampleCount");
        copy(metadata, evidence, prefix + ".shuffle");
        copy(metadata, evidence, prefix + ".hasShuffleSeed");
        copy(metadata, evidence, prefix + ".reshuffleEachEpoch");
        return Map.copyOf(evidence);
    }

    private static void copy(Map<String, Object> source, Map<String, Object> target, String key) {
        if (source.containsKey(key)) {
            target.put(key, source.get(key));
        }
    }

    private static int droppedSamples(int sampleCount, int batchSize, int batchCount, boolean dropLast) {
        if (!dropLast || sampleCount <= 0 || batchSize <= 0 || batchCount <= 0) {
            return 0;
        }
        return Math.max(0, sampleCount - (batchCount * batchSize));
    }

    private static String code(String phase, String suffix) {
        return "data-loader-" + phase + "-" + suffix;
    }

    private static String percent(double value) {
        return Math.round(value * 1000.0) / 10.0 + "%";
    }
}
