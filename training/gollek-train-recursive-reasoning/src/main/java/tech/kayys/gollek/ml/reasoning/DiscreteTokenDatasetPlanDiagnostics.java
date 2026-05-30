package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;

/**
 * Compact health report for a materialized token dataset plan.
 */
public record DiscreteTokenDatasetPlanDiagnostics(
        int exampleCount,
        int taskCount,
        int trainCount,
        int validationCount,
        int testCount,
        int trainBatchCount,
        int validationBatchCount,
        int testBatchCount,
        long emittedTrainingExamples,
        long droppedTrainingExamples,
        double knownSolutionCoverageRate,
        double trainPaddingRate,
        long trainPaddingTokenCount,
        long trainPaddedTokenCapacity,
        List<DiscreteTokenDatasetTaskSplitDiagnostics> taskSplits,
        List<String> warnings) {

    public static final double DEFAULT_HIGH_PADDING_RATE_THRESHOLD = 0.50d;

    public DiscreteTokenDatasetPlanDiagnostics {
        exampleCount = requireNonNegative(exampleCount, "exampleCount");
        taskCount = requireNonNegative(taskCount, "taskCount");
        trainCount = requireNonNegative(trainCount, "trainCount");
        validationCount = requireNonNegative(validationCount, "validationCount");
        testCount = requireNonNegative(testCount, "testCount");
        trainBatchCount = requireNonNegative(trainBatchCount, "trainBatchCount");
        validationBatchCount = requireNonNegative(validationBatchCount, "validationBatchCount");
        testBatchCount = requireNonNegative(testBatchCount, "testBatchCount");
        if (trainCount + validationCount + testCount != exampleCount) {
            throw new IllegalArgumentException("split counts must sum to exampleCount");
        }
        if (taskCount > exampleCount) {
            throw new IllegalArgumentException("taskCount cannot exceed exampleCount");
        }
        if (emittedTrainingExamples < 0L) {
            throw new IllegalArgumentException("emittedTrainingExamples must be >= 0");
        }
        if (droppedTrainingExamples < 0L) {
            throw new IllegalArgumentException("droppedTrainingExamples must be >= 0");
        }
        if (emittedTrainingExamples + droppedTrainingExamples != trainCount) {
            throw new IllegalArgumentException(
                    "emittedTrainingExamples + droppedTrainingExamples must equal trainCount");
        }
        if (!isRate(knownSolutionCoverageRate)) {
            throw new IllegalArgumentException(
                    "knownSolutionCoverageRate must be finite and in [0, 1] but was "
                            + knownSolutionCoverageRate);
        }
        if (!isRate(trainPaddingRate)) {
            throw new IllegalArgumentException(
                    "trainPaddingRate must be finite and in [0, 1] but was " + trainPaddingRate);
        }
        if (trainPaddingTokenCount < 0L) {
            throw new IllegalArgumentException("trainPaddingTokenCount must be >= 0");
        }
        if (trainPaddedTokenCapacity < 0L) {
            throw new IllegalArgumentException("trainPaddedTokenCapacity must be >= 0");
        }
        if (trainPaddingTokenCount > trainPaddedTokenCapacity) {
            throw new IllegalArgumentException("trainPaddingTokenCount cannot exceed trainPaddedTokenCapacity");
        }
        taskSplits = List.copyOf(Objects.requireNonNull(taskSplits, "taskSplits must not be null"));
        if (taskSplits.size() != taskCount) {
            throw new IllegalArgumentException("taskSplits size must equal taskCount");
        }
        int observedExamples = 0;
        int observedTrain = 0;
        int observedValidation = 0;
        int observedTest = 0;
        for (int index = 0; index < taskSplits.size(); index++) {
            DiscreteTokenDatasetTaskSplitDiagnostics split =
                    Objects.requireNonNull(taskSplits.get(index), "taskSplits[" + index + "] must not be null");
            observedExamples += split.exampleCount();
            observedTrain += split.trainCount();
            observedValidation += split.validationCount();
            observedTest += split.testCount();
        }
        if (observedExamples != exampleCount) {
            throw new IllegalArgumentException("sum of task split example counts must equal exampleCount");
        }
        if (observedTrain != trainCount) {
            throw new IllegalArgumentException("sum of task split train counts must equal trainCount");
        }
        if (observedValidation != validationCount) {
            throw new IllegalArgumentException("sum of task split validation counts must equal validationCount");
        }
        if (observedTest != testCount) {
            throw new IllegalArgumentException("sum of task split test counts must equal testCount");
        }
        warnings = List.copyOf(Objects.requireNonNull(warnings, "warnings must not be null"));
        for (int index = 0; index < warnings.size(); index++) {
            String warning = Objects.requireNonNull(warnings.get(index), "warnings[" + index + "] must not be null");
            if (warning.isBlank()) {
                throw new IllegalArgumentException("warnings[" + index + "] must not be blank");
            }
        }
    }

    public static DiscreteTokenDatasetPlanDiagnostics from(DiscreteTokenDatasetPlan plan) {
        return from(plan, DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults());
    }

    public static DiscreteTokenDatasetPlanDiagnostics from(
            DiscreteTokenDatasetPlan plan,
            double highPaddingRateThreshold) {
        return from(
                plan,
                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                        .withHighPaddingRateThreshold(highPaddingRateThreshold));
    }

    public static DiscreteTokenDatasetPlanDiagnostics from(
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetPlanDiagnosticsPolicy policy) {
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(policy, "policy must not be null");

        DiscreteTokenDatasetProfile profile = plan.profile();
        DiscreteTokenDatasetSplit split = plan.split();
        DiscreteTokenDatasetEpoch trainEpoch = plan.trainEpoch();
        List<DiscreteTokenDatasetTaskSplitDiagnostics> taskSplits = taskSplits(plan);
        List<String> warnings = warnings(plan, taskSplits, policy);

        return new DiscreteTokenDatasetPlanDiagnostics(
                profile.exampleCount(),
                profile.taskCount(),
                split.trainCount(),
                split.validationCount(),
                split.testCount(),
                trainEpoch.batchCount(),
                plan.validationEpoch().batchCount(),
                plan.testEpoch().batchCount(),
                trainEpoch.emittedExampleCount(),
                trainEpoch.droppedExampleCount(),
                profile.knownSolutionCoverageRate(),
                trainEpoch.paddingRate(),
                trainEpoch.paddingTokenCount(),
                trainEpoch.paddedTokenCapacity(),
                taskSplits,
                warnings);
    }

    public static DiscreteTokenDatasetPlanDiagnostics fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Map<?, ?> split = requiredMap(metadata, "split");
        Map<?, ?> batches = requiredMap(metadata, "batches");
        Map<?, ?> trainingExamples = requiredMap(metadata, "trainingExamples");
        Map<?, ?> trainPadding = requiredMap(metadata, "trainPadding");
        DiscreteTokenDatasetPlanDiagnostics diagnostics = new DiscreteTokenDatasetPlanDiagnostics(
                requiredInt(metadata, "exampleCount"),
                requiredInt(metadata, "taskCount"),
                requiredInt(split, "trainCount"),
                requiredInt(split, "validationCount"),
                requiredInt(split, "testCount"),
                requiredInt(batches, "trainBatchCount"),
                requiredInt(batches, "validationBatchCount"),
                requiredInt(batches, "testBatchCount"),
                requiredLong(trainingExamples, "emittedTrainingExamples"),
                requiredLong(trainingExamples, "droppedTrainingExamples"),
                requiredDouble(metadata, "knownSolutionCoverageRate"),
                requiredDouble(trainPadding, "rate"),
                requiredLong(trainPadding, "tokenCount"),
                requiredLong(trainPadding, "paddedTokenCapacity"),
                requiredTaskSplits(metadata, "taskSplits"),
                requiredStringList(metadata, "warnings"));
        verifyOptionalMetadata(metadata, diagnostics);
        return diagnostics;
    }

    public boolean hasWarnings() {
        return !warnings.isEmpty();
    }

    public boolean hasValidationSet() {
        return validationCount > 0;
    }

    public boolean hasTestSet() {
        return testCount > 0;
    }

    public boolean isReadyForTraining() {
        return exampleCount > 0 && trainBatchCount > 0 && emittedTrainingExamples > 0;
    }

    public boolean hasKnownSolutionCoverage() {
        return knownSolutionCoverageRate > 0.0d;
    }

    public String status() {
        if (!isReadyForTraining()) {
            return "blocked";
        }
        return hasWarnings() ? "warning" : "ready";
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status());
        metadata.put("readyForTraining", isReadyForTraining());
        metadata.put("hasWarnings", hasWarnings());
        metadata.put("exampleCount", exampleCount);
        metadata.put("taskCount", taskCount);
        metadata.put("split", section(
                "trainCount", trainCount,
                "validationCount", validationCount,
                "testCount", testCount));
        metadata.put("batches", section(
                "trainBatchCount", trainBatchCount,
                "validationBatchCount", validationBatchCount,
                "testBatchCount", testBatchCount));
        metadata.put("trainingExamples", section(
                "emittedTrainingExamples", emittedTrainingExamples,
                "droppedTrainingExamples", droppedTrainingExamples));
        metadata.put("knownSolutionCoverageRate", knownSolutionCoverageRate);
        metadata.put("knownSolutionCoverageAvailable", hasKnownSolutionCoverage());
        metadata.put("trainPadding", section(
                "rate", trainPaddingRate,
                "tokenCount", trainPaddingTokenCount,
                "paddedTokenCapacity", trainPaddedTokenCapacity));
        metadata.put("taskSplits", taskSplits.stream()
                .map(DiscreteTokenDatasetTaskSplitDiagnostics::toMetadata)
                .toList());
        metadata.put("warnings", warnings);
        return immutableMap(metadata);
    }

    private static List<String> warnings(
            DiscreteTokenDatasetPlan plan,
            List<DiscreteTokenDatasetTaskSplitDiagnostics> taskSplits,
            DiscreteTokenDatasetPlanDiagnosticsPolicy policy) {
        List<String> warnings = new ArrayList<>();
        DiscreteTokenDatasetProfile profile = plan.profile();
        DiscreteTokenDatasetSplit split = plan.split();
        DiscreteTokenDatasetEpoch trainEpoch = plan.trainEpoch();

        if (profile.exampleCount() == 0) {
            warnings.add("dataset is empty");
        }
        if (split.trainCount() == 0) {
            warnings.add("train split is empty");
        }
        if (policy.warnOnMissingValidationSplit() && !plan.hasValidationEpoch()) {
            warnings.add("validation split is empty");
        }
        if (policy.warnOnMissingTestSplit() && !plan.hasTestEpoch()) {
            warnings.add("test split is empty");
        }
        if (policy.warnOnDroppedTrainingExamples() && trainEpoch.droppedExampleCount() > 0) {
            warnings.add("train epoch dropped " + trainEpoch.droppedExampleCount() + " example(s)");
        }
        if (policy.warnOnMissingKnownSolutionCounts()
                && profile.knownSolutionExampleCount() == 0
                && profile.exampleCount() > 0) {
            warnings.add("dataset has no known solution counts");
        } else if (policy.warnOnPartialKnownSolutionCoverage()
                && profile.knownSolutionExampleCount() > 0
                && profile.unknownSolutionExampleCount() > 0) {
            warnings.add("dataset has partial known solution coverage");
        }
        if (policy.warnOnHighPaddingRate()
                && trainEpoch.paddingRate() > policy.highPaddingRateThreshold()) {
            warnings.add("train padding rate is high: " + trainEpoch.paddingRate());
        }
        for (DiscreteTokenDatasetTaskSplitDiagnostics taskSplit : taskSplits) {
            if (policy.warnOnMissingTaskTrainCoverage() && !taskSplit.hasTrainExamples()) {
                warnings.add("train split is missing task: " + taskSplit.taskId());
            }
            if (policy.warnOnMissingTaskValidationCoverage()
                    && split.validationCount() > 0
                    && !taskSplit.hasValidationExamples()) {
                warnings.add("validation split is missing task: " + taskSplit.taskId());
            }
            if (policy.warnOnMissingTaskTestCoverage()
                    && split.testCount() > 0
                    && !taskSplit.hasTestExamples()) {
                warnings.add("test split is missing task: " + taskSplit.taskId());
            }
        }
        return warnings;
    }

    private static List<DiscreteTokenDatasetTaskSplitDiagnostics> taskSplits(DiscreteTokenDatasetPlan plan) {
        DiscreteTokenDatasetProfile profile = plan.profile();
        DiscreteTokenDatasetSplit split = plan.split();
        Map<String, Integer> trainCounts = split.trainProfile().taskExampleCounts();
        Map<String, Integer> validationCounts = split.validationProfile().taskExampleCounts();
        Map<String, Integer> testCounts = split.testProfile().taskExampleCounts();

        List<DiscreteTokenDatasetTaskSplitDiagnostics> taskSplits = new ArrayList<>();
        for (String taskId : new TreeSet<>(profile.taskExampleCounts().keySet())) {
            taskSplits.add(new DiscreteTokenDatasetTaskSplitDiagnostics(
                    taskId,
                    profile.taskExampleCounts().getOrDefault(taskId, 0),
                    trainCounts.getOrDefault(taskId, 0),
                    validationCounts.getOrDefault(taskId, 0),
                    testCounts.getOrDefault(taskId, 0)));
        }
        return taskSplits;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }

    private static boolean isRate(double value) {
        return Double.isFinite(value) && value >= 0.0d && value <= 1.0d;
    }

    private static Map<String, Object> immutableMap(Map<String, Object> values) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    private static Map<String, Object> section(Object... keyValues) {
        if (keyValues.length % 2 != 0) {
            throw new IllegalArgumentException("metadata sections require key/value pairs");
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            values.put((String) keyValues[i], keyValues[i + 1]);
        }
        return immutableMap(values);
    }

    private static List<DiscreteTokenDatasetTaskSplitDiagnostics> requiredTaskSplits(
            Map<?, ?> metadata,
            String key) {
        Object value = required(metadata, key);
        if (value instanceof List<?> list) {
            return list.stream()
                    .map(item -> {
                        if (item instanceof Map<?, ?> map) {
                            return DiscreteTokenDatasetTaskSplitDiagnostics.fromMetadata(map);
                        }
                        throw new IllegalArgumentException("metadata field '" + key + "' entries must be maps");
                    })
                    .toList();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    private static List<String> requiredStringList(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof List<?> list) {
            return list.stream()
                    .map(item -> {
                        if (item instanceof CharSequence text) {
                            return text.toString();
                        }
                        throw new IllegalArgumentException("metadata field '" + key + "' entries must be strings");
                    })
                    .toList();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    private static Map<?, ?> requiredMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static int requiredInt(Map<?, ?> metadata, String key) {
        long value = requiredLong(metadata, key);
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("metadata field '" + key + "' must fit in an int");
        }
        return (int) value;
    }

    private static long requiredLong(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            double numericValue = number.doubleValue();
            if (!Double.isFinite(numericValue)
                    || Math.rint(numericValue) != numericValue
                    || numericValue < Long.MIN_VALUE
                    || numericValue > Long.MAX_VALUE) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
            }
            return number.longValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Long.parseLong(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
    }

    private static double requiredDouble(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Double.parseDouble(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be a number", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a number");
    }

    private static boolean optionalBoolean(Map<?, ?> metadata, String key, boolean expected) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return expected;
        }
        Object value = metadata.get(key);
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof CharSequence text) {
            String normalized = text.toString().trim().toLowerCase();
            if ("true".equals(normalized)) {
                return true;
            }
            if ("false".equals(normalized)) {
                return false;
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a boolean");
    }

    private static String optionalString(Map<?, ?> metadata, String key, String expected) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return expected;
        }
        Object value = metadata.get(key);
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }

    private static void verifyOptionalMetadata(
            Map<?, ?> metadata,
            DiscreteTokenDatasetPlanDiagnostics diagnostics) {
        if (!optionalString(metadata, "status", diagnostics.status()).equals(diagnostics.status())) {
            throw new IllegalArgumentException("metadata field 'status' does not match diagnostics");
        }
        if (optionalBoolean(metadata, "readyForTraining", diagnostics.isReadyForTraining())
                != diagnostics.isReadyForTraining()) {
            throw new IllegalArgumentException("metadata field 'readyForTraining' does not match diagnostics");
        }
        if (optionalBoolean(metadata, "hasWarnings", diagnostics.hasWarnings()) != diagnostics.hasWarnings()) {
            throw new IllegalArgumentException("metadata field 'hasWarnings' does not match diagnostics");
        }
    }
}
