package tech.kayys.gollek.ml.reasoning;

import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Stable SHA-256 fingerprint for token datasets and materialized dataset plans.
 */
public record DiscreteTokenDatasetFingerprint(
        String algorithm,
        String value,
        int exampleCount) {

    public static final String ALGORITHM = "SHA-256";
    public static final String DATASET_VERSION = "gollek.discrete-token-dataset.v1";
    public static final String PLAN_VERSION = "gollek.discrete-token-dataset-plan.v1";
    public static final String METADATA_KEY = "fingerprint";
    public static final int DEFAULT_SHORT_LENGTH = 12;

    public DiscreteTokenDatasetFingerprint {
        algorithm = DiscreteTokenDatasetMetadataSupport.requireText(algorithm, "algorithm");
        value = DiscreteTokenDatasetMetadataSupport.requireText(value, "value").toLowerCase();
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
    }

    public static DiscreteTokenDatasetFingerprint fromExamples(List<DiscreteTokenDatasetExample> examples) {
        Objects.requireNonNull(examples, "examples must not be null");
        MessageDigest digest = sha256();
        updateString(digest, DATASET_VERSION);
        updateInt(digest, examples.size());
        for (int index = 0; index < examples.size(); index++) {
            updateExample(digest, Objects.requireNonNull(examples.get(index), "examples[" + index + "] must not be null"));
        }
        return new DiscreteTokenDatasetFingerprint(ALGORITHM, hex(digest.digest()), examples.size());
    }

    public static DiscreteTokenDatasetFingerprint fromPlan(DiscreteTokenDatasetPlan plan) {
        Objects.requireNonNull(plan, "plan must not be null");
        MessageDigest digest = sha256();
        updateString(digest, PLAN_VERSION);
        updateConfig(digest, plan.config());
        updateSplit(digest, plan.split());
        updateEpoch(digest, "train", plan.trainEpoch());
        updateEpoch(digest, "validation", plan.validationEpoch());
        updateEpoch(digest, "test", plan.testEpoch());
        return new DiscreteTokenDatasetFingerprint(ALGORITHM, hex(digest.digest()), plan.profile().exampleCount());
    }

    public static DiscreteTokenDatasetFingerprint fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetFingerprint(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "algorithm"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "value"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "exampleCount"));
    }

    public static DiscreteTokenDatasetFingerprint fromMetadataSection(Map<?, ?> metadata) {
        return fromMetadataSection(metadata, METADATA_KEY);
    }

    public static DiscreteTokenDatasetFingerprint fromMetadataSection(Map<?, ?> metadata, String key) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Objects.requireNonNull(key, "key must not be null");
        if (key.isBlank()) {
            throw new IllegalArgumentException("key must not be blank");
        }
        Object value = DiscreteTokenDatasetMetadataSupport.required(metadata, key);
        if (value instanceof Map<?, ?> fingerprintMetadata) {
            return fromMetadata(fingerprintMetadata);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a fingerprint metadata map");
    }

    public String shortValue() {
        return shortValue(DEFAULT_SHORT_LENGTH);
    }

    public String shortValue(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("length must be > 0 but was " + length);
        }
        return value.length() <= length ? value : value.substring(0, length);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("algorithm", algorithm);
        metadata.put("value", value);
        metadata.put("shortValue", shortValue());
        metadata.put("exampleCount", exampleCount);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void updateConfig(MessageDigest digest, DiscreteTokenDatasetPlanConfig config) {
        updateString(digest, "config");
        updateDouble(digest, config.validationFraction());
        updateDouble(digest, config.testFraction());
        updateString(digest, config.splitMode().name());
        updateLong(digest, config.splitSeed());
        updateInt(digest, config.trainBatchSize());
        updateInt(digest, config.evaluationBatchSize());
        updateInt(digest, config.inputPadToken());
        updateInt(digest, config.targetPadToken());
        updateString(digest, config.trainEpochMode().name());
        updateLong(digest, config.trainEpochSeed());
        updateBoolean(digest, config.dropLastTrain());
    }

    private static void updateSplit(MessageDigest digest, DiscreteTokenDatasetSplit split) {
        updateString(digest, "split");
        updateBoolean(digest, split.shuffled());
        updateLong(digest, split.seed());
        updateExamples(digest, "train", split.trainExamples());
        updateExamples(digest, "validation", split.validationExamples());
        updateExamples(digest, "test", split.testExamples());
    }

    private static void updateExamples(
            MessageDigest digest,
            String section,
            List<DiscreteTokenDatasetExample> examples) {
        updateString(digest, section);
        updateInt(digest, examples.size());
        for (DiscreteTokenDatasetExample example : examples) {
            updateExample(digest, example);
        }
    }

    private static void updateExample(MessageDigest digest, DiscreteTokenDatasetExample example) {
        updateString(digest, "example");
        updateString(digest, example.taskId());
        updateInt(digest, example.exampleIndex());
        updateIntArray(digest, example.inputTokens());
        updateIntArray(digest, example.targetTokens());
        updateInt(digest, example.knownSolutionCount());
        updateValue(digest, example.metadata());
    }

    private static void updateEpoch(MessageDigest digest, String section, DiscreteTokenDatasetEpoch epoch) {
        updateString(digest, "epoch");
        updateString(digest, section);
        updateInt(digest, epoch.exampleCount());
        updateInt(digest, epoch.emittedExampleCount());
        updateInt(digest, epoch.droppedExampleCount());
        updateInt(digest, epoch.requestedBatchSize());
        updateBoolean(digest, epoch.shuffled());
        updateBoolean(digest, epoch.dropLast());
        updateLong(digest, epoch.seed());
        updateInt(digest, epoch.inputPadToken());
        updateInt(digest, epoch.targetPadToken());
        updateInt(digest, epoch.batches().size());
        for (DiscreteTokenDatasetBatch batch : epoch.batches()) {
            updateBatch(digest, batch);
        }
    }

    private static void updateBatch(MessageDigest digest, DiscreteTokenDatasetBatch batch) {
        updateString(digest, "batch");
        updateStringArray(digest, batch.taskIds());
        updateIntArray(digest, batch.exampleIndices());
        updateIntArray(digest, batch.inputLengths());
        updateIntArray(digest, batch.targetLengths());
        updateIntArray(digest, batch.knownSolutionCounts());
        updateInt(digest, batch.inputPadToken());
        updateInt(digest, batch.targetPadToken());
    }

    private static void updateValue(MessageDigest digest, Object value) {
        if (value == null) {
            updateString(digest, "null");
            return;
        }
        if (value instanceof Map<?, ?> map) {
            updateString(digest, "map");
            updateInt(digest, map.size());
            map.entrySet().stream()
                    .sorted(Comparator.comparing(entry -> String.valueOf(entry.getKey())))
                    .forEach(entry -> {
                        updateString(digest, String.valueOf(entry.getKey()));
                        updateValue(digest, entry.getValue());
                    });
            return;
        }
        if (value instanceof Iterable<?> iterable) {
            updateString(digest, "iterable");
            int count = 0;
            for (Object ignored : iterable) {
                count++;
            }
            updateInt(digest, count);
            for (Object item : iterable) {
                updateValue(digest, item);
            }
            return;
        }
        if (value.getClass().isArray()) {
            updateString(digest, "array");
            int length = Array.getLength(value);
            updateInt(digest, length);
            for (int index = 0; index < length; index++) {
                updateValue(digest, Array.get(value, index));
            }
            return;
        }
        if (value instanceof Number || value instanceof Boolean || value instanceof CharSequence || value instanceof Enum<?>) {
            updateString(digest, value.getClass().getName());
            updateString(digest, String.valueOf(value));
            return;
        }
        updateString(digest, value.getClass().getName());
        updateString(digest, String.valueOf(value));
    }

    private static void updateStringArray(MessageDigest digest, String[] values) {
        updateInt(digest, values.length);
        for (String value : values) {
            updateString(digest, value);
        }
    }

    private static void updateIntArray(MessageDigest digest, int[] values) {
        updateInt(digest, values.length);
        for (int value : values) {
            updateInt(digest, value);
        }
    }

    private static void updateString(MessageDigest digest, String value) {
        byte[] bytes = Objects.requireNonNull(value, "value must not be null").getBytes(StandardCharsets.UTF_8);
        updateInt(digest, bytes.length);
        digest.update(bytes);
    }

    private static void updateBoolean(MessageDigest digest, boolean value) {
        digest.update((byte) (value ? 1 : 0));
    }

    private static void updateInt(MessageDigest digest, int value) {
        digest.update((byte) (value >>> 24));
        digest.update((byte) (value >>> 16));
        digest.update((byte) (value >>> 8));
        digest.update((byte) value);
    }

    private static void updateLong(MessageDigest digest, long value) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            digest.update((byte) (value >>> shift));
        }
    }

    private static void updateDouble(MessageDigest digest, double value) {
        updateLong(digest, Double.doubleToLongBits(value));
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance(ALGORITHM);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(ALGORITHM + " is not available", e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(Character.forDigit((value >>> 4) & 0x0f, 16));
            builder.append(Character.forDigit(value & 0x0f, 16));
        }
        return builder.toString();
    }
}
