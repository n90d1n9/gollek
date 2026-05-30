package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Optional expected checkpoint identity fields for stricter resume preflights.
 */
public record DiscreteTokenDatasetCheckpointResumeExpectation(
        String experimentName,
        String runId,
        String modelFamily,
        Long seed,
        Long checkpointStep,
        Long minimumCheckpointStep) {

    private static final DiscreteTokenDatasetCheckpointResumeExpectation NONE =
            new DiscreteTokenDatasetCheckpointResumeExpectation(null, null, null, null, null, null);

    public DiscreteTokenDatasetCheckpointResumeExpectation {
        experimentName = optionalText(experimentName, "experimentName");
        runId = optionalText(runId, "runId");
        modelFamily = optionalText(modelFamily, "modelFamily");
        seed = optionalNonNegative(seed, "seed");
        checkpointStep = optionalNonNegative(checkpointStep, "checkpointStep");
        minimumCheckpointStep = optionalNonNegative(minimumCheckpointStep, "minimumCheckpointStep");
        if (checkpointStep != null && minimumCheckpointStep != null && checkpointStep < minimumCheckpointStep) {
            throw new IllegalArgumentException(
                    "checkpointStep must be >= minimumCheckpointStep when both expectations are set");
        }
    }

    public static DiscreteTokenDatasetCheckpointResumeExpectation none() {
        return NONE;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DiscreteTokenDatasetCheckpointResumeExpectation fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeExpectation(
                optionalString(metadata, "experimentName"),
                optionalString(metadata, "runId"),
                optionalString(metadata, "modelFamily"),
                optionalLong(metadata, "seed"),
                optionalLong(metadata, "checkpointStep"),
                optionalLong(metadata, "minimumCheckpointStep"));
    }

    public static DiscreteTokenDatasetCheckpointResumeExpectation exactFromManifest(
            DiscreteTokenDatasetCheckpointManifest manifest) {
        Objects.requireNonNull(manifest, "manifest must not be null");
        return builder()
                .experimentName(manifest.experimentName())
                .runId(manifest.runId())
                .modelFamily(manifest.modelFamily())
                .seed(manifest.seed())
                .checkpointStep(manifest.checkpointStep())
                .build();
    }

    public static DiscreteTokenDatasetCheckpointResumeExpectation exactFromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        return builder()
                .experimentName(snapshot.experimentName())
                .runId(snapshot.runId())
                .modelFamily(snapshot.modelFamily())
                .seed(snapshot.seed())
                .checkpointStep(snapshot.checkpointStep())
                .build();
    }

    public boolean active() {
        return experimentName != null
                || runId != null
                || modelFamily != null
                || seed != null
                || checkpointStep != null
                || minimumCheckpointStep != null;
    }

    public boolean accepts(DiscreteTokenDatasetCheckpointManifestSnapshot snapshot) {
        return rejectionReasons(snapshot).isEmpty();
    }

    public List<String> rejectionReasons(DiscreteTokenDatasetCheckpointManifestSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        if (!active()) {
            return List.of();
        }

        List<String> reasons = new ArrayList<>();
        addStringMismatch(reasons, "experimentName", experimentName, snapshot.experimentName());
        addStringMismatch(reasons, "runId", runId, snapshot.runId());
        addStringMismatch(reasons, "modelFamily", modelFamily, snapshot.modelFamily());
        addLongMismatch(reasons, "seed", seed, snapshot.seed());
        addLongMismatch(reasons, "checkpointStep", checkpointStep, snapshot.checkpointStep());
        if (minimumCheckpointStep != null && snapshot.checkpointStep() < minimumCheckpointStep) {
            reasons.add("checkpointStep expected >= "
                    + minimumCheckpointStep
                    + " but checkpoint has "
                    + snapshot.checkpointStep());
        }
        return List.copyOf(reasons);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("active", active());
        putIfPresent(metadata, "experimentName", experimentName);
        putIfPresent(metadata, "runId", runId);
        putIfPresent(metadata, "modelFamily", modelFamily);
        putIfPresent(metadata, "seed", seed);
        putIfPresent(metadata, "checkpointStep", checkpointStep);
        putIfPresent(metadata, "minimumCheckpointStep", minimumCheckpointStep);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void addStringMismatch(
            List<String> reasons,
            String field,
            String expected,
            String actual) {
        if (expected != null && !expected.equals(actual)) {
            reasons.add(field + " expected " + expected + " but checkpoint has " + actual);
        }
    }

    private static void addLongMismatch(
            List<String> reasons,
            String field,
            Long expected,
            long actual) {
        if (expected != null && expected.longValue() != actual) {
            reasons.add(field + " expected " + expected + " but checkpoint has " + actual);
        }
    }

    private static void putIfPresent(Map<String, Object> metadata, String key, Object value) {
        if (value != null) {
            metadata.put(key, value);
        }
    }

    private static String optionalString(Map<?, ?> metadata, String key) {
        Object value = optional(metadata, key);
        if (value == null) {
            return null;
        }
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static Long optionalLong(Map<?, ?> metadata, String key) {
        Object value = optional(metadata, key);
        if (value == null) {
            return null;
        }
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

    private static Object optional(Map<?, ?> metadata, String key) {
        return metadata.containsKey(key) ? metadata.get(key) : null;
    }

    private static String optionalText(String value, String name) {
        if (value == null) {
            return null;
        }
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    private static Long optionalNonNegative(Long value, String name) {
        if (value == null) {
            return null;
        }
        if (value < 0L) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }

    public static final class Builder {
        private String experimentName;
        private String runId;
        private String modelFamily;
        private Long seed;
        private Long checkpointStep;
        private Long minimumCheckpointStep;

        private Builder() {}

        public Builder experimentName(String experimentName) {
            this.experimentName = experimentName;
            return this;
        }

        public Builder runId(String runId) {
            this.runId = runId;
            return this;
        }

        public Builder modelFamily(String modelFamily) {
            this.modelFamily = modelFamily;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder checkpointStep(long checkpointStep) {
            this.checkpointStep = checkpointStep;
            return this;
        }

        public Builder minimumCheckpointStep(long minimumCheckpointStep) {
            this.minimumCheckpointStep = minimumCheckpointStep;
            return this;
        }

        public DiscreteTokenDatasetCheckpointResumeExpectation build() {
            return new DiscreteTokenDatasetCheckpointResumeExpectation(
                    experimentName,
                    runId,
                    modelFamily,
                    seed,
                    checkpointStep,
                    minimumCheckpointStep);
        }
    }
}
