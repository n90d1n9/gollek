package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight ancestry metadata for recursive-reasoning trainer checkpoints.
 */
public record DiscreteTokenDatasetCheckpointLineage(
        String originRunId,
        String parentRunId,
        Long parentCheckpointStep,
        String parentDatasetFingerprint,
        int generation,
        String relation,
        Map<String, Object> attributes) {

    public static final String ROOT_RELATION = "root";
    public static final String RESUME_RELATION = "resume";

    public DiscreteTokenDatasetCheckpointLineage {
        originRunId = requireText(originRunId, "originRunId");
        parentRunId = optionalText(parentRunId, "parentRunId");
        parentCheckpointStep = optionalNonNegative(parentCheckpointStep, "parentCheckpointStep");
        parentDatasetFingerprint = optionalText(parentDatasetFingerprint, "parentDatasetFingerprint");
        if (generation < 0) {
            throw new IllegalArgumentException("generation must be >= 0 but was " + generation);
        }
        relation = requireText(relation, "relation");
        attributes = immutableAttributes(attributes);
        if (ROOT_RELATION.equals(relation)) {
            if (generation != 0 || parentRunId != null || parentCheckpointStep != null || parentDatasetFingerprint != null) {
                throw new IllegalArgumentException("root lineage must have generation 0 and no parent fields");
            }
        } else {
            if (generation < 1 || parentRunId == null || parentCheckpointStep == null || parentDatasetFingerprint == null) {
                throw new IllegalArgumentException("non-root lineage must include parent run, step, fingerprint, and generation >= 1");
            }
        }
    }

    public static DiscreteTokenDatasetCheckpointLineage root(String runId) {
        return new DiscreteTokenDatasetCheckpointLineage(
                runId,
                null,
                null,
                null,
                0,
                ROOT_RELATION,
                Map.of());
    }

    public static DiscreteTokenDatasetCheckpointLineage resumedFrom(
            DiscreteTokenDatasetCheckpointManifestSnapshot parent) {
        return resumedFrom(parent, Map.of());
    }

    public static DiscreteTokenDatasetCheckpointLineage resumedFrom(
            DiscreteTokenDatasetCheckpointManifestSnapshot parent,
            Map<String, Object> attributes) {
        Objects.requireNonNull(parent, "parent must not be null");
        return new DiscreteTokenDatasetCheckpointLineage(
                parent.lineage().originRunId(),
                parent.runId(),
                parent.checkpointStep(),
                parent.fingerprint().value(),
                parent.lineage().generation() + 1,
                RESUME_RELATION,
                attributes);
    }

    public static DiscreteTokenDatasetCheckpointLineage fromMetadata(
            Map<?, ?> metadata,
            String fallbackRunId) {
        if (metadata == null || metadata.isEmpty()) {
            return root(fallbackRunId);
        }
        return new DiscreteTokenDatasetCheckpointLineage(
                optionalString(metadata, "originRunId", fallbackRunId),
                optionalString(metadata, "parentRunId", null),
                optionalLong(metadata, "parentCheckpointStep"),
                optionalString(metadata, "parentDatasetFingerprint", null),
                optionalInt(metadata, "generation", optionalString(metadata, "parentRunId", null) == null ? 0 : 1),
                optionalString(metadata, "relation", optionalString(metadata, "parentRunId", null) == null
                        ? ROOT_RELATION
                        : RESUME_RELATION),
                optionalMap(metadata, "attributes"));
    }

    public boolean root() {
        return ROOT_RELATION.equals(relation);
    }

    public boolean hasParent() {
        return parentRunId != null;
    }

    public String summary() {
        if (root()) {
            return "lineage root " + originRunId;
        }
        return "lineage "
                + relation
                + " generation "
                + generation
                + " from "
                + parentRunId
                + " step "
                + parentCheckpointStep;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("originRunId", originRunId);
        metadata.put("generation", generation);
        metadata.put("relation", relation);
        metadata.put("root", root());
        if (parentRunId != null) {
            metadata.put("parentRunId", parentRunId);
        }
        if (parentCheckpointStep != null) {
            metadata.put("parentCheckpointStep", parentCheckpointStep);
        }
        if (parentDatasetFingerprint != null) {
            metadata.put("parentDatasetFingerprint", parentDatasetFingerprint);
        }
        metadata.put("attributes", attributes);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Map<String, Object> optionalMap(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return Map.of();
        }
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                Object rawKey = Objects.requireNonNull(entry.getKey(), key + " key must not be null");
                if (!(rawKey instanceof CharSequence text)) {
                    throw new IllegalArgumentException("metadata field '" + key + "' keys must be strings");
                }
                String entryKey = requireText(text.toString(), key + " key");
                copy.put(entryKey, Objects.requireNonNull(entry.getValue(), key + " field '" + entryKey + "' must not be null"));
            }
            return Collections.unmodifiableMap(copy);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static String optionalString(Map<?, ?> metadata, String key, String defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static int optionalInt(Map<?, ?> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        long parsed = longValue(value, key);
        if (parsed < Integer.MIN_VALUE || parsed > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("metadata field '" + key + "' must fit an int");
        }
        return (int) parsed;
    }

    private static Long optionalLong(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return null;
        }
        return longValue(value, key);
    }

    private static long longValue(Object value, String key) {
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

    private static String optionalText(String value, String name) {
        if (value == null) {
            return null;
        }
        return requireText(value, name);
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

    private static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    private static Map<String, Object> immutableAttributes(Map<String, Object> attributes) {
        if (attributes == null || attributes.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            String key = requireText(entry.getKey(), "attribute key");
            Object value = Objects.requireNonNull(entry.getValue(), "attribute '" + key + "' must not be null");
            copy.put(key, value);
        }
        return Collections.unmodifiableMap(copy);
    }
}
