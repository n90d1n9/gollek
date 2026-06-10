package tech.kayys.gollek.ml.reasoning;

import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.immutableStringMap;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredString;

import java.util.Map;

/**
 * Typed read-side wrapper for one schema-lock validation mismatch.
 */
public record DiscreteTokenDatasetSchemaLockMismatch(Map<String, Object> metadata) {
    public DiscreteTokenDatasetSchemaLockMismatch {
        metadata = immutableStringMap(metadata, "metadata");
        requiredString(metadata, "type");
        requiredString(metadata, "path");
        requirePresent(metadata, "expected");
        requirePresent(metadata, "actual");
    }

    public static DiscreteTokenDatasetSchemaLockMismatch fromMetadata(Map<?, ?> metadata) {
        return new DiscreteTokenDatasetSchemaLockMismatch(immutableStringMap(metadata, "metadata"));
    }

    public String type() {
        return requiredString(metadata, "type");
    }

    public String path() {
        return requiredString(metadata, "path");
    }

    public Object expected() {
        return metadata.get("expected");
    }

    public Object actual() {
        return metadata.get("actual");
    }

    public String summary() {
        return path() + " (" + type() + ")";
    }

    public Map<String, Object> toMetadata() {
        return metadata;
    }

    private static void requirePresent(Map<String, Object> metadata, String field) {
        if (!metadata.containsKey(field)) {
            throw new IllegalArgumentException(field + " must be present");
        }
    }
}
