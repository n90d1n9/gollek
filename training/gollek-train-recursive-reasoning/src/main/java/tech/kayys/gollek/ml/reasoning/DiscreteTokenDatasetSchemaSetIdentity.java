package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Typed schema-set identity used by schema catalogs, schema locks, and validation reports.
 */
public record DiscreteTokenDatasetSchemaSetIdentity(String schemaSetSha256, int schemaCount) {
    public DiscreteTokenDatasetSchemaSetIdentity {
        schemaSetSha256 = Objects.requireNonNull(schemaSetSha256, "schemaSetSha256 must not be null");
        if (schemaSetSha256.isBlank()) {
            throw new IllegalArgumentException("schemaSetSha256 must not be blank");
        }
        if (schemaCount < 0) {
            throw new IllegalArgumentException("schemaCount must not be negative");
        }
    }

    public static DiscreteTokenDatasetSchemaSetIdentity fromCatalogSnapshot(
            DiscreteTokenDatasetSchemaCatalogSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        return new DiscreteTokenDatasetSchemaSetIdentity(snapshot.schemaSetSha256(), snapshot.schemaCount());
    }

    public static DiscreteTokenDatasetSchemaSetIdentity fromLockSnapshot(
            DiscreteTokenDatasetSchemaLockSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        return new DiscreteTokenDatasetSchemaSetIdentity(snapshot.schemaSetSha256(), snapshot.schemaCount());
    }

    public boolean matches(DiscreteTokenDatasetSchemaSetIdentity other) {
        return equals(Objects.requireNonNull(other, "other must not be null"));
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireMatches(DiscreteTokenDatasetSchemaSetIdentity other) {
        return requireMatches(other, "schema set");
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireMatches(
            DiscreteTokenDatasetSchemaSetIdentity other,
            String label) {
        Objects.requireNonNull(other, "other must not be null");
        label = Objects.requireNonNull(label, "label must not be null");
        if (label.isBlank()) {
            throw new IllegalArgumentException("label must not be blank");
        }
        if (!matches(other)) {
            throw new IllegalStateException(label
                    + " mismatch: expected "
                    + schemaSetSha256
                    + " / "
                    + schemaCount
                    + " schema(s), actual "
                    + other.schemaSetSha256
                    + " / "
                    + other.schemaCount
                    + " schema(s)");
        }
        return this;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("schemaSetSha256", schemaSetSha256);
        metadata.put("schemaCount", schemaCount);
        return Collections.unmodifiableMap(metadata);
    }
}
