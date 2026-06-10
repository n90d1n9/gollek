package tech.kayys.gollek.ml.reasoning;

import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.immutableStringMap;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredBoolean;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredInt;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredMapList;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredString;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Typed read-side wrapper for schema-lock validation metadata.
 */
public record DiscreteTokenDatasetSchemaLockValidationReport(Map<String, Object> metadata) {
    public DiscreteTokenDatasetSchemaLockValidationReport {
        metadata = immutableStringMap(metadata, "metadata");
        boolean valid = requiredBoolean(metadata, "valid");
        String expectedStatus = valid ? "valid" : "invalid";
        if (!expectedStatus.equals(requiredString(metadata, "status"))) {
            throw new IllegalArgumentException("status must be '" + expectedStatus + "' when valid=" + valid);
        }
        requiredString(metadata, "kind");
        requiredString(metadata, "schemaVersion");
        requiredString(metadata, "validationSchemaId");
        requiredString(metadata, "validationSchemaResource");
        requiredString(metadata, "code");
        requiredString(metadata, "message");
        requiredString(metadata, "expectedSchemaSetSha256");
        requiredString(metadata, "actualSchemaSetSha256");
        requiredBoolean(metadata, "schemaSetMatch");
        requiredInt(metadata, "expectedSchemaCount");
        requiredInt(metadata, "actualSchemaCount");
        requiredString(metadata, "expectedLockSchemaId");
        requiredString(metadata, "actualLockSchemaId");
        List<Map<String, Object>> mismatches = requiredMapList(metadata, "mismatches");
        int mismatchCount = requiredInt(metadata, "mismatchCount");
        if (mismatchCount != mismatches.size()) {
            throw new IllegalArgumentException("mismatchCount must equal mismatches size");
        }
        mismatches.forEach(DiscreteTokenDatasetSchemaLockMismatch::fromMetadata);
        requiredString(metadata, "summary");
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport fromMetadata(Map<?, ?> metadata) {
        return new DiscreteTokenDatasetSchemaLockValidationReport(immutableStringMap(metadata, "metadata"));
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport fromJson(String json) {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json));
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport fromPath(Path path) throws IOException {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.read(path));
    }

    public String kind() {
        return requiredString(metadata, "kind");
    }

    public String schemaVersion() {
        return requiredString(metadata, "schemaVersion");
    }

    public String status() {
        return requiredString(metadata, "status");
    }

    public boolean valid() {
        return requiredBoolean(metadata, "valid");
    }

    public boolean invalid() {
        return !valid();
    }

    public String code() {
        return requiredString(metadata, "code");
    }

    public String message() {
        return requiredString(metadata, "message");
    }

    public String expectedSchemaSetSha256() {
        return requiredString(metadata, "expectedSchemaSetSha256");
    }

    public String actualSchemaSetSha256() {
        return requiredString(metadata, "actualSchemaSetSha256");
    }

    public boolean schemaSetMatch() {
        return requiredBoolean(metadata, "schemaSetMatch");
    }

    public DiscreteTokenDatasetSchemaSetIdentity expectedSchemaSetIdentity() {
        return new DiscreteTokenDatasetSchemaSetIdentity(expectedSchemaSetSha256(), expectedSchemaCount());
    }

    public DiscreteTokenDatasetSchemaSetIdentity actualSchemaSetIdentity() {
        return new DiscreteTokenDatasetSchemaSetIdentity(actualSchemaSetSha256(), actualSchemaCount());
    }

    public boolean schemaSetIdentityMatch() {
        return expectedSchemaSetIdentity().matches(actualSchemaSetIdentity());
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireSchemaSetIdentityMatch() {
        return expectedSchemaSetIdentity().requireMatches(
                actualSchemaSetIdentity(),
                "schema lock validation schema set");
    }

    public int expectedSchemaCount() {
        return requiredInt(metadata, "expectedSchemaCount");
    }

    public int actualSchemaCount() {
        return requiredInt(metadata, "actualSchemaCount");
    }

    public String expectedLockSchemaId() {
        return requiredString(metadata, "expectedLockSchemaId");
    }

    public String actualLockSchemaId() {
        return requiredString(metadata, "actualLockSchemaId");
    }

    public int mismatchCount() {
        return requiredInt(metadata, "mismatchCount");
    }

    public boolean hasMismatches() {
        return mismatchCount() > 0;
    }

    public List<Map<String, Object>> mismatches() {
        return requiredMapList(metadata, "mismatches");
    }

    public List<DiscreteTokenDatasetSchemaLockMismatch> mismatchDetails() {
        return mismatches().stream()
                .map(DiscreteTokenDatasetSchemaLockMismatch::fromMetadata)
                .toList();
    }

    public Optional<DiscreteTokenDatasetSchemaLockMismatch> firstMismatch() {
        return mismatchDetails().stream().findFirst();
    }

    public boolean hasMismatchAtPath(String path) {
        Objects.requireNonNull(path, "path must not be null");
        return mismatchDetails().stream().anyMatch(mismatch -> mismatch.path().equals(path));
    }

    public boolean hasMismatchType(String type) {
        Objects.requireNonNull(type, "type must not be null");
        return mismatchDetails().stream().anyMatch(mismatch -> mismatch.type().equals(type));
    }

    public List<DiscreteTokenDatasetSchemaLockMismatch> mismatchesForPath(String path) {
        Objects.requireNonNull(path, "path must not be null");
        return mismatchDetails().stream()
                .filter(mismatch -> mismatch.path().equals(path))
                .toList();
    }

    public List<DiscreteTokenDatasetSchemaLockMismatch> mismatchesForType(String type) {
        Objects.requireNonNull(type, "type must not be null");
        return mismatchDetails().stream()
                .filter(mismatch -> mismatch.type().equals(type))
                .toList();
    }

    public List<String> mismatchPaths() {
        return mismatchDetails().stream()
                .map(DiscreteTokenDatasetSchemaLockMismatch::path)
                .toList();
    }

    public List<String> mismatchTypes() {
        return mismatchDetails().stream()
                .map(DiscreteTokenDatasetSchemaLockMismatch::type)
                .distinct()
                .toList();
    }

    public Map<String, List<DiscreteTokenDatasetSchemaLockMismatch>> mismatchesByPath() {
        return groupMismatches(false);
    }

    public Map<String, List<DiscreteTokenDatasetSchemaLockMismatch>> mismatchesByType() {
        return groupMismatches(true);
    }

    public String summary() {
        return requiredString(metadata, "summary");
    }

    public Map<String, Object> toMetadata() {
        return metadata;
    }

    public void requireValid() {
        if (!valid()) {
            throw new IllegalStateException(failureMessage());
        }
    }

    private String failureMessage() {
        StringBuilder builder = new StringBuilder(message());
        builder.append(": ").append(summary());
        if (!mismatches().isEmpty()) {
            builder.append("; mismatches=");
            int count = 0;
            for (DiscreteTokenDatasetSchemaLockMismatch mismatch : mismatchDetails()) {
                if (count > 0) {
                    builder.append(", ");
                }
                if (count >= 5) {
                    builder.append("...");
                    break;
                }
                builder.append(mismatch.summary());
                count++;
            }
        }
        return builder.toString();
    }

    private Map<String, List<DiscreteTokenDatasetSchemaLockMismatch>> groupMismatches(boolean byType) {
        Map<String, List<DiscreteTokenDatasetSchemaLockMismatch>> grouped = new LinkedHashMap<>();
        for (DiscreteTokenDatasetSchemaLockMismatch mismatch : mismatchDetails()) {
            String key = byType ? mismatch.type() : mismatch.path();
            grouped.computeIfAbsent(key, ignored -> new ArrayList<>()).add(mismatch);
        }
        Map<String, List<DiscreteTokenDatasetSchemaLockMismatch>> immutable = new LinkedHashMap<>();
        grouped.forEach((key, value) -> immutable.put(key, List.copyOf(value)));
        return Collections.unmodifiableMap(immutable);
    }
}
