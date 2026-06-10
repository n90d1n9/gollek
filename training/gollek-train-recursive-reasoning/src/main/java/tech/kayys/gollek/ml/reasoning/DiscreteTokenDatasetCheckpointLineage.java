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
        originRunId = DiscreteTokenDatasetMetadataSupport.requireText(originRunId, "originRunId");
        parentRunId = DiscreteTokenDatasetMetadataSupport.optionalText(parentRunId, "parentRunId");
        parentCheckpointStep =
                DiscreteTokenDatasetMetadataSupport.optionalNonNegative(parentCheckpointStep, "parentCheckpointStep");
        parentDatasetFingerprint =
                DiscreteTokenDatasetMetadataSupport.optionalText(parentDatasetFingerprint, "parentDatasetFingerprint");
        if (generation < 0) {
            throw new IllegalArgumentException("generation must be >= 0 but was " + generation);
        }
        relation = DiscreteTokenDatasetMetadataSupport.requireText(relation, "relation");
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
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "originRunId", fallbackRunId),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "parentRunId", null),
                DiscreteTokenDatasetMetadataSupport.optionalLong(metadata, "parentCheckpointStep"),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "parentDatasetFingerprint", null),
                DiscreteTokenDatasetMetadataSupport.optionalInt(
                        metadata,
                        "generation",
                        DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "parentRunId", null) == null
                                ? 0
                                : 1),
                DiscreteTokenDatasetMetadataSupport.optionalString(
                        metadata,
                        "relation",
                        DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "parentRunId", null) == null
                        ? ROOT_RELATION
                        : RESUME_RELATION),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "attributes"));
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

    private static Map<String, Object> immutableAttributes(Map<String, Object> attributes) {
        if (attributes == null || attributes.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            String key = DiscreteTokenDatasetMetadataSupport.requireText(entry.getKey(), "attribute key");
            Object value = Objects.requireNonNull(entry.getValue(), "attribute '" + key + "' must not be null");
            copy.put(key, value);
        }
        return Collections.unmodifiableMap(copy);
    }
}
