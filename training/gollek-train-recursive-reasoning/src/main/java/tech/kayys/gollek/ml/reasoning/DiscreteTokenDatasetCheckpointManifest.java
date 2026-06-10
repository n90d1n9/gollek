package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Trainer/checkpoint manifest that carries dataset preflight provenance.
 */
public record DiscreteTokenDatasetCheckpointManifest(
        String schemaVersion,
        String experimentName,
        String runId,
        String modelFamily,
        long seed,
        long checkpointStep,
        long createdAtEpochMillis,
        String createdBy,
        DiscreteTokenDatasetPlanReport datasetPlanReport,
        DiscreteTokenDatasetCheckpointLineage lineage,
        Map<String, Object> attributes) {

    public static final String SCHEMA_VERSION = "gollek.discrete-token-checkpoint-manifest.v1";
    public static final String DATASET_PLAN_REPORT_METADATA_KEY = "datasetPlanReport";

    public DiscreteTokenDatasetCheckpointManifest {
        schemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(schemaVersion, "schemaVersion");
        experimentName = DiscreteTokenDatasetMetadataSupport.requireText(experimentName, "experimentName");
        runId = DiscreteTokenDatasetMetadataSupport.requireText(runId, "runId");
        modelFamily = DiscreteTokenDatasetMetadataSupport.requireText(modelFamily, "modelFamily");
        if (checkpointStep < 0L) {
            throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
        }
        if (createdAtEpochMillis < 0L) {
            throw new IllegalArgumentException("createdAtEpochMillis must be >= 0 but was " + createdAtEpochMillis);
        }
        createdBy = DiscreteTokenDatasetMetadataSupport.requireText(createdBy, "createdBy");
        datasetPlanReport = Objects.requireNonNull(datasetPlanReport, "datasetPlanReport must not be null");
        lineage = lineage == null ? DiscreteTokenDatasetCheckpointLineage.root(runId) : lineage;
        attributes = immutableAttributes(attributes);
    }

    public DiscreteTokenDatasetCheckpointManifest(
            String schemaVersion,
            String experimentName,
            String runId,
            String modelFamily,
            long seed,
            long checkpointStep,
            long createdAtEpochMillis,
            String createdBy,
            DiscreteTokenDatasetPlanReport datasetPlanReport,
            Map<String, Object> attributes) {
        this(
                schemaVersion,
                experimentName,
                runId,
                modelFamily,
                seed,
                checkpointStep,
                createdAtEpochMillis,
                createdBy,
                datasetPlanReport,
                null,
                attributes);
    }

    public static Builder builder(DiscreteTokenDatasetPlanReport datasetPlanReport) {
        return new Builder(datasetPlanReport);
    }

    public DiscreteTokenDatasetFingerprint fingerprint() {
        return datasetPlanReport.fingerprint();
    }

    public boolean datasetAccepted() {
        return datasetPlanReport.accepted();
    }

    public String datasetGateStatus() {
        return datasetPlanReport.gateStatus();
    }

    public void requireDatasetAccepted() {
        datasetPlanReport.requireAccepted();
    }

    public DiscreteTokenDatasetFingerprintMatch verifyPlan(DiscreteTokenDatasetPlan plan) {
        return DiscreteTokenDatasetFingerprintMatch.verify(fingerprint(), plan);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyReport(DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetFingerprintMatch.verify(fingerprint(), report);
    }

    public String summary() {
        return "checkpoint "
                + runId
                + " step "
                + checkpointStep
                + " dataset "
                + fingerprint().shortValue()
                + " "
                + datasetGateStatus();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("schemaVersion", schemaVersion);
        metadata.put("experimentName", experimentName);
        metadata.put("runId", runId);
        metadata.put("modelFamily", modelFamily);
        metadata.put("seed", seed);
        metadata.put("checkpointStep", checkpointStep);
        metadata.put("createdAtEpochMillis", createdAtEpochMillis);
        metadata.put("createdBy", createdBy);
        metadata.put("fingerprint", fingerprint().toMetadata());
        metadata.put("datasetAccepted", datasetAccepted());
        metadata.put("datasetGateStatus", datasetGateStatus());
        metadata.put(DATASET_PLAN_REPORT_METADATA_KEY, datasetPlanReport.toMetadata());
        metadata.put("lineage", lineage.toMetadata());
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

    public static final class Builder {
        private String schemaVersion = SCHEMA_VERSION;
        private String experimentName = "default";
        private String runId;
        private String modelFamily = "recursive-reasoning";
        private long seed;
        private long checkpointStep;
        private long createdAtEpochMillis = System.currentTimeMillis();
        private String createdBy = "gollek";
        private final DiscreteTokenDatasetPlanReport datasetPlanReport;
        private DiscreteTokenDatasetCheckpointLineage lineage;
        private Map<String, Object> attributes = Map.of();

        private Builder(DiscreteTokenDatasetPlanReport datasetPlanReport) {
            this.datasetPlanReport = Objects.requireNonNull(datasetPlanReport, "datasetPlanReport must not be null");
            this.runId = datasetPlanReport.fingerprint().shortValue();
        }

        public Builder schemaVersion(String schemaVersion) {
            this.schemaVersion = schemaVersion;
            return this;
        }

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

        public Builder createdAtEpochMillis(long createdAtEpochMillis) {
            this.createdAtEpochMillis = createdAtEpochMillis;
            return this;
        }

        public Builder createdBy(String createdBy) {
            this.createdBy = createdBy;
            return this;
        }

        public Builder lineage(DiscreteTokenDatasetCheckpointLineage lineage) {
            this.lineage = Objects.requireNonNull(lineage, "lineage must not be null");
            return this;
        }

        public Builder lineageFrom(DiscreteTokenDatasetCheckpointManifestSnapshot parent) {
            this.lineage = DiscreteTokenDatasetCheckpointLineage.resumedFrom(parent);
            return this;
        }

        public Builder lineageFrom(
                DiscreteTokenDatasetCheckpointManifestSnapshot parent,
                Map<String, Object> attributes) {
            this.lineage = DiscreteTokenDatasetCheckpointLineage.resumedFrom(parent, attributes);
            return this;
        }

        public Builder attributes(Map<String, Object> attributes) {
            this.attributes = attributes;
            return this;
        }

        public DiscreteTokenDatasetCheckpointManifest build() {
            String resolvedRunId = runId == null ? datasetPlanReport.fingerprint().shortValue() : runId;
            return new DiscreteTokenDatasetCheckpointManifest(
                    schemaVersion,
                    experimentName,
                    resolvedRunId,
                    modelFamily,
                    seed,
                    checkpointStep,
                    createdAtEpochMillis,
                    createdBy,
                    datasetPlanReport,
                    lineage == null ? DiscreteTokenDatasetCheckpointLineage.root(resolvedRunId) : lineage,
                    attributes);
        }
    }
}
