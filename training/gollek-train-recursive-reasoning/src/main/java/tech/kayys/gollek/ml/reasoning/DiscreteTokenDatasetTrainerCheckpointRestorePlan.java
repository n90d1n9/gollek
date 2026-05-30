package tech.kayys.gollek.ml.reasoning;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Typed handoff for resuming a recursive-reasoning trainer from a selected checkpoint.
 */
public record DiscreteTokenDatasetTrainerCheckpointRestorePlan(
        Path rootDir,
        Path checkpointDir,
        Path manifestPath,
        Path resumeReportPath,
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy,
        DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {

    public DiscreteTokenDatasetTrainerCheckpointRestorePlan {
        rootDir = Objects.requireNonNull(rootDir, "rootDir must not be null");
        checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
        manifestPath = Objects.requireNonNull(manifestPath, "manifestPath must not be null");
        resumeReportPath = Objects.requireNonNull(resumeReportPath, "resumeReportPath must not be null");
        policy = Objects.requireNonNull(policy, "policy must not be null");
        checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        if (!checkpointDir.equals(checkpoint.checkpointDir())
                || !manifestPath.equals(checkpoint.manifestPath())
                || !resumeReportPath.equals(checkpoint.resumeReportPath())) {
            throw new IllegalArgumentException("restore plan paths must match the selected checkpoint snapshot");
        }
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePlan fromInspectionReport(
            DiscreteTokenDatasetTrainerCheckpointInspectionReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return fromCheckpoint(report.rootDir(), report.requireSelectedCheckpoint(), report.policy());
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePlan fromCheckpoint(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        return new DiscreteTokenDatasetTrainerCheckpointRestorePlan(
                rootDir,
                checkpoint.checkpointDir(),
                checkpoint.manifestPath(),
                checkpoint.resumeReportPath(),
                policy,
                checkpoint);
    }

    public DiscreteTokenDatasetCheckpointManifestSnapshot manifest() {
        return checkpoint.manifest();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeReportSnapshot> resumeReport() {
        return checkpoint.resumeReport();
    }

    public String experimentName() {
        return manifest().experimentName();
    }

    public String runId() {
        return manifest().runId();
    }

    public String modelFamily() {
        return manifest().modelFamily();
    }

    public long seed() {
        return manifest().seed();
    }

    public long checkpointStep() {
        return manifest().checkpointStep();
    }

    public long createdAtEpochMillis() {
        return manifest().createdAtEpochMillis();
    }

    public String status() {
        return checkpoint.status();
    }

    public boolean ready() {
        return checkpoint.ready();
    }

    public boolean manifestReady() {
        return checkpoint.manifestReady();
    }

    public boolean resumeReportPresent() {
        return checkpoint.resumeReportPresent();
    }

    public boolean resumeReady() {
        return checkpoint.resumeReady();
    }

    public boolean resumeReportRequired() {
        return policy.requireResumeReport();
    }

    public boolean strictInventory() {
        return policy.failOnInventoryFailures();
    }

    public void requireReady() {
        checkpoint.requireReady();
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireResumeReport() {
        return checkpoint.requireResumeReport();
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireResumeReady() {
        return checkpoint.requireResumeReady();
    }

    public String summary() {
        return "restore checkpoint "
                + runId()
                + " step "
                + checkpointStep()
                + " from "
                + checkpointDir
                + " ("
                + status()
                + ")";
    }

    public Map<String, Object> pathsMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rootDir", rootDir.toString());
        metadata.put("checkpointDir", checkpointDir.toString());
        metadata.put("manifestPath", manifestPath.toString());
        metadata.put("resumeReportPath", resumeReportPath.toString());
        metadata.put("manifestFileName", DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME);
        metadata.put("resumeReportFileName", DiscreteTokenDatasetTrainerCheckpointBridge.RESUME_REPORT_FILE_NAME);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Map<String, Object> identityMetadata() {
        DiscreteTokenDatasetCheckpointManifestSnapshot manifest = manifest();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("experimentName", manifest.experimentName());
        metadata.put("runId", manifest.runId());
        metadata.put("modelFamily", manifest.modelFamily());
        metadata.put("seed", manifest.seed());
        metadata.put("checkpointStep", manifest.checkpointStep());
        metadata.put("createdAtEpochMillis", manifest.createdAtEpochMillis());
        metadata.put("createdBy", manifest.createdBy());
        metadata.put("fingerprint", manifest.fingerprint().toMetadata());
        metadata.put("fingerprintShort", manifest.fingerprint().shortValue());
        metadata.put("lineage", manifest.lineage().toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Map<String, Object> lineageMetadata() {
        return manifest().lineage().toMetadata();
    }

    public Map<String, Object> checkpointMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("checkpointDir", checkpointDir.toString());
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("manifestReady", manifestReady());
        metadata.put("resumeReportPresent", resumeReportPresent());
        metadata.put("resumeReady", resumeReady());
        metadata.put("resumeStatus", checkpoint.resumeStatus());
        metadata.put("identity", identityMetadata());
        metadata.put("lineage", lineageMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Optional<Map<String, Object>> resumeMetadata() {
        return resumeReport().map(report -> {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("status", report.status());
            metadata.put("ready", report.ready());
            metadata.put("runId", report.runId());
            metadata.put("checkpointStep", report.checkpointStep());
            metadata.put("schemaAccepted", report.schemaAccepted());
            metadata.put("datasetAccepted", report.datasetAccepted());
            metadata.put("fingerprintMatched", report.fingerprintMatched());
            metadata.put("expectationAccepted", report.expectationAccepted());
            metadata.put("currentPlanChecked", report.currentPlanChecked());
            metadata.put("currentPlanAccepted", report.currentPlanAccepted());
            metadata.put("currentPlanGateStatus", report.currentPlanGateStatus());
            metadata.put("policyTracked", report.policyTracked());
            metadata.put("rejectionReasons", report.rejectionReasons());
            metadata.put("expectation", report.expectation().toMetadata());
            metadata.put("fingerprintMatch", report.fingerprintMatch().toMetadata());
            if (report.policyTracked()) {
                metadata.put("policy", report.policyMetadata());
            }
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        });
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("summary", summary());
        metadata.put("rootDir", rootDir.toString());
        metadata.put("checkpointDir", checkpointDir.toString());
        metadata.put("manifestPath", manifestPath.toString());
        metadata.put("resumeReportPath", resumeReportPath.toString());
        metadata.put("experimentName", experimentName());
        metadata.put("runId", runId());
        metadata.put("modelFamily", modelFamily());
        metadata.put("seed", seed());
        metadata.put("checkpointStep", checkpointStep());
        metadata.put("createdAtEpochMillis", createdAtEpochMillis());
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("manifestReady", manifestReady());
        metadata.put("resumeReportRequired", resumeReportRequired());
        metadata.put("resumeReportPresent", resumeReportPresent());
        metadata.put("resumeReady", resumeReady());
        metadata.put("strictInventory", strictInventory());
        metadata.put("paths", pathsMetadata());
        metadata.put("identity", identityMetadata());
        metadata.put("lineage", lineageMetadata());
        metadata.put("policy", policy.toMetadata());
        metadata.put("checkpoint", checkpointMetadata());
        resumeMetadata().ifPresent(resume -> metadata.put("resume", resume));
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }
}
