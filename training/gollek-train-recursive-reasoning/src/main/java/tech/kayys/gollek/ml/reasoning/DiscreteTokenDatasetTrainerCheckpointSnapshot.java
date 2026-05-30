package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Read-side snapshot of recursive-reasoning trainer checkpoint sidecars.
 */
public record DiscreteTokenDatasetTrainerCheckpointSnapshot(
        Path checkpointDir,
        Path manifestPath,
        Path resumeReportPath,
        DiscreteTokenDatasetCheckpointManifestSnapshot manifest,
        Optional<DiscreteTokenDatasetCheckpointResumeReportSnapshot> resumeReport) {

    public DiscreteTokenDatasetTrainerCheckpointSnapshot {
        checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
        manifestPath = Objects.requireNonNull(manifestPath, "manifestPath must not be null");
        resumeReportPath = Objects.requireNonNull(resumeReportPath, "resumeReportPath must not be null");
        manifest = Objects.requireNonNull(manifest, "manifest must not be null");
        resumeReport = Objects.requireNonNullElse(resumeReport, Optional.empty());
        if (resumeReport.isPresent()) {
            verifyResumeMatchesManifest(manifest, resumeReport.orElseThrow());
        }
    }

    public static DiscreteTokenDatasetTrainerCheckpointSnapshot read(Path checkpointDir) throws IOException {
        Path manifestPath = DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir);
        Path resumeReportPath = DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir);
        Optional<DiscreteTokenDatasetCheckpointResumeReportSnapshot> resumeReport = Files.isRegularFile(resumeReportPath)
                ? Optional.of(DiscreteTokenDatasetCheckpointMetadataJson.readResumeReportSnapshot(resumeReportPath))
                : Optional.empty();
        return new DiscreteTokenDatasetTrainerCheckpointSnapshot(
                checkpointDir,
                manifestPath,
                resumeReportPath,
                DiscreteTokenDatasetCheckpointMetadataJson.readSnapshot(manifestPath),
                resumeReport);
    }

    public boolean manifestReady() {
        return manifest.isCurrentSchema() && manifest.datasetAccepted();
    }

    public boolean resumeReportPresent() {
        return resumeReport.isPresent();
    }

    public boolean resumeReady() {
        return resumeReport.map(DiscreteTokenDatasetCheckpointResumeReportSnapshot::ready).orElse(false);
    }

    public String resumeStatus() {
        return resumeReport.map(DiscreteTokenDatasetCheckpointResumeReportSnapshot::status).orElse("missing");
    }

    public boolean ready() {
        return manifestReady() && resumeReport.map(DiscreteTokenDatasetCheckpointResumeReportSnapshot::ready).orElse(true);
    }

    public String status() {
        if (!manifest.isCurrentSchema()) {
            return "manifest-schema-mismatch";
        }
        if (!manifest.datasetAccepted()) {
            return "manifest-blocked";
        }
        return resumeReport.map(DiscreteTokenDatasetCheckpointResumeReportSnapshot::status).orElse("manifest-only");
    }

    public String summary() {
        String suffix = resumeReport
                .map(report -> "; " + report.summary())
                .orElse("; resume report missing");
        return "checkpoint directory " + status() + ": " + manifest.summary() + suffix;
    }

    public void requireReady() {
        manifest.requireCurrentSchema();
        manifest.requireDatasetAccepted();
        resumeReport.ifPresent(DiscreteTokenDatasetCheckpointResumeReportSnapshot::requireReady);
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireResumeReport() {
        return resumeReport.orElseThrow(() -> new IllegalStateException(
                "checkpoint resume report is missing: " + resumeReportPath));
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireResumeReady() {
        DiscreteTokenDatasetCheckpointResumeReportSnapshot report = requireResumeReport();
        report.requireReady();
        return report;
    }

    public Map<String, Object> trainingReportMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("manifestFileName", DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME);
        metadata.put("resumeReportFileName", DiscreteTokenDatasetTrainerCheckpointBridge.RESUME_REPORT_FILE_NAME);
        metadata.put("checkpoint", manifest.toMetadata());
        metadata.put("fingerprint", manifest.fingerprint().toMetadata());
        metadata.put("datasetAccepted", manifest.datasetAccepted());
        metadata.put("datasetGateStatus", manifest.datasetGateStatus());
        metadata.put("lineage", manifest.lineage().toMetadata());
        resumeReport.ifPresent(report -> metadata.put("resume", report.toMetadata()));
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("checkpointDir", checkpointDir.toString());
        metadata.put("manifestPath", manifestPath.toString());
        metadata.put("resumeReportPath", resumeReportPath.toString());
        metadata.put("manifestFileName", DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME);
        metadata.put("resumeReportFileName", DiscreteTokenDatasetTrainerCheckpointBridge.RESUME_REPORT_FILE_NAME);
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("manifestReady", manifestReady());
        metadata.put("resumeReportPresent", resumeReportPresent());
        metadata.put("resumeReady", resumeReady());
        metadata.put("resumeStatus", resumeStatus());
        metadata.put("checkpoint", manifest.toMetadata());
        metadata.put("lineage", manifest.lineage().toMetadata());
        resumeReport.ifPresent(report -> metadata.put("resume", report.toMetadata()));
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyResumeMatchesManifest(
            DiscreteTokenDatasetCheckpointManifestSnapshot manifest,
            DiscreteTokenDatasetCheckpointResumeReportSnapshot resumeReport) {
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint = resumeReport.checkpoint();
        if (!manifest.schemaVersion().equals(checkpoint.schemaVersion())
                || !manifest.runId().equals(checkpoint.runId())
                || manifest.checkpointStep() != checkpoint.checkpointStep()
                || !manifest.fingerprint().equals(checkpoint.fingerprint())
                || manifest.datasetAccepted() != checkpoint.datasetAccepted()
                || !manifest.datasetGateStatus().equals(checkpoint.datasetGateStatus())) {
            throw new IllegalArgumentException(
                    "resume report checkpoint snapshot must match the checkpoint manifest sidecar");
        }
    }
}
