package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Bridges recursive-reasoning dataset provenance into trainer checkpoint folders.
 */
public final class DiscreteTokenDatasetTrainerCheckpointBridge {
    public static final String MANIFEST_FILE_NAME = "discrete-token-dataset-manifest.json";
    public static final String RESUME_REPORT_FILE_NAME = "discrete-token-dataset-resume-report.json";
    public static final String TRAINING_REPORT_METADATA_KEY = "recursiveReasoningDataset";

    private DiscreteTokenDatasetTrainerCheckpointBridge() {}

    public static Path manifestPath(Path checkpointDir) {
        return checkpointDir(checkpointDir).resolve(MANIFEST_FILE_NAME);
    }

    public static Path resumeReportPath(Path checkpointDir) {
        return checkpointDir(checkpointDir).resolve(RESUME_REPORT_FILE_NAME);
    }

    public static void writeManifest(
            Path checkpointDir,
            DiscreteTokenDatasetCheckpointManifest manifest) throws IOException {
        Objects.requireNonNull(manifest, "manifest must not be null");
        DiscreteTokenDatasetCheckpointMetadataJson.write(manifestPath(checkpointDir), manifest);
    }

    public static Map<String, Object> readManifestMetadata(Path checkpointDir) throws IOException {
        return DiscreteTokenDatasetCheckpointMetadataJson.read(manifestPath(checkpointDir));
    }

    public static DiscreteTokenDatasetCheckpointManifestSnapshot readSnapshot(Path checkpointDir) throws IOException {
        return DiscreteTokenDatasetCheckpointMetadataJson.readSnapshot(manifestPath(checkpointDir));
    }

    public static Map<String, Object> readResumeReportMetadata(Path checkpointDir) throws IOException {
        return DiscreteTokenDatasetCheckpointMetadataJson.read(resumeReportPath(checkpointDir));
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot readResumeReportSnapshot(
            Path checkpointDir) throws IOException {
        return DiscreteTokenDatasetCheckpointMetadataJson.readResumeReportSnapshot(resumeReportPath(checkpointDir));
    }

    public static DiscreteTokenDatasetTrainerCheckpointSnapshot readCheckpointSnapshot(
            Path checkpointDir) throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointSnapshot.read(checkpointDir);
    }

    public static DiscreteTokenDatasetTrainerCheckpointInventory scanCheckpointInventory(
            Path rootDir) throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);
    }

    public static Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> selectCheckpoint(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.select(scanCheckpointInventory(rootDir));
    }

    public static DiscreteTokenDatasetTrainerCheckpointInspectionReport inspectCheckpoints(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        Objects.requireNonNull(policy, "policy must not be null");
        return DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(rootDir, policy);
    }

    public static Optional<DiscreteTokenDatasetTrainerCheckpointRestorePlan> selectRestorePlan(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        return inspectCheckpoints(rootDir, policy).selectedRestorePlan();
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePlan requireRestorePlan(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        return inspectCheckpoints(rootDir, policy).requireRestorePlan();
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluateRestorePreflight(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        return DiscreteTokenDatasetTrainerCheckpointRestorePreflight.evaluate(
                restorePlan,
                currentPlan,
                resumePolicy);
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluateRestorePreflight(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        return DiscreteTokenDatasetTrainerCheckpointRestorePreflight.evaluate(
                restorePlan,
                currentPlanReport,
                resumePolicy);
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireRestorePreflightReady(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        return evaluateRestorePreflight(restorePlan, currentPlan, resumePolicy).requireReady();
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireRestorePreflightReady(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        return evaluateRestorePreflight(restorePlan, currentPlanReport, resumePolicy).requireReady();
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluateRestorePreflight(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy selectionPolicy,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) throws IOException {
        return evaluateRestorePreflight(
                requireRestorePlan(rootDir, selectionPolicy),
                currentPlan,
                resumePolicy);
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluateRestorePreflight(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy selectionPolicy,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) throws IOException {
        return evaluateRestorePreflight(
                requireRestorePlan(rootDir, selectionPolicy),
                currentPlanReport,
                resumePolicy);
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireRestorePreflightReady(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy selectionPolicy,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) throws IOException {
        return evaluateRestorePreflight(rootDir, selectionPolicy, currentPlan, resumePolicy).requireReady();
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireRestorePreflightReady(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy selectionPolicy,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) throws IOException {
        return evaluateRestorePreflight(rootDir, selectionPolicy, currentPlanReport, resumePolicy).requireReady();
    }

    public static DiscreteTokenDatasetTrainerCheckpointSnapshot requireCheckpoint(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.require(scanCheckpointInventory(rootDir));
    }

    public static DiscreteTokenDatasetCheckpointResumeReport evaluateResume(
            Path checkpointDir,
            DiscreteTokenDatasetPlan currentPlan) throws IOException {
        return evaluateResume(
                checkpointDir,
                currentPlan,
                DiscreteTokenDatasetCheckpointResumePolicy.training());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport evaluateResume(
            Path checkpointDir,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy policy) throws IOException {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.evaluate(readSnapshot(checkpointDir), currentPlan);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport requireResumeReady(
            Path checkpointDir,
            DiscreteTokenDatasetPlan currentPlan) throws IOException {
        return requireResumeReady(
                checkpointDir,
                currentPlan,
                DiscreteTokenDatasetCheckpointResumePolicy.training());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport requireResumeReady(
            Path checkpointDir,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy policy) throws IOException {
        DiscreteTokenDatasetCheckpointResumeReport report = evaluateResume(checkpointDir, currentPlan, policy);
        report.requireReady();
        return report;
    }

    public static void writeResumeReport(
            Path checkpointDir,
            DiscreteTokenDatasetCheckpointResumeReport report) throws IOException {
        Objects.requireNonNull(report, "report must not be null");
        DiscreteTokenDatasetCheckpointMetadataJson.write(resumeReportPath(checkpointDir), report.toMetadata());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport requireResumeReadyAndWriteReport(
            Path checkpointDir,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy policy) throws IOException {
        DiscreteTokenDatasetCheckpointResumeReport report = requireResumeReady(checkpointDir, currentPlan, policy);
        writeResumeReport(checkpointDir, report);
        return report;
    }

    public static Map<String, Object> trainingReportMetadata(
            DiscreteTokenDatasetCheckpointManifest manifest) {
        return trainingReportMetadata(manifest, null);
    }

    public static Map<String, Object> trainingReportMetadata(
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        Objects.requireNonNull(manifest, "manifest must not be null");
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("manifestFileName", MANIFEST_FILE_NAME);
        metadata.put("resumeReportFileName", RESUME_REPORT_FILE_NAME);
        metadata.put("checkpoint", manifest.toMetadata());
        metadata.put("fingerprint", manifest.fingerprint().toMetadata());
        metadata.put("datasetAccepted", manifest.datasetAccepted());
        metadata.put("datasetGateStatus", manifest.datasetGateStatus());
        if (resumeReport != null) {
            metadata.put("resume", resumeReport.toMetadata());
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public static Map<String, Object> attachToTrainingMetadata(
            Map<String, ?> trainingMetadata,
            DiscreteTokenDatasetCheckpointManifest manifest) {
        return attachToTrainingMetadata(trainingMetadata, manifest, null);
    }

    public static Map<String, Object> attachToTrainingMetadata(
            Map<String, ?> trainingMetadata,
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (trainingMetadata != null) {
            for (Map.Entry<String, ?> entry : trainingMetadata.entrySet()) {
                String key = Objects.requireNonNull(entry.getKey(), "training metadata key must not be null");
                if (key.isBlank()) {
                    throw new IllegalArgumentException("training metadata key must not be blank");
                }
                metadata.put(key, Objects.requireNonNull(
                        entry.getValue(),
                        "training metadata field '" + key + "' must not be null"));
            }
        }
        metadata.put(TRAINING_REPORT_METADATA_KEY, trainingReportMetadata(manifest, resumeReport));
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public static TrainingSummary attachToTrainingSummary(
            TrainingSummary summary,
            DiscreteTokenDatasetCheckpointManifest manifest) {
        return attachToTrainingSummary(summary, manifest, null);
    }

    public static TrainingSummary attachToTrainingSummary(
            TrainingSummary summary,
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        Objects.requireNonNull(summary, "summary must not be null");
        return new TrainingSummary(
                summary.epochCount(),
                summary.bestValidationLoss(),
                summary.bestValidationEpoch(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss(),
                summary.durationMs(),
                attachToTrainingMetadata(summary.metadata(), manifest, resumeReport));
    }

    private static Path checkpointDir(Path checkpointDir) {
        return Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
    }
}
