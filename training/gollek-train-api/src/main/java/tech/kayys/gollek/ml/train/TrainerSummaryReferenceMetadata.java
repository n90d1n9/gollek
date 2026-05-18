package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Publishes optional file references and error messages for trainer summaries.
 */
final class TrainerSummaryReferenceMetadata {
    private TrainerSummaryReferenceMetadata() {
    }

    static void put(Map<String, Object> metadata, Paths paths, Errors errors) {
        putPaths(metadata, paths);
        putErrors(metadata, errors);
    }

    private static void putPaths(Map<String, Object> metadata, Paths paths) {
        TrainerSummaryMetadata.putOptionalPath(metadata, "modelCheckpointFile", paths.modelCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(metadata, "modelCheckpointMetadataFile", paths.modelCheckpointMetadata());
        TrainerSummaryMetadata.putOptionalPath(metadata, "bestModelCheckpointFile", paths.bestModelCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(metadata, "optimizerCheckpointFile", paths.optimizerCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(metadata, "schedulerCheckpointFile", paths.schedulerCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(
                metadata,
                "gradScalerCheckpointFile",
                paths.gradScalerCheckpoint(),
                paths.includeGradScalerCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(metadata, "trainingHistoryFile", paths.history());
        TrainerSummaryMetadata.putOptionalPath(metadata, "trainingReportFile", paths.report());
        TrainerSummaryMetadata.putOptionalPath(metadata, "runtimeCheckpointFile", paths.runtimeCheckpoint());
        TrainerSummaryMetadata.putOptionalPath(metadata, "checkpointManifestFile", paths.checkpointManifest());
    }

    private static void putErrors(Map<String, Object> metadata, Errors errors) {
        TrainerSummaryMetadata.putOptionalError(metadata, "modelCheckpointLoadError", errors.modelCheckpointLoad());
        TrainerSummaryMetadata.putOptionalError(metadata, "modelCheckpointSaveError", errors.modelCheckpointSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "modelCheckpointMetadataLoadError",
                errors.modelCheckpointMetadataLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "modelCheckpointMetadataSaveError",
                errors.modelCheckpointMetadataSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "bestModelCheckpointSaveError",
                errors.bestModelCheckpointSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "bestModelCheckpointLoadError",
                errors.bestModelCheckpointLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "optimizerCheckpointLoadError",
                errors.optimizerCheckpointLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "optimizerCheckpointSaveError",
                errors.optimizerCheckpointSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "schedulerCheckpointLoadError",
                errors.schedulerCheckpointLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "schedulerCheckpointSaveError",
                errors.schedulerCheckpointSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "gradScalerCheckpointLoadError",
                errors.gradScalerCheckpointLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "gradScalerCheckpointSaveError",
                errors.gradScalerCheckpointSave());
        TrainerSummaryMetadata.putOptionalError(metadata, "trainingHistoryLoadError", errors.historyLoad());
        TrainerSummaryMetadata.putOptionalError(metadata, "trainingHistorySaveError", errors.historySave());
        TrainerSummaryMetadata.putOptionalError(metadata, "trainingReportSaveError", errors.reportSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "checkpointManifestLoadError",
                errors.checkpointManifestLoad());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "checkpointManifestSaveError",
                errors.checkpointManifestSave());
        TrainerSummaryMetadata.putOptionalError(
                metadata,
                "runtimeCheckpointLoadError",
                errors.runtimeCheckpointLoad());
    }

    record Paths(
            Path modelCheckpoint,
            Path modelCheckpointMetadata,
            Path bestModelCheckpoint,
            Path optimizerCheckpoint,
            Path schedulerCheckpoint,
            Path gradScalerCheckpoint,
            boolean includeGradScalerCheckpoint,
            Path history,
            Path report,
            Path runtimeCheckpoint,
            Path checkpointManifest) {
    }

    record Errors(
            String modelCheckpointLoad,
            String modelCheckpointSave,
            String modelCheckpointMetadataLoad,
            String modelCheckpointMetadataSave,
            String bestModelCheckpointSave,
            String bestModelCheckpointLoad,
            String optimizerCheckpointLoad,
            String optimizerCheckpointSave,
            String schedulerCheckpointLoad,
            String schedulerCheckpointSave,
            String gradScalerCheckpointLoad,
            String gradScalerCheckpointSave,
            String historyLoad,
            String historySave,
            String reportSave,
            String checkpointManifestLoad,
            String checkpointManifestSave,
            String runtimeCheckpointLoad) {
    }
}
