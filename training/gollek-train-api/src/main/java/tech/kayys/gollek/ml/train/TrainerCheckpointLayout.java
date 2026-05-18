package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

record TrainerCheckpointLayout(
        Path model,
        Path modelMetadata,
        Path bestModel,
        Path optimizer,
        Path scheduler,
        Path gradScaler,
        Path history,
        Path report,
        Path runtime,
        Path manifest) {
    static final String MODEL_FILE_NAME = "canonical-model.safetensors";
    static final String MODEL_METADATA_FILE_NAME = "canonical-model.metadata";
    static final String BEST_MODEL_FILE_NAME = "canonical-best-model.safetensors";
    static final String OPTIMIZER_FILE_NAME = "canonical-optimizer.state";
    static final String SCHEDULER_FILE_NAME = "canonical-scheduler.state";
    static final String GRAD_SCALER_FILE_NAME = "canonical-grad-scaler.state";
    static final String HISTORY_FILE_NAME = "canonical-history.csv";
    static final String REPORT_FILE_NAME = "canonical-report.json";
    static final String RUNTIME_FILE_NAME = "canonical-runtime.state";
    static final String MANIFEST_FILE_NAME = "canonical-checkpoints.metadata";

    static TrainerCheckpointLayout from(Path checkpointDir) {
        return new TrainerCheckpointLayout(
                resolve(checkpointDir, MODEL_FILE_NAME),
                resolve(checkpointDir, MODEL_METADATA_FILE_NAME),
                resolve(checkpointDir, BEST_MODEL_FILE_NAME),
                resolve(checkpointDir, OPTIMIZER_FILE_NAME),
                resolve(checkpointDir, SCHEDULER_FILE_NAME),
                resolve(checkpointDir, GRAD_SCALER_FILE_NAME),
                resolve(checkpointDir, HISTORY_FILE_NAME),
                resolve(checkpointDir, REPORT_FILE_NAME),
                resolve(checkpointDir, RUNTIME_FILE_NAME),
                resolve(checkpointDir, MANIFEST_FILE_NAME));
    }

    Map<String, Path> manifestArtifacts() {
        Map<String, Path> artifacts = new LinkedHashMap<>();
        artifacts.put("runtime", runtime);
        artifacts.put("model", model);
        artifacts.put("modelMetadata", modelMetadata);
        artifacts.put("bestModel", bestModel);
        artifacts.put("optimizer", optimizer);
        artifacts.put("scheduler", scheduler);
        artifacts.put("gradScaler", gradScaler);
        artifacts.put("history", history);
        artifacts.put("report", report);
        return artifacts;
    }

    private static Path resolve(Path checkpointDir, String fileName) {
        return checkpointDir == null ? null : checkpointDir.resolve(fileName);
    }
}
