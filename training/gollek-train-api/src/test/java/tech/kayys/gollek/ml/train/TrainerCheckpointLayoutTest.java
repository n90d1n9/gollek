package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class TrainerCheckpointLayoutTest {

    @Test
    void fromNullCheckpointDirectoryDisablesAllCheckpointPaths() {
        TrainerCheckpointLayout layout = TrainerCheckpointLayout.from(null);

        assertNull(layout.model());
        assertNull(layout.modelMetadata());
        assertNull(layout.bestModel());
        assertNull(layout.optimizer());
        assertNull(layout.scheduler());
        assertNull(layout.gradScaler());
        assertNull(layout.history());
        assertNull(layout.report());
        assertNull(layout.runtime());
        assertNull(layout.manifest());
    }

    @Test
    void fromCheckpointDirectoryResolvesCanonicalFileNames() {
        Path checkpointDir = Path.of("/tmp/gollek-checkpoint-layout");
        TrainerCheckpointLayout layout = TrainerCheckpointLayout.from(checkpointDir);

        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.MODEL_FILE_NAME), layout.model());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.MODEL_METADATA_FILE_NAME), layout.modelMetadata());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.BEST_MODEL_FILE_NAME), layout.bestModel());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.OPTIMIZER_FILE_NAME), layout.optimizer());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.SCHEDULER_FILE_NAME), layout.scheduler());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.GRAD_SCALER_FILE_NAME), layout.gradScaler());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.HISTORY_FILE_NAME), layout.history());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.REPORT_FILE_NAME), layout.report());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.RUNTIME_FILE_NAME), layout.runtime());
        assertEquals(checkpointDir.resolve(TrainerCheckpointLayout.MANIFEST_FILE_NAME), layout.manifest());
    }

    @Test
    void manifestArtifactsKeepStableNamesAndOrder() {
        TrainerCheckpointLayout layout = TrainerCheckpointLayout.from(Path.of("/tmp/gollek-checkpoint-layout"));

        assertEquals(
                List.of(
                        "runtime",
                        "model",
                        "modelMetadata",
                        "bestModel",
                        "optimizer",
                        "scheduler",
                        "gradScaler",
                        "history",
                        "report"),
                List.copyOf(layout.manifestArtifacts().keySet()));
    }
}
