package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TrainerModelCheckpointStatusTest {
    @Test
    void recordsModelResumeAndMetadataState() {
        TrainerModelCheckpointStatus status = new TrainerModelCheckpointStatus();

        status.recordResume(TrainerModelCheckpointResume.Result.loaded(true, false, null));

        TrainerSummaryAssembler.ModelCheckpoint model = status.modelSummary(Path.of("model.safetensors"));
        TrainerSummaryAssembler.Artifact metadata = status.metadataSummary(Path.of("model.metadata.json"));

        assertFalse(model.missingOnResume());
        assertTrue(model.loaded());
        assertFalse(model.saved());
        assertFalse(model.compatibilityMismatch());
        assertTrue(metadata.loaded());
        assertFalse(metadata.missingOnResume());
        assertNull(metadata.loadError());
    }

    @Test
    void recordsModelAndMetadataPersistenceErrorsIndependently() {
        TrainerModelCheckpointStatus status = new TrainerModelCheckpointStatus();
        status.recordModelPersistence(TrainerModelCheckpointWriter.WriteResult.failed(
                new IllegalStateException("model write failed")));
        status.recordMetadataPersistence(TrainerModelCheckpointWriter.WriteResult.failed(
                new IllegalStateException("metadata write failed")));

        TrainerSummaryAssembler.ModelCheckpoint model = status.modelSummary(Path.of("model.safetensors"));
        TrainerSummaryAssembler.Artifact metadata = status.metadataSummary(Path.of("model.metadata.json"));

        assertFalse(model.saved());
        assertEquals("model write failed", model.saveError());
        assertFalse(metadata.saved());
        assertEquals("metadata write failed", metadata.saveError());

        status.recordModelPersistence(TrainerModelCheckpointWriter.WriteResult.success());
        status.recordMetadataPersistence(TrainerModelCheckpointWriter.WriteResult.success());

        assertTrue(status.modelSummary(Path.of("model.safetensors")).saved());
        assertNull(status.modelSummary(Path.of("model.safetensors")).saveError());
        assertTrue(status.metadataSummary(Path.of("model.metadata.json")).saved());
        assertNull(status.metadataSummary(Path.of("model.metadata.json")).saveError());
    }

    @Test
    void compatibilityMismatchIsPublishedForSummaryAndMissingArtifacts() {
        TrainerModelCheckpointStatus status = new TrainerModelCheckpointStatus();
        status.recordResume(TrainerModelCheckpointResume.Result.compatibilityMismatch(
                false,
                true,
                "metadata missing",
                "shape mismatch",
                null));

        TrainerSummaryAssembler.ModelCheckpoint model = status.modelSummary(Path.of("model.safetensors"));
        TrainerSummaryAssembler.Artifact metadata = status.metadataSummary(Path.of("model.metadata.json"));

        assertFalse(status.missingOnResume());
        assertFalse(model.loaded());
        assertTrue(model.compatibilityMismatch());
        assertEquals("shape mismatch", model.loadError());
        assertTrue(metadata.missingOnResume());
        assertEquals("metadata missing", metadata.loadError());
    }
}
