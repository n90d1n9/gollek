package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TrainerStateCheckpointStatusTest {
    @Test
    void optimizerStatusTracksResumeAndPersistenceState() {
        TrainerStateCheckpointStatus status = new TrainerStateCheckpointStatus();
        status.recordOptimizerResume(TrainerStateCheckpointResume.Result.loaded(7));
        status.recordOptimizerPersistence(TrainerStateCheckpointPersistence.Result.failed("disk full"));

        TrainerSummaryAssembler.StateCheckpoint summary =
                status.optimizerSummary(Path.of("optimizer.json"), true);

        assertTrue(summary.enabled());
        assertTrue(summary.supported());
        assertFalse(summary.missingOnResume());
        assertTrue(summary.loaded());
        assertFalse(summary.saved());
        assertEquals("disk full", summary.saveError());
        assertNull(summary.loadError());
        assertFalse(status.optimizerMissingOnResume());
        assertEquals("disk full", status.optimizerSaveError());
    }

    @Test
    void schedulerStatusPublishesResumeRequestedFlag() {
        TrainerStateCheckpointStatus status = new TrainerStateCheckpointStatus();
        status.recordSchedulerResume(TrainerStateCheckpointResume.Result.missing(null));
        status.recordSchedulerPersistence(TrainerStateCheckpointPersistence.Result.savedResult());

        TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint summary =
                status.schedulerSummary(Path.of("scheduler.json"), true, true);

        assertTrue(summary.enabled());
        assertTrue(summary.supported());
        assertTrue(summary.resumeRequested());
        assertTrue(summary.missingOnResume());
        assertFalse(summary.loaded());
        assertTrue(summary.saved());
        assertTrue(status.schedulerMissingOnResume());
    }

    @Test
    void gradScalerFallbackIsSetOnlyForRecoverableLoadErrors() {
        TrainerStateCheckpointStatus status = new TrainerStateCheckpointStatus();
        status.recordGradScalerResume(TrainerStateCheckpointResume.Result.failed("bad metadata", null));

        TrainerMixedPrecisionMetadata.GradScalerCheckpoint failed =
                status.gradScalerSummary(Path.of("grad-scaler.json"), true, true, true);

        assertFalse(failed.loaded());
        assertTrue(failed.fallbackUsed());
        assertEquals("bad metadata", failed.loadError());

        status.recordGradScalerResume(TrainerStateCheckpointResume.Result.loaded(3));
        TrainerMixedPrecisionMetadata.GradScalerCheckpoint loaded =
                status.gradScalerSummary(Path.of("grad-scaler.json"), true, true, true);

        assertTrue(loaded.loaded());
        assertFalse(loaded.fallbackUsed());
    }
}
