package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.ml.optim.StepLR;

class TrainerStateCheckpointPersistenceTest {
    @TempDir
    Path tempDir;

    @Test
    void persistsOptimizerCheckpointWithTrainerMetadata() throws Exception {
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f)
                .momentum(0.9f)
                .build();
        Path checkpoint = tempDir.resolve("optimizer.state");

        TrainerStateCheckpointPersistence.Result result =
                TrainerStateCheckpointPersistence.persistOptimizer(checkpoint, optimizer, 7);

        assertTrue(result.stateChanged());
        assertTrue(result.saved());
        assertNull(result.saveError());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "optimizer");
        assertEquals("SGD", state.get("optimizer"));
        assertEquals(7, state.get("trainerOptimizerStepCount"));
        assertEquals(optimizer.getClass().getName(), state.get("trainerOptimizerClass"));
        assertEquals(
                TrainerMetadataSupport.parameterSignature(optimizer.parameters()),
                state.get("trainerOptimizerParameterSignature"));
    }

    @Test
    void persistsSchedulerCheckpointWithOptimizerSignature() throws Exception {
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        StepLR scheduler = new StepLR(optimizer, 2, 0.5f);
        Path checkpoint = tempDir.resolve("scheduler.state");

        TrainerStateCheckpointPersistence.Result result =
                TrainerStateCheckpointPersistence.persistScheduler(
                        checkpoint,
                        scheduler,
                        "EPOCH",
                        3,
                        optimizer);

        assertTrue(result.stateChanged());
        assertTrue(result.saved());
        assertNull(result.saveError());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "scheduler");
        assertEquals("StepLR", state.get("scheduler"));
        assertEquals(scheduler.getClass().getName(), state.get("trainerSchedulerClass"));
        assertEquals("EPOCH", state.get("trainerSchedulerStepUnit"));
        assertEquals(3, state.get("trainerSchedulerStepCount"));
        assertEquals(
                TrainerMetadataSupport.parameterSignature(optimizer.parameters()),
                state.get("trainerSchedulerOptimizerParameterSignature"));
    }

    @Test
    void persistsGradScalerCheckpointWithOverflowSkipCount() throws Exception {
        GradScaler gradScaler = GradScaler.builder()
                .initScale(256.0)
                .growthInterval(4)
                .build();
        Path checkpoint = tempDir.resolve("grad-scaler.state");

        TrainerStateCheckpointPersistence.Result result =
                TrainerStateCheckpointPersistence.persistGradScaler(checkpoint, gradScaler, 2);

        assertTrue(result.stateChanged());
        assertTrue(result.saved());
        assertNull(result.saveError());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "GradScaler");
        assertEquals("GradScaler", state.get("scaler"));
        assertEquals(gradScaler.getClass().getName(), state.get("trainerGradScalerClass"));
        assertEquals(true, state.get("trainerMixedPrecisionEnabled"));
        assertEquals(2, state.get("trainerMixedPrecisionOverflowSkipCount"));
    }

    @Test
    void skipsWhenArtifactOrComponentIsUnavailable() {
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();

        TrainerStateCheckpointPersistence.Result noFile =
                TrainerStateCheckpointPersistence.persistOptimizer(null, optimizer, 1);
        TrainerStateCheckpointPersistence.Result noScheduler =
                TrainerStateCheckpointPersistence.persistScheduler(
                        tempDir.resolve("scheduler.state"),
                        null,
                        "EPOCH",
                        1,
                        optimizer);
        TrainerStateCheckpointPersistence.Result noScaler =
                TrainerStateCheckpointPersistence.persistGradScaler(
                        tempDir.resolve("grad-scaler.state"),
                        null,
                        1);

        assertFalse(noFile.stateChanged());
        assertFalse(noFile.saved());
        assertNull(noFile.saveError());
        assertFalse(noScheduler.stateChanged());
        assertFalse(noScaler.stateChanged());
    }
}
