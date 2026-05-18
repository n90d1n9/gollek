package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerSummaryAssemblerTest {

    @TempDir
    Path tempDir;

    @Test
    void enrichPublishesTrainerSummarySnapshotAcrossRuntimeDomains() throws Exception {
        TrainingSummary base = new TrainingSummary(
                3,
                0.12,
                2,
                0.34,
                0.23,
                987L,
                Map.of("seed", 42));
        TrainerFailureState failureState = new TrainerFailureState();

        TrainingSummary summary = TrainerSummaryAssembler.enrich(base, request(tempDir, failureState));

        Map<String, Object> metadata = summary.metadata();
        assertEquals(3, summary.epochCount());
        assertEquals(42, metadata.get("seed"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
        assertEquals(List.of("scheduler"), metadata.get("checkpointResumeMissingArtifacts"));
        assertEquals(Boolean.TRUE, metadata.get("modelCheckpointLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("modelCheckpointPresent"));
        assertEquals(Boolean.FALSE, metadata.get("modelCheckpointCompatibilityMismatch"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointSupported"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointSaved"));
        assertEquals(Boolean.TRUE, metadata.get("mixedPrecisionEnabled"));
        assertEquals(32.0, metadata.get("mixedPrecisionLossScale"));
        assertEquals(32.0, metadata.get("mixedPrecisionGradScalerState.scale"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointSupported"));
        assertEquals(5, metadata.get("learningRateSchedulerStepCount"));
        assertEquals("accuracy", metadata.get("learningRateSchedulerMonitor"));
        assertEquals(0.001, metadata.get("learningRate"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistorySaved"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestIntegrityMismatch"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointResumeSkipped"));
        assertEquals(Boolean.TRUE, metadata.get("earlyStoppingEnabled"));
        assertEquals("validationLoss", metadata.get("earlyStoppingMonitor"));
        assertEquals(Boolean.TRUE, metadata.get("metricsEnabled"));
        assertEquals(0.91, metadata.get("trainMetric.accuracy"));
        assertEquals("steady", metadata.get("validationMetricDetails.comment"));
        assertEquals(1, metadata.get("epochHistorySize"));
        assertEquals(4, metadata.get("gradientAccumulationSteps"));
        assertEquals(2, metadata.get("pendingGradientAccumulationBatches"));
        assertEquals(7, metadata.get("optimizerStepCount"));
        assertEquals(Boolean.TRUE, metadata.get("gradientClipEnabled"));
        assertEquals(4.0, metadata.get("latestGradientL2NormBeforeClip"));
        assertEquals(8.0, metadata.get("latestParameterL2Norm"));
        assertEquals(2L, metadata.get("trainBatchCount"));
        assertEquals(5.0, metadata.get("trainSamplesPerSecond"));
        assertEquals("metal", metadata.get("requestedDevice"));
        assertEquals("metal", metadata.get("executionBackend"));
        assertEquals(3L, metadata.get("acceleratedMatmulCallsDelta"));
        assertTrue(((String) metadata.get("modelCheckpointFile")).endsWith("model.safetensors"));
        assertEquals(Boolean.FALSE, metadata.get("nonFiniteDetected"));

        assertThrows(UnsupportedOperationException.class, () -> summary.metadata().put("late", true));
    }

    @Test
    void failureStateCanPublishTerminalStopReasonThroughAssembler() throws Exception {
        TrainerFailureState failureState = new TrainerFailureState();
        failureState.recordNonFinite("train", "gradient", Double.NaN, "gradient", true);

        TrainingSummary summary = TrainerSummaryAssembler.enrich(
                new TrainingSummary(1, Double.NaN, -1, 1.0, null, 10L, Map.of()),
                request(tempDir, failureState));

        Map<String, Object> metadata = summary.metadata();
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteDetected"));
        assertEquals("gradient", metadata.get("nonFiniteKind"));
        assertEquals("non-finite-train-gradient", metadata.get("stopReason"));
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteOptimizerStepSkipped"));
    }

    private static TrainerSummaryAssembler.Request request(
            Path tempDir,
            TrainerFailureState failureState) throws Exception {
        Path model = Files.writeString(tempDir.resolve("model.safetensors"), "model");
        Path metadata = Files.writeString(tempDir.resolve("model.properties"), "metadata");
        Path best = Files.writeString(tempDir.resolve("best.safetensors"), "best");
        Path optimizer = Files.writeString(tempDir.resolve("optimizer.json"), "{}");
        Path scheduler = Files.writeString(tempDir.resolve("scheduler.json"), "{}");
        Path gradScaler = Files.writeString(tempDir.resolve("grad-scaler.json"), "{}");
        Path history = Files.writeString(tempDir.resolve("history.csv"), "epoch");
        Path report = Files.writeString(tempDir.resolve("report.json"), "{}");
        Path manifest = Files.writeString(tempDir.resolve("manifest.properties"), "manifest");
        Path runtime = Files.writeString(tempDir.resolve("runtime.json"), "{}");

        return new TrainerSummaryAssembler.Request(
                new TrainerSummaryAssembler.Resume(
                        true,
                        List.of("scheduler"),
                        List.of("optimizer: parameter signature mismatch")),
                new TrainerSummaryAssembler.ModelCheckpoint(model, false, true, true, null, null, false),
                new TrainerSummaryAssembler.Artifact(metadata, false, true, true, null, null),
                new TrainerSummaryAssembler.BestModel(
                        true,
                        true,
                        best,
                        "accuracy",
                        CanonicalTrainer.BestModelMonitorMode.MAX,
                        new TrainerBestModelCheckpointMetadata.State(true, false, 2, 0.23, 0.92),
                        null,
                        null),
                new TrainerSummaryAssembler.StateCheckpoint(
                        true,
                        true,
                        optimizer,
                        false,
                        true,
                        true,
                        null,
                        null),
                new TrainerSummaryAssembler.MixedPrecision(
                        true,
                        32.0,
                        false,
                        1,
                        Map.of("scale", 32.0),
                        new TrainerMixedPrecisionMetadata.GradScalerCheckpoint(
                                true,
                                true,
                                true,
                                gradScaler,
                                false,
                                true,
                                true,
                                null,
                                null,
                                false)),
                new TrainerSummaryAssembler.Scheduler(
                        true,
                        CanonicalTrainer.SchedulerStepUnit.VALIDATION.name(),
                        5,
                        "cosine",
                        "accuracy",
                        true,
                        Map.of("lastEpoch", 5),
                        0.001,
                        new TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint(
                                true,
                                true,
                                true,
                                scheduler,
                                true,
                                false,
                                true,
                                "missing on resume",
                                null)),
                new TrainerSummaryAssembler.RuntimeArtifacts(
                        new TrainerRuntimeArtifactMetadata.Artifact(
                                true,
                                history,
                                false,
                                true,
                                true,
                                null,
                                null),
                        new TrainerRuntimeArtifactMetadata.SaveOnlyArtifact(
                                true,
                                report,
                                true,
                                null),
                        new TrainerRuntimeArtifactMetadata.Artifact(
                                true,
                                manifest,
                                false,
                                true,
                                true,
                                null,
                                null),
                        true,
                        new TrainerRuntimeArtifactMetadata.RuntimeCheckpoint(
                                runtime,
                                true,
                                true,
                                "runtime mismatch")),
                new TrainerSummaryAssembler.EarlyStopping(
                        3,
                        0.01,
                        "validationLoss",
                        CanonicalTrainer.BestModelMonitorMode.MIN,
                        false,
                        new TrainerEarlyStoppingMetadata.MonitorState(2, 0.12, 0.23, 1),
                        false,
                        -1),
                failureState,
                new TrainerSummaryAssembler.Metrics(
                        true,
                        Map.of("accuracy", 0.91),
                        Map.of("accuracy", 0.89),
                        Map.of("comment", "warm"),
                        Map.of("comment", "steady")),
                new TrainerSummaryAssembler.History(
                        List.of(Map.of("epoch", 1, "trainLoss", 0.5)),
                        1),
                new TrainerSummaryAssembler.Optimization(
                        4,
                        2,
                        7,
                        1.5,
                        new TrainerOptimizationMetadata.GradientDiagnostics(
                                4.0,
                                1.0,
                                3.0,
                                0.75,
                                2,
                                10,
                                true),
                        new TrainerOptimizationMetadata.ParameterDiagnostics(
                                8.0,
                                6.0,
                                2,
                                10)),
                new TrainerSummaryAssembler.Throughput(
                        new ThroughputSnapshot(2, 10, 20, 5, 2_000_000_000L),
                        new ThroughputSnapshot(1, 4, 8, 2, 1_000_000_000L)),
                new TrainerSummaryAssembler.AccelerationInfo(
                        "metal",
                        new Acceleration.BackendStatus("metal", "Apple GPU", true, true, 8L),
                        new Acceleration.BackendStatus("metal", "Apple GPU", true, true, 5L)),
                new TrainerSummaryAssembler.References(
                        new TrainerSummaryReferenceMetadata.Paths(
                                model,
                                metadata,
                                best,
                                optimizer,
                                scheduler,
                                gradScaler,
                                true,
                                history,
                                report,
                                runtime,
                                manifest),
                        new TrainerSummaryReferenceMetadata.Errors(
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                "missing on resume",
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                null,
                                "runtime mismatch")));
    }
}
