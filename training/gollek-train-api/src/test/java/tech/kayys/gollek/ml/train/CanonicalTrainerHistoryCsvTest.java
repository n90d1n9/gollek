package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class CanonicalTrainerHistoryCsvTest {

    @Test
    void canonicalTrainerReportsInvalidHistoryCsvOnResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-invalid");
        Files.writeString(checkpointDir.resolve("canonical-history.csv"), "epoch,trainLoss\n\"unterminated");

        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train);

            TrainingSummary summary = trainer.summary();
            assertEquals(1, summary.epochCount());
            assertEquals(Boolean.TRUE, summary.metadata().get("checkpointResumePartial"));
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistoryLoaded"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryLoadFailed"));
            assertTrue(String.valueOf(summary.metadata().get("trainingHistoryLoadError"))
                    .contains("unterminated quoted cell"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistorySaved"));
            assertEquals(1, epochHistory(summary).size());
        }
    }

    @Test
    void canonicalTrainerReportsAmbiguousHistoryCsvHeaderOnLenientResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-duplicate-header");
        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainLoss,trainLoss\n0,1.0,2.0\n");

        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train);

            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistoryLoaded"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryLoadFailed"));
            assertTrue(String.valueOf(summary.metadata().get("trainingHistoryLoadError"))
                    .contains("duplicate column 'trainLoss'"));
            assertEquals(1, epochHistory(summary).size());
        }
    }

    @Test
    void canonicalTrainerRejectsHistoryCsvRowsWithExtraCellsInStrictResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-extra-cell");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        Linear seedModel = new Linear(1, 1);
        try (CanonicalTrainer seed = CanonicalTrainer.builder()
                .model(seedModel)
                .optimizer(SGD.builder(seedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            seed.fit(train, null);
        }

        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainLoss\n0,1.0,unexpected\n");
        Files.deleteIfExists(checkpointDir.resolve("canonical-checkpoints.metadata"));

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException error = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(error.getMessage().contains("Failed to load training history checkpoint"));
        assertNotNull(error.getCause());
        assertTrue(error.getCause().getMessage().contains("row 2"));
        assertTrue(error.getCause().getMessage().contains("cells"));
    }

    @Test
    void canonicalTrainerReportsDuplicateHistoryEpochOnLenientResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-duplicate-epoch");
        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainLoss\n0,1.0\n0,2.0\n");

        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train);

            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistoryLoaded"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryLoadFailed"));
            assertTrue(String.valueOf(summary.metadata().get("trainingHistoryLoadError"))
                    .contains("duplicate epoch 0"));
            assertEquals(1, epochHistory(summary).size());
        }
    }

    @Test
    void canonicalTrainerRejectsNonIntegerHistoryEpochInStrictResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-invalid-epoch");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        Linear seedModel = new Linear(1, 1);
        try (CanonicalTrainer seed = CanonicalTrainer.builder()
                .model(seedModel)
                .optimizer(SGD.builder(seedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            seed.fit(train, null);
        }

        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainLoss\n1.5,1.0\n");
        Files.deleteIfExists(checkpointDir.resolve("canonical-checkpoints.metadata"));

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException error = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(error.getMessage().contains("Failed to load training history checkpoint"));
        assertNotNull(error.getCause());
        assertTrue(error.getCause().getMessage().contains("row 2"));
        assertTrue(error.getCause().getMessage().contains("non-negative integer"));
    }

    @Test
    void canonicalTrainerReportsMalformedStructuredHistoryJsonOnLenientResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-json-invalid");
        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainMetrics\n0,\"{\"\"mae\"\":1.0\"\n");

        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train);

            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistoryLoaded"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryLoadFailed"));
            assertTrue(String.valueOf(summary.metadata().get("trainingHistoryLoadError"))
                    .contains("trainMetrics"));
            assertTrue(String.valueOf(summary.metadata().get("trainingHistoryLoadError"))
                    .contains("Invalid JSON"));
            assertEquals(1, epochHistory(summary).size());
        }
    }

    @Test
    void canonicalTrainerRejectsMalformedStructuredHistoryJsonInStrictResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-json-strict");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        Linear seedModel = new Linear(1, 1);
        try (CanonicalTrainer seed = CanonicalTrainer.builder()
                .model(seedModel)
                .optimizer(SGD.builder(seedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            seed.fit(train, null);
        }

        Files.writeString(
                checkpointDir.resolve("canonical-history.csv"),
                "epoch,trainMetricDetails\n0,\"{\"\"matrix_detail\"\":{\"\"matrix\"\":[[1,2]]}\"\n");
        Files.deleteIfExists(checkpointDir.resolve("canonical-checkpoints.metadata"));

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException error = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(error.getMessage().contains("Failed to load training history checkpoint"));
        assertNotNull(error.getCause());
        assertTrue(error.getCause().getMessage().contains("trainMetricDetails"));
        assertTrue(error.getCause().getMessage().contains("Invalid JSON"));
    }

    private static Batch batch(float[] inputs, float[] targets, int rows) {
        return new Batch(
                GradTensor.of(inputs, rows, 1),
                GradTensor.of(targets, rows, 1));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> epochHistory(TrainingSummary summary) {
        Object value = summary.metadata().get("epochHistory");
        assertTrue(value instanceof List<?>);
        return (List<Map<String, Object>>) value;
    }
}
