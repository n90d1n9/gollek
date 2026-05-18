package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class ByteLatentTrainerLifecycleTest {

    @Test
    void notifiesListenersAndPersistsCheckpointArtifacts() throws Exception {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab", "cde"));
        Path checkpointDir = Files.createTempDirectory("gollek-byte-latent-checkpoint");
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(1)
                .windowLength(3)
                .epochs(1)
                .shuffle(false)
                .checkpointDir(checkpointDir)
                .build();

        List<String> events = new ArrayList<>();
        ByteLatentTrainingListener listener = new ByteLatentTrainingListener() {
            @Override
            public void onTrainingStart(ByteLatentTrainerSession session) {
                events.add("training-start");
            }

            @Override
            public void onEpochStart(ByteLatentTrainerSession session, int epoch) {
                events.add("epoch-start-" + epoch);
            }

            @Override
            public void onBatchStart(ByteLatentTrainerSession session, int step) {
                events.add("batch-start-" + step);
            }

            @Override
            public void onBatchEnd(ByteLatentTrainerSession session, int step, double loss) {
                events.add("batch-end-" + step);
            }

            @Override
            public void onEpochEnd(ByteLatentTrainerSession session, int epoch, double trainLoss) {
                events.add("epoch-end-" + epoch);
            }

            @Override
            public void onTrainingEnd(ByteLatentTrainerSession session, TrainingSummary summary) {
                events.add("training-end");
            }
        };

        TrainingSummary summary = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(config)
                .listeners(List.of(listener))
                .build()
                .fit();

        assertEquals(List.of(
                "training-start",
                "epoch-start-1",
                "batch-start-0",
                "batch-end-1",
                "batch-start-1",
                "batch-end-2",
                "epoch-end-1",
                "training-end"), events);
        assertTrue(Files.isRegularFile(checkpointDir.resolve("byte-latent-summary.json")));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("byte-latent-checkpoint.metadata")));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("byte-latent-history.csv")));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("byte-latent-report.json")));
        String reportJson = Files.readString(checkpointDir.resolve("byte-latent-report.json"));
        assertTrue(reportJson.contains("\"schema\":\"gollek.byte-latent.report.v1\""));
        assertTrue(reportJson.contains("\"historyCount\":1"));
        assertEquals(checkpointDir.toString(), summary.metadata().get("checkpointDir"));
        assertEquals(checkpointDir.resolve("byte-latent-history.csv").toString(), summary.metadata().get("historyFile"));
        assertEquals(checkpointDir.resolve("byte-latent-report.json").toString(), summary.metadata().get("reportFile"));
        assertEquals(1, summary.metadata().get("historyRowCount"));
        assertEquals(1, summary.metadata().get("listenerCount"));
    }
}
