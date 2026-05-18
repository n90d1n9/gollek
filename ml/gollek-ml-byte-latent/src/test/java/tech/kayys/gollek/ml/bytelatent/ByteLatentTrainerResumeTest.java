package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class ByteLatentTrainerResumeTest {

    @Test
    void resumesFromCheckpointManifestAndContinuesEpochs() throws Exception {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab", "cde", "fghi"));
        Path checkpointDir = Files.createTempDirectory("gollek-byte-latent-resume");

        ByteLatentTrainerConfig firstConfig = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(2)
                .windowLength(3)
                .epochs(1)
                .shuffle(false)
                .checkpointDir(checkpointDir)
                .build();
        TrainingSummary first = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(firstConfig)
                .lossEvaluator((batch, cfg, epoch, batchIndex) -> 1.0d)
                .build()
                .fit();

        assertEquals(1, first.epochCount());

        ByteLatentTrainerConfig resumedConfig = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(2)
                .windowLength(3)
                .epochs(3)
                .shuffle(false)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint(true)
                .build();
        TrainingSummary resumed = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(resumedConfig)
                .lossEvaluator((batch, cfg, epoch, batchIndex) -> epoch)
                .build()
                .fit();

        assertEquals(3, resumed.epochCount());
        assertEquals(1, resumed.bestValidationEpoch());
        assertEquals(1.0d, resumed.bestValidationLoss());
        assertEquals(3.0d, resumed.latestTrainLoss());
        assertTrue((Boolean) resumed.metadata().get("resumeRequested"));
        assertTrue((Boolean) resumed.metadata().get("resumeLoaded"));
        assertTrue((Boolean) resumed.metadata().get("historyLoaded"));
        assertEquals(3, resumed.metadata().get("historyRowCount"));
        assertEquals(6, resumed.metadata().get("globalStep"));
        List<String> historyLines = Files.readAllLines(checkpointDir.resolve("byte-latent-history.csv"));
        assertEquals(4, historyLines.size());
        assertEquals("epoch,globalStep,batchCount,trainLoss", historyLines.get(0));
        assertEquals("1,2,2,1.0", historyLines.get(1));
        assertEquals("2,4,2,2.0", historyLines.get(2));
        assertEquals("3,6,2,3.0", historyLines.get(3));
    }
}
