package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class ByteLatentTrainerTest {

    @Test
    void runsTrainerSessionAndProducesSummary() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab", "cde", "fghi"));
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(2)
                .windowLength(3)
                .epochs(2)
                .shuffle(false)
                .build();

        ByteLatentTrainerSession session = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(config)
                .lossEvaluator((batch, ignoredConfig, epoch, batchIndex) -> epoch + (batchIndex * 0.5d))
                .build();

        TrainingSummary summary = session.fit();

        assertEquals(2, summary.epochCount());
        assertEquals(1.25d, summary.bestValidationLoss());
        assertEquals(1, summary.bestValidationEpoch());
        assertEquals(2.25d, summary.latestTrainLoss());
        assertEquals(4, session.globalStep());
        assertEquals(ByteLatentModelFamily.FAMILY_ID, summary.metadata().get("familyId"));
        assertEquals(3, summary.metadata().get("datasetSize"));
        assertFalse(session.isStopped());
    }

    @Test
    void stopMarksSessionStopped() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab"));
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .build();

        ByteLatentTrainerSession session = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(config)
                .build();

        session.stop();

        assertTrue(session.isStopped());
    }

    @Test
    void usesReferenceModelForwardPassByDefault() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab", "cd"));
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 32, 2, 2, 16))
                .batchSize(2)
                .windowLength(2)
                .epochs(1)
                .shuffle(false)
                .build();

        ByteLatentTrainerSession session = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(config)
                .build();

        TrainingSummary summary = session.fit();

        assertEquals(1, summary.epochCount());
        assertEquals("model-forward", summary.metadata().get("lossMode"));
        assertEquals("ReferenceByteLatentModel", summary.metadata().get("modelClass"));
        assertEquals(ByteLatentModelFamily.FAMILY_ID, summary.metadata().get("modelFamilyId"));
        assertEquals(256, summary.metadata().get("modelVocabularySize"));
    }
}
