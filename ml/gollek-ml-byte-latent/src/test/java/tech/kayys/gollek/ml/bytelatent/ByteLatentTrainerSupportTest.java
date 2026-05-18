package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteLatentTrainerSupportTest {

    @Test
    void buildsTrainerReadyPlanWithoutShuffle() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("ab", "cde", "fg"));
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(2)
                .windowLength(3)
                .padTokenId(0)
                .shuffle(false)
                .build();

        ByteLatentTrainingPlan plan = ByteLatentTrainerSupport.plan(dataset, config);

        assertEquals(3, plan.exampleCount());
        assertEquals(2, plan.batchCount());
        assertEquals(2, plan.batches().size());

        assertArrayEquals(new int[] {97, 0, 0}, plan.batches().getFirst().inputIds()[0]);
        assertArrayEquals(new int[] {98, 0, 0}, plan.batches().getFirst().targetIds()[0]);
        assertArrayEquals(new int[] {99, 100, 0}, plan.batches().getFirst().inputIds()[1]);
        assertArrayEquals(new int[] {100, 101, 0}, plan.batches().getFirst().targetIds()[1]);
    }

    @Test
    void usesDeterministicShuffleWhenEnabled() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("aa", "bb", "cc"));
        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 64, 2, 4, 32))
                .batchSize(3)
                .windowLength(2)
                .padTokenId(0)
                .shuffle(true)
                .seed(7L)
                .build();

        ByteLatentTrainingPlan left = ByteLatentTrainerSupport.plan(dataset, config);
        ByteLatentTrainingPlan right = ByteLatentTrainerSupport.plan(dataset, config);

        assertArrayEquals(left.batches().getFirst().inputIds()[0], right.batches().getFirst().inputIds()[0]);
        assertArrayEquals(left.batches().getFirst().inputIds()[1], right.batches().getFirst().inputIds()[1]);
        assertArrayEquals(left.batches().getFirst().inputIds()[2], right.batches().getFirst().inputIds()[2]);
    }
}
