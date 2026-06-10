package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class TensorBatchingTest {
    @Test
    void fixedBatchesPreserveRemaindersUnlessDropped() {
        List<Integer> indices = List.of(0, 1, 2, 3, 4);

        assertEquals(
                List.of(List.of(0, 1), List.of(2, 3), List.of(4)),
                TensorBatching.fixedBatches(indices, 2, false));
        assertEquals(
                List.of(List.of(0, 1), List.of(2, 3)),
                TensorBatching.fixedBatches(indices, 2, true));
        assertThrows(NullPointerException.class, () -> TensorBatching.fixedBatches(null, 2, false));
    }

    @Test
    void batchCountRejectsInvalidInputs() {
        assertEquals(3, TensorBatching.batchCount(5, 2, false));
        assertEquals(2, TensorBatching.batchCount(5, 2, true));
        assertThrows(IllegalArgumentException.class, () -> TensorBatching.batchCount(-1, 2, false));
        assertThrows(IllegalArgumentException.class, () -> TensorBatching.batchCount(5, 0, false));
    }

    @Test
    void epochIndicesAreDeterministicForSeededShuffle() {
        DataLoader.TensorDatasetAdapter dataset = sizedDataset(8);
        List<Integer> expectedEpoch0 = shuffledRange(8, 123L);
        long epoch1Seed = DataLoaderShuffleSeeds.forEpoch(123L, true, 1L);
        List<Integer> expectedEpoch1 = shuffledRange(8, epoch1Seed);

        assertNull(DataLoaderShuffleSeeds.forEpoch(null, true, 1L));
        assertEquals(123L, DataLoaderShuffleSeeds.forEpoch(123L, true, 0L));
        assertEquals(expectedEpoch0, TensorBatching.epochIndices(dataset, null, true, 123L, true, 0L));
        assertEquals(expectedEpoch1, TensorBatching.epochIndices(dataset, null, true, 123L, true, 1L));
        assertNotEquals(
                TensorBatching.epochIndices(dataset, null, true, 123L, true, 0L),
                TensorBatching.epochIndices(dataset, null, true, 123L, true, 1L));
        assertThrows(NullPointerException.class, () -> TensorBatching.epochIndices(null, null, false, null, false, 0L));
        assertThrows(
                IllegalArgumentException.class,
                () -> TensorBatching.epochIndices(-1, null, false, null, false, 0L));
        assertThrows(
                IllegalArgumentException.class,
                () -> TensorBatching.epochIndices(sizedDataset(-1), null, false, null, false, 0L));
    }

    @Test
    void samplerOrderBypassesShuffle() {
        IndexSampler sampler = DataLoader.subsetSampler(3, 1, 3);

        assertEquals(List.of(3, 1, 3), TensorBatching.epochIndices(sizedDataset(5), sampler, true, 123L, true, 7L));
    }

    private static DataLoader.TensorDatasetAdapter sizedDataset(int size) {
        return new DataLoader.TensorDatasetAdapter() {
            @Override
            public int size() {
                return size;
            }

            @Override
            public GradTensor[] get(int index) {
                throw new UnsupportedOperationException("TensorBatching only needs dataset size");
            }
        };
    }

    private static List<Integer> shuffledRange(int size, long seed) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(seed));
        return indices;
    }
}
