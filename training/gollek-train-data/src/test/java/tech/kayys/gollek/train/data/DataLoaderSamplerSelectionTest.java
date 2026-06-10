package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class DataLoaderSamplerSelectionTest {
    @Test
    void validatesSamplerReferences() {
        IndexSampler sampler = sampler();
        BatchSampler batchSampler = batchSampler();

        assertSame(sampler, DataLoaderSamplerSelection.requireSampler(sampler));
        assertSame(batchSampler, DataLoaderSamplerSelection.requireBatchSampler(batchSampler));
        assertThrows(NullPointerException.class, () -> DataLoaderSamplerSelection.requireSampler(null));
        assertThrows(NullPointerException.class, () -> DataLoaderSamplerSelection.requireBatchSampler(null));
    }

    @Test
    void rejectsIndexSamplerAndBatchSamplerTogether() {
        IndexSampler sampler = sampler();
        BatchSampler batchSampler = batchSampler();

        DataLoaderSamplerSelection.requireExclusive(sampler, null);
        DataLoaderSamplerSelection.requireExclusive(null, batchSampler);
        DataLoaderSamplerSelection.requireExclusive(null, null);
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderSamplerSelection.requireExclusive(sampler, batchSampler));
    }

    private static IndexSampler sampler() {
        return new IndexSampler() {
            @Override
            public List<Integer> sample(int datasetSize) {
                return List.of();
            }

            @Override
            public int sampleCount(int datasetSize) {
                return 0;
            }
        };
    }

    private static BatchSampler batchSampler() {
        return new BatchSampler() {
            @Override
            public List<List<Integer>> sampleBatches(int datasetSize) {
                return List.of();
            }

            @Override
            public int sampleCount(int datasetSize) {
                return 0;
            }

            @Override
            public int batchCount(int datasetSize) {
                return 0;
            }
        };
    }
}
