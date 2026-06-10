package tech.kayys.gollek.train.data.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class DataLoaderRulesTest {
    @Test
    void validatesBatchSizesAndCounts() {
        assertEquals(8, DataLoaderBatchSizeRules.requirePositive(8));
        assertEquals(0, DataLoaderCountRules.requireDatasetSize(0));
        assertEquals(3, DataLoaderCountRules.requireSampleCount(3));
        assertEquals(2, DataLoaderCountRules.requireBatchCount(2));

        assertThrows(IllegalArgumentException.class, () -> DataLoaderBatchSizeRules.requirePositive(0));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCountRules.requireDatasetSize(-1));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCountRules.requireSampleCount(-1));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCountRules.requireBatchCount(-1));
    }

    @Test
    void advancesEpochOnlyWhenReshuffling() {
        AtomicLong counter = DataLoaderEpochRules.counter(5L);

        assertEquals(5L, DataLoaderEpochRules.next(counter, false, 5L));
        assertEquals(5L, DataLoaderEpochRules.next(counter, true, 5L));
        assertEquals(6L, DataLoaderEpochRules.next(counter, true, 5L));

        assertThrows(IllegalArgumentException.class, () -> DataLoaderEpochRules.counter(-1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderEpochRules.requireEpoch(-1L));
        assertThrows(NullPointerException.class, () -> DataLoaderEpochRules.next(null, true, 0L));
    }

    @Test
    void derivesStableShuffleSeedsForEpochs() {
        assertEquals(null, DataLoaderShuffleSeedRules.forEpoch(null, true, 1L));
        assertEquals(42L, DataLoaderShuffleSeedRules.forEpoch(42L, false, 7L));
        assertEquals(42L, DataLoaderShuffleSeedRules.forEpoch(42L, true, 0L));
        assertEquals(
                DataLoaderShuffleSeedRules.forEpoch(42L, true, 7L),
                DataLoaderShuffleSeedRules.forEpoch(42L, true, 7L));
    }

    @Test
    void validatesSharedSamplerSelectionRules() {
        Object sampler = new Object();
        Object batchSampler = new Object();

        assertEquals(sampler, DataLoaderSamplerSelectionRules.requireSampler(sampler));
        assertEquals(batchSampler, DataLoaderSamplerSelectionRules.requireBatchSampler(batchSampler));
        DataLoaderSamplerSelectionRules.requireExclusive(sampler, null);
        DataLoaderSamplerSelectionRules.requireExclusive(null, batchSampler);

        assertThrows(NullPointerException.class, () -> DataLoaderSamplerSelectionRules.requireSampler(null));
        assertThrows(NullPointerException.class, () -> DataLoaderSamplerSelectionRules.requireBatchSampler(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderSamplerSelectionRules.requireExclusive(sampler, batchSampler));
    }

    @Test
    void materializesRowsFromIndexReader() {
        List<String> values = List.of("zero", "one", "two", "three");

        assertEquals(
                List.of("three", "one", "three"),
                DataLoaderBatchMaterializationRules.materialize(values::get, List.of(3, 1, 3)));
        assertThrows(
                NullPointerException.class,
                () -> DataLoaderBatchMaterializationRules.materialize(null, List.of(0)));
        assertThrows(
                NullPointerException.class,
                () -> DataLoaderBatchMaterializationRules.materialize(values::get, null));
    }

    @Test
    void createsEpochIndicesAndFixedBatches() {
        assertEquals(List.of(0, 1, 2), DataLoaderBatchingRules.epochIndices(3, null, false, null, false, 0L));
        assertEquals(
                List.of(2, 0, 2),
                DataLoaderBatchingRules.epochIndices(5, ignored -> List.of(2, 0, 2), true, 123L, true, 7L));
        assertEquals(3, DataLoaderBatchingRules.batchCount(5, 2, false));
        assertEquals(2, DataLoaderBatchingRules.batchCount(5, 2, true));
        assertEquals(
                List.of(List.of(0, 1), List.of(2)),
                DataLoaderBatchingRules.fixedBatches(List.of(0, 1, 2), 2, false));

        assertThrows(IllegalArgumentException.class, () -> DataLoaderBatchingRules.epochIndices(-1, null, false, null, false, 0L));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderBatchingRules.batchCount(-1, 2, false));
        assertThrows(NullPointerException.class, () -> DataLoaderBatchingRules.fixedBatches(null, 2, false));
    }

    @Test
    void collatesTensorBatchThroughFunction() {
        assertEquals(
                "size=3 dataset=rows",
                DataLoaderTensorCollationRules.collate(
                        (indices, dataset) -> "size=" + indices.size() + " dataset=" + dataset,
                        List.of(0, 1, 2),
                        "rows"));

        assertThrows(
                NullPointerException.class,
                () -> DataLoaderTensorCollationRules.collate(null, List.of(0), "rows"));
        assertThrows(
                NullPointerException.class,
                () -> DataLoaderTensorCollationRules.collate((indices, dataset) -> "", null, "rows"));
        assertThrows(
                NullPointerException.class,
                () -> DataLoaderTensorCollationRules.collate((indices, dataset) -> "", List.of(0), null));
    }

    @Test
    void infersSequenceLengthsFromTensorsAndRows() {
        assertEquals(4, DataLoaderSequenceLengthRules.sequenceLength(GradTensor.zeros(4, 2)));
        assertEquals(
                List.of(2, 3),
                java.util.Arrays.stream(DataLoaderSequenceLengthRules.lengths(
                                2,
                                index -> index == 0 ? "aa" : "bbb",
                                String::length))
                        .boxed()
                        .toList());

        assertThrows(NullPointerException.class, () -> DataLoaderSequenceLengthRules.sequenceLength(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderSequenceLengthRules.sequenceLength(GradTensor.scalar(1.0f)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderSequenceLengthRules.lengths(1, index -> "bad", ignored -> -1));
        assertThrows(
                NullPointerException.class,
                () -> DataLoaderSequenceLengthRules.lengths(1, index -> null, String::length));
    }

    @Test
    void validatesTensorBatchAndPaddedShapes() {
        GradTensor inputs = GradTensor.zeros(2, 3, 1);
        GradTensor labels = GradTensor.zeros(2, 3, 1);
        GradTensor mask = GradTensor.zeros(2, 3);

        assertEquals(inputs, DataLoaderTensorBatchValidationRules.requireTensor("inputs", inputs));
        DataLoaderTensorBatchValidationRules.requireCompatibleBatchDimensions(inputs, labels);
        DataLoaderTensorBatchValidationRules.validatePadded("inputs", inputs, mask, new int[] {3, 1});
        assertEquals(3, DataLoaderTensorBatchValidationRules.paddedLength("inputs", inputs));

        assertThrows(
                NullPointerException.class,
                () -> DataLoaderTensorBatchValidationRules.requireTensor("inputs", null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderTensorBatchValidationRules.requireCompatibleBatchDimensions(
                        GradTensor.zeros(2, 1),
                        GradTensor.zeros(3, 1)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoaderTensorBatchValidationRules.validatePadded(
                        "inputs",
                        inputs,
                        mask,
                        new int[] {4, 1}));
    }

    @Test
    void computesPaddingStatsValues() {
        DataLoaderPaddingStatsRules.Values stats =
                DataLoaderPaddingStatsRules.fromLengths(4, new int[] {4, 2, 0});

        assertEquals(3, stats.batchSize());
        assertEquals(4, stats.maxLength());
        assertEquals(6L, stats.realTokens());
        assertEquals(12L, stats.paddedTokens());
        assertEquals(6L, stats.paddingTokens());
        assertEquals(0.5, stats.paddingRatio());
        assertEquals(0.5, stats.utilization());

        DataLoaderPaddingStatsRules.Values merged =
                stats.merge(new DataLoaderPaddingStatsRules.Values(1, 2, 2L, 2L));
        assertEquals(4, merged.batchSize());
        assertEquals(4, merged.maxLength());
        assertEquals(8L, merged.realTokens());
        assertEquals(14L, merged.paddedTokens());

        assertThrows(NullPointerException.class, () -> DataLoaderPaddingStatsRules.fromLengths(4, null));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderPaddingStatsRules.fromLengths(-1, new int[0]));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderPaddingStatsRules.fromLengths(4, new int[] {5}));
        assertThrows(IllegalArgumentException.class, () -> new DataLoaderPaddingStatsRules.Values(1, 1, 2L, 1L));
    }
}
