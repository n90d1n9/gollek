package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class DataLoaderTest {

    @Test
    public void testDataLoaderBatching() {
        var inputs = GradTensor.randn(10, 4);
        var targets = GradTensor.randn(10, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);
        var loader = DataLoader.tensorBuilder(dataset)
            .batchSize(3)
            .shuffle(false)
            .dropLast(false)
            .build();

        int batchCount = 0;
        int totalSamples = 0;
        for (DataLoader.Batch batch : loader) {
            int bs = (int) batch.inputs().shape()[0];
            assertTrue(bs <= 3);
            totalSamples += bs;
            batchCount++;
        }
        assertEquals(4, batchCount); // ceil(10/3) = 4
        assertEquals(10, totalSamples);
    }

    @Test
    public void testDataLoaderDropLast() {
        var inputs = GradTensor.randn(10, 4);
        var targets = GradTensor.randn(10, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);
        var loader = DataLoader.tensorBuilder(dataset)
            .batchSize(3)
            .shuffle(false)
            .dropLast(true)
            .build();

        int batchCount = 0;
        for (DataLoader.Batch batch : loader) {
            assertEquals(3, batch.inputs().shape()[0]);
            batchCount++;
        }
        assertEquals(3, batchCount); // 10 / 3 = 3 full batches
    }

    @Test
    public void testDataLoaderNumBatches() {
        var inputs = GradTensor.randn(20, 4);
        var targets = GradTensor.randn(20, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);

        var loader1 = DataLoader.tensorBuilder(dataset).batchSize(5).build();
        assertEquals(4, loader1.numBatches());

        var loader2 = DataLoader.tensorBuilder(dataset).batchSize(7).build();
        assertEquals(3, loader2.numBatches());

        var loader3 = DataLoader.tensorBuilder(dataset).batchSize(7).dropLast(true).build();
        assertEquals(2, loader3.numBatches());
    }

    @Test
    public void testGenericDataLoaderSupportsSamplers() {
        Dataset<Integer> dataset = Dataset.of(0, 1, 2, 3, 4, 5);
        var subset = new DataLoader<>(dataset, 2, true, false, DataLoader.subsetSampler(4, 1, 4));

        assertEquals(6, subset.size());
        assertEquals(2, subset.batchSize());
        assertEquals(3, subset.sampleCount());
        assertEquals(2, subset.numBatches());
        assertTrue(subset.sampled());
        assertTrue(subset.shuffle());
        assertEquals(List.of(4, 1, 4), flattenGeneric(subset));

        DataLoader<String> mapped = subset.map(value -> "v" + value);
        assertEquals(3, mapped.sampleCount());
        assertTrue(mapped.sampled());
        assertEquals(List.of("v4", "v1", "v4"), flattenGeneric(mapped));

        var random = new DataLoader<>(dataset, 3, false, false, DataLoader.randomSampler(4, true, 7L));
        List<Integer> expectedRandom = DataLoader.randomSampler(4, true, 7L).sample(dataset.size()).stream()
                .map(dataset::get)
                .toList();
        assertEquals(expectedRandom, flattenGeneric(random));

        var invalidSubset = new DataLoader<>(dataset, 2, false, false, DataLoader.subsetSampler(6));
        assertThrows(IllegalArgumentException.class, invalidSubset::numBatches);
    }

    @Test
    public void testGenericDataLoaderCollatesCustomBatches() {
        record WordBatch(List<String> words, int characterCount) {}

        Dataset<String> dataset = Dataset.of("alpha", "b", "gamma", "delta");
        var sampled = new DataLoader<>(dataset, 2, true, false, DataLoader.subsetSampler(2, 0, 2));
        DataLoader.CollatingDataLoader<String, WordBatch> loader = sampled.collate(batch -> new WordBatch(
                List.copyOf(batch),
                batch.stream().mapToInt(String::length).sum()));

        List<WordBatch> batches = new ArrayList<>();
        for (WordBatch batch : loader) {
            batches.add(batch);
        }

        assertEquals(4, loader.size());
        assertEquals(3, loader.sampleCount());
        assertEquals(2, loader.numBatches());
        assertEquals(2, loader.batchSize());
        assertTrue(loader.sampled());
        assertTrue(loader.shuffle());
        assertEquals(List.of("gamma", "alpha"), batches.get(0).words());
        assertEquals(10, batches.get(0).characterCount());
        assertEquals(List.of("gamma"), batches.get(1).words());
        assertEquals(5, batches.get(1).characterCount());

        DataLoader<Integer> base = new DataLoader<>(Dataset.of(1, 2, 3, 4), 3);
        DataLoader.CollatingDataLoader<Integer, Integer> sums =
                base.collate(batch -> batch.stream().mapToInt(Integer::intValue).sum());
        List<Integer> collected = new ArrayList<>();
        for (Integer sum : sums) {
            collected.add(sum);
        }
        assertEquals(List.of(6, 4), collected);
        assertThrows(NullPointerException.class, () -> base.collate(null));
    }

    @Test
    public void testGenericTensorCollatorsBuildTrainingBatches() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f}, 2), GradTensor.of(new float[] {0f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {3f, 4f}, 2), GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f, 6f}, 2), GradTensor.of(new float[] {0f}, 1)));

        DataLoader.CollatingDataLoader<Dataset.Sample, DataLoader.Batch> loader =
                new DataLoader<>(samples, 2).collate(DataLoader.sampleBatchCollate());

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 2}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {2, 1}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, batches.get(0).inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {0f, 1f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 2}, batches.get(1).inputs().shape());

        Dataset<Dataset.Pair<GradTensor, GradTensor>> pairs = Dataset.of(
                new Dataset.Pair<>(GradTensor.of(new float[] {7f, 8f}, 2), GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Pair<>(GradTensor.of(new float[] {9f, 10f}, 2), GradTensor.of(new float[] {0f}, 1)));
        DataLoader.Batch pairBatch = new DataLoader<>(pairs, 2)
                .collate(DataLoader.tensorPairBatchCollate())
                .iterator()
                .next();

        assertArrayEquals(new float[] {7f, 8f, 9f, 10f}, pairBatch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 0f}, pairBatch.labels().data(), 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.sampleBatchCollate().apply(List.of()));
    }

    @Test
    public void testPaddedTensorCollatorsBatchVariableLengthSamples() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3),
                        GradTensor.of(new float[] {10f, 11f}, 2)),
                new Dataset.Sample(
                        GradTensor.of(new float[] {4f}, 1),
                        GradTensor.of(new float[] {12f, 13f, 14f}, 3)));

        DataLoader.PaddedBatch batch = new DataLoader<>(samples, 2)
                .collate(DataLoader.paddedSampleBatchCollate(-1f, -9f))
                .iterator()
                .next();

        assertArrayEquals(new long[] {2, 3}, batch.inputs().shape());
        assertArrayEquals(new long[] {2, 3}, batch.labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f, -1f, -1f}, batch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {10f, 11f, -9f, 12f, 13f, 14f}, batch.labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 1f, 1f, 1f, 0f, 0f}, batch.inputMask().data(), 1e-6f);
        assertArrayEquals(new int[] {3, 1}, batch.inputLengths());
        assertArrayEquals(new int[] {2, 3}, batch.labelLengths());
        assertEquals(4, batch.inputPaddingStats().realTokens());
        assertEquals(6, batch.inputPaddingStats().paddedTokens());
        assertEquals(2, batch.inputPaddingStats().paddingTokens());
        assertEquals(5, batch.labelPaddingStats().realTokens());
        assertEquals(1, batch.labelPaddingStats().paddingTokens());
        assertEquals(9, batch.paddingEfficiency().realTokens());
        assertEquals(12, batch.paddingEfficiency().paddedTokens());
        assertEquals(3, batch.paddingEfficiency().paddingTokens());
        assertEquals(0.25, batch.paddingEfficiency().paddingRatio(), 1e-6);
        assertEquals(0.75, batch.paddingEfficiency().utilization(), 1e-6);

        Dataset<Dataset.Pair<GradTensor, GradTensor>> pairs = Dataset.of(
                new Dataset.Pair<>(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                        GradTensor.of(new float[] {1f, 0f}, 2)),
                new Dataset.Pair<>(
                        GradTensor.of(new float[] {5f, 6f}, 1, 2),
                        GradTensor.of(new float[] {1f}, 1)));
        DataLoader.PaddedBatch pairBatch = new DataLoader<>(pairs, 2)
                .collate(DataLoader.paddedTensorPairBatchCollate())
                .iterator()
                .next();

        assertArrayEquals(new long[] {2, 2, 2}, pairBatch.inputs().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 0f, 0f}, pairBatch.inputs().data(), 1e-6f);
        assertArrayEquals(new int[] {2, 1}, pairBatch.inputLengths());

        Dataset<Dataset.Sample> mismatched = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f, 6f, 7f}, 1, 3),
                        GradTensor.of(new float[] {0f}, 1)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DataLoader.paddedSampleBatchCollate().apply(mismatched.toList()));
    }

    @Test
    public void testCausalLanguageModelingCollatorPadsLabelsWithIgnoreIndex() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3),
                        GradTensor.of(new float[] {2f, 3f, 4f}, 3)),
                new Dataset.Sample(
                        GradTensor.of(new float[] {5f}, 1),
                        GradTensor.of(new float[] {6f}, 1)));

        DataLoader.PaddedBatch batch = new DataLoader<>(samples, 2)
                .collate(DataLoader.causalLanguageModelingBatchCollate(0))
                .iterator()
                .next();

        assertEquals(-100.0f, DataLoader.DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX, 1e-6f);
        assertArrayEquals(new long[] {2, 3}, batch.inputs().shape());
        assertArrayEquals(new long[] {2, 3}, batch.labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 5f, 0f, 0f}, batch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 3f, 4f, 6f, -100f, -100f}, batch.labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 1f, 1f, 1f, 0f, 0f}, batch.inputMask().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 1f, 1f, 1f, 0f, 0f}, batch.labelMask().data(), 1e-6f);

        DataLoader.PaddedBatch custom = DataLoader.causalLanguageModelingBatchCollate(99f, -1f)
                .apply(samples.toList());
        assertArrayEquals(new float[] {1f, 2f, 3f, 5f, 99f, 99f}, custom.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 3f, 4f, 6f, -1f, -1f}, custom.labels().data(), 1e-6f);
    }

    @Test
    public void testLengthBucketBatchSamplerGroupsSimilarSequenceLengths() {
        int[] lengths = {9, 1, 8, 2, 7, 3, 6, 4, 5};
        LengthBucketBatchSampler sampler = DataLoader.lengthBucketBatchSampler(
                lengths,
                3,
                1,
                false,
                false,
                false,
                17L);

        List<Integer> order = sampler.sample(lengths.length);
        List<Integer> sorted = new ArrayList<>(order);
        Collections.sort(sorted);

        assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8), sorted);
        assertLengthSpreadWithin(lengths, order, 3, 2);

        LengthBucketBatchSampler dropped = DataLoader.lengthBucketBatchSampler(
                new int[] {5, 1, 4, 2, 3},
                2,
                1,
                false,
                false,
                true,
                17L);
        assertEquals(4, dropped.sampleCount(5));
        assertEquals(4, dropped.sample(5).size());
        assertThrows(IllegalArgumentException.class, () -> sampler.sample(8));
    }

    @Test
    public void testSequenceLengthHelpersInferLengthsFromTensorDatasets() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f}, 1),
                        GradTensor.of(new float[] {0f, 1f}, 2)),
                new Dataset.Sample(GradTensor.of(new float[] {6f, 7f, 8f}, 3),
                        GradTensor.of(new float[] {1f}, 1)));

        assertArrayEquals(new int[] {4, 1, 3}, DataLoader.sampleInputLengths(samples));
        assertArrayEquals(new int[] {1, 2, 1}, DataLoader.sampleLabelLengths(samples));

        Dataset<Dataset.Pair<GradTensor, GradTensor>> pairs = Dataset.of(
                new Dataset.Pair<>(GradTensor.of(new float[] {1f, 2f}, 2),
                        GradTensor.of(new float[] {0f}, 1)),
                new Dataset.Pair<>(GradTensor.of(new float[] {3f, 4f, 5f}, 3),
                        GradTensor.of(new float[] {1f, 0f}, 2)));

        assertArrayEquals(new int[] {2, 3}, DataLoader.tensorPairInputLengths(pairs));
        assertArrayEquals(new int[] {1, 2}, DataLoader.tensorPairLabelLengths(pairs));

        LengthBucketBatchSampler sampler = DataLoader.lengthBucketBatchSampler(
                samples,
                DataLoader.sampleInputLength(),
                2,
                1,
                false,
                false,
                false,
                99L);
        assertEquals(List.of(1, 2, 0), sampler.sample(samples.size()));

        Dataset<Dataset.Sample> scalarSamples = Dataset.of(
                new Dataset.Sample(GradTensor.scalar(1f), GradTensor.of(new float[] {1f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.sampleInputLengths(scalarSamples));
    }

    @Test
    public void testLengthBucketSamplerCombinesWithPaddedCollation() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f}, 1),
                        GradTensor.of(new float[] {0f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {6f, 7f, 8f}, 3),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {9f, 10f}, 2),
                        GradTensor.of(new float[] {0f}, 1)));
        int[] lengths = {4, 1, 3, 2};
        DataLoader<Dataset.Sample> sampled = new DataLoader<>(
                samples,
                2,
                false,
                false,
                DataLoader.lengthBucketBatchSampler(lengths, 2, 1, false, false, false, 99L));

        List<int[]> batchLengths = new ArrayList<>();
        List<DataLoader.PaddedBatch> batches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : sampled.collate(DataLoader.paddedSampleBatchCollate())) {
            batchLengths.add(batch.inputLengths());
            batches.add(batch);
        }

        assertArrayEquals(new int[] {1, 2}, batchLengths.get(0));
        assertArrayEquals(new int[] {3, 4}, batchLengths.get(1));

        PaddingEfficiencyReport report = DataLoader.paddingEfficiency(batches);
        assertEquals(4, report.inputs().batchSize());
        assertEquals(4, report.inputs().maxLength());
        assertEquals(10, report.inputs().realTokens());
        assertEquals(12, report.inputs().paddedTokens());
        assertEquals(2, report.inputs().paddingTokens());
        assertEquals(14, report.realTokens());
        assertEquals(16, report.paddedTokens());
        assertEquals(2, report.paddingTokens());
        assertEquals(0.125, report.paddingRatio(), 1e-6);
        assertEquals(0.875, report.utilization(), 1e-6);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.paddingEfficiency(List.of()));
    }

    @Test
    public void testTokenBudgetBatchSamplerBuildsVariableSizeSequenceBatches() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                sampleWithInputLength(1),
                sampleWithInputLength(2),
                sampleWithInputLength(3),
                sampleWithInputLength(4),
                sampleWithInputLength(5),
                sampleWithInputLength(6));

        TokenBudgetBatchSampler sampler = DataLoader.tokenBudgetBatchSampler(
                samples,
                DataLoader.sampleInputLength(),
                10,
                Integer.MAX_VALUE,
                false,
                false,
                7L);

        assertEquals(List.of(List.of(0, 1, 2), List.of(3, 4), List.of(5)), sampler.sampleBatches(samples.size()));
        TokenBudgetBatchReport plan = sampler.report(samples.size());
        assertEquals(6, plan.sampleCount());
        assertEquals(3, plan.batchCount());
        assertEquals(10, plan.maxTokens());
        assertArrayEquals(new int[] {3, 2, 1}, plan.batchSizes());
        assertArrayEquals(new int[] {3, 5, 6}, plan.batchMaxLengths());
        assertArrayEquals(new long[] {9L, 10L, 6L}, plan.batchTokenCosts());
        assertEquals(1, plan.minBatchSize());
        assertEquals(3, plan.maxBatchSize());
        assertEquals(6, plan.maxObservedLength());
        assertEquals(0, plan.oversizedExampleCount());
        assertEquals(21, plan.realTokens());
        assertEquals(25, plan.paddedTokens());
        assertEquals(4, plan.paddingTokens());
        assertEquals(0.16, plan.paddingRatio(), 1e-6);
        assertEquals(0.84, plan.utilization(), 1e-6);
        assertEquals(2.0, plan.averageBatchSize(), 1e-6);
        assertEquals(25.0 / 3.0, plan.averageTokenCost(), 1e-6);
        assertEquals(10L, plan.maxBatchTokenCost());
        int[] mutableBatchSizes = plan.batchSizes();
        mutableBatchSizes[0] = 99;
        assertArrayEquals(new int[] {3, 2, 1}, plan.batchSizes());

        DataLoader<Dataset.Sample> sampled = new DataLoader<>(
                samples,
                32,
                false,
                false,
                null,
                sampler);

        List<int[]> batchLengths = new ArrayList<>();
        List<DataLoader.PaddedBatch> variableBatches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : sampled.collate(DataLoader.paddedSampleBatchCollate())) {
            batchLengths.add(batch.inputLengths());
            variableBatches.add(batch);
        }

        assertEquals(3, sampled.numBatches());
        assertTrue(sampled.sampled());
        assertEquals(6, sampled.sampleCount());
        assertArrayEquals(new int[] {1, 2, 3}, batchLengths.get(0));
        assertArrayEquals(new int[] {4, 5}, batchLengths.get(1));
        assertArrayEquals(new int[] {6}, batchLengths.get(2));

        PaddingEfficiencyReport variableReport = DataLoader.paddingEfficiency(variableBatches);
        assertEquals(21, variableReport.inputs().realTokens());
        assertEquals(25, variableReport.inputs().paddedTokens());
        assertEquals(4, variableReport.inputs().paddingTokens());

        List<DataLoader.PaddedBatch> fixedBatches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : new DataLoader<>(samples, 3)
                .collate(DataLoader.paddedSampleBatchCollate())) {
            fixedBatches.add(batch);
        }
        PaddingEfficiencyReport fixedReport = DataLoader.paddingEfficiency(fixedBatches);
        assertEquals(27, fixedReport.inputs().paddedTokens());
        assertTrue(variableReport.inputs().paddingRatio() < fixedReport.inputs().paddingRatio());
        assertEquals(1, DataLoader.tokenBudgetBatchSampler(new int[] {11}, 10, 7L)
                .report(1)
                .oversizedExampleCount());
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tokenBudgetBatchSampler(new int[] {1}, 0, 7L));
        assertThrows(IllegalArgumentException.class, () -> sampler.sampleBatches(5));
    }

    @Test
    public void testCausalLanguageModelingDatasetBuildsNextTokenWindows() {
        int[] tokenIds = {10, 20, 30, 40, 50, 60, 70};
        Dataset<Dataset.Sample> dataset = DataLoader.causalLanguageModelingDataset(tokenIds, 3, 2);
        tokenIds[0] = 999;

        assertEquals(2, dataset.size());
        assertArrayEquals(new float[] {10f, 20f, 30f}, dataset.get(0).input().data(), 1e-6f);
        assertArrayEquals(new float[] {20f, 30f, 40f}, dataset.get(0).label().data(), 1e-6f);
        assertArrayEquals(new float[] {30f, 40f, 50f}, dataset.get(1).input().data(), 1e-6f);
        assertArrayEquals(new float[] {40f, 50f, 60f}, dataset.get(1).label().data(), 1e-6f);
        assertArrayEquals(new int[] {3, 3}, DataLoader.sampleInputLengths(dataset));

        DataLoader<Dataset.Sample> sampled = new DataLoader<>(
                dataset,
                32,
                false,
                false,
                null,
                DataLoader.tokenBudgetBatchSampler(dataset, DataLoader.sampleInputLength(), 6, 11L));

        List<DataLoader.PaddedBatch> batches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : sampled.collate(DataLoader.paddedSampleBatchCollate())) {
            batches.add(batch);
        }

        assertEquals(1, sampled.numBatches());
        assertArrayEquals(new int[] {3, 3}, batches.get(0).inputLengths());
        assertEquals(6, DataLoader.paddingEfficiency(batches).inputs().paddedTokens());

        assertEquals(0, DataLoader.causalLanguageModelingDataset(new int[] {1, 2, 3}, 3).size());
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.causalLanguageModelingDataset(new int[] {1, 2, 3, 4}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.causalLanguageModelingDataset(new int[] {1, 2, 3, 4}, 2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> dataset.get(2));
        assertThrows(ArithmeticException.class,
                () -> DataLoader.causalLanguageModelingDataset(new long[] {1L, (long) Integer.MAX_VALUE + 1L}, 1));
    }

    @Test
    public void testPackedCausalLanguageModelingDatasetAddsEosBetweenDocuments() {
        int[][] documents = {
                {1, 2, 3},
                {},
                {4, 5, 6}
        };
        Dataset<Dataset.Sample> dataset = DataLoader.packedCausalLanguageModelingDataset(documents, 99, 3, 2);
        documents[0][0] = 777;
        documents[2][0] = 888;

        assertEquals(2, dataset.size());
        assertArrayEquals(new float[] {1f, 2f, 3f}, dataset.get(0).input().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 3f, 99f}, dataset.get(0).label().data(), 1e-6f);
        assertArrayEquals(new float[] {3f, 99f, 4f}, dataset.get(1).input().data(), 1e-6f);
        assertArrayEquals(new float[] {99f, 4f, 5f}, dataset.get(1).label().data(), 1e-6f);
        assertArrayEquals(new int[] {3, 3}, DataLoader.sampleInputLengths(dataset));

        assertEquals(0, DataLoader.packedCausalLanguageModelingDataset(new int[][] {{}, {7}}, 99, 1).size());
        assertThrows(NullPointerException.class,
                () -> DataLoader.packedCausalLanguageModelingDataset(new int[][] {null}, 99, 3));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.packedCausalLanguageModelingDataset(new int[][] {{1, 2, 3}}, 99, 0));
        assertThrows(ArithmeticException.class,
                () -> DataLoader.packedCausalLanguageModelingDataset(
                        new long[][] {{1L, (long) Integer.MAX_VALUE + 1L}},
                        99L,
                        1));
        assertThrows(ArithmeticException.class,
                () -> DataLoader.packedCausalLanguageModelingDataset(
                        new long[][] {{1L, 2L}},
                        (long) Integer.MAX_VALUE + 1L,
                        1));
    }

    @Test
    public void testTensorDatasetDefensivelyCopiesSampleTuples() {
        GradTensor input = GradTensor.of(new float[] {1f, 2f}, 2);
        GradTensor label = GradTensor.of(new float[] {1f}, 1);
        GradTensor[] sample = new GradTensor[] {input, label};

        var dataset = new DataLoader.TensorDataset(sample);
        sample[0] = GradTensor.of(new float[] {99f, 100f}, 2);
        sample[1] = GradTensor.of(new float[] {0f}, 1);

        assertArrayEquals(new float[] {1f, 2f}, dataset.get(0)[0].data(), 1e-6f);
        assertArrayEquals(new float[] {1f}, dataset.get(0)[1].data(), 1e-6f);
    }

    @Test
    public void testTensorDatasetSplitAndSeededLoaderCompatibility() {
        var split = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                7L);

        var first = split.trainLoader(2, true, 42L);
        var second = split.trainLoader(2, true, 42L);

        assertEquals(4, split.train().size());
        assertEquals(2, split.validation().size());
        assertEquals(4, first.size());
        assertEquals(2, first.batchSize());
        assertTrue(first.shuffle());
        assertEquals(flattenInputs(first), flattenInputs(second));
        assertEquals(1, split.validationLoader(2).numBatches());
    }

    @Test
    public void testClassificationStratifiedSplitPreservesLabelsOnBothSides() {
        var first = DataLoader.classificationStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.2f, 0.8f,
                        0.8f, 0.2f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 1, 1, 0, 1, 0, 1},
                0.5,
                2026L);
        var second = DataLoader.classificationStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.2f, 0.8f,
                        0.8f, 0.2f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 1, 1, 0, 1, 0, 1},
                0.5,
                2026L);

        List<Float> trainLabels = flattenLabels(first.train());
        List<Float> validationLabels = flattenLabels(first.validation());
        assertEquals(4, first.train().size());
        assertEquals(4, first.validation().size());
        assertEquals(2, Collections.frequency(trainLabels, 0f));
        assertEquals(2, Collections.frequency(trainLabels, 1f));
        assertEquals(2, Collections.frequency(validationLabels, 0f));
        assertEquals(2, Collections.frequency(validationLabels, 1f));
        assertEquals(trainLabels, flattenLabels(second.train()));
        assertEquals(validationLabels, flattenLabels(second.validation()));
    }

    @Test
    public void testBinaryStratifiedSplitPreservesLabelsOnBothSides() {
        var split = DataLoader.binaryStratifiedSplit(
                GradTensor.of(new float[] {
                        -1f, -1f,
                        -0.8f, -0.7f,
                        -1.2f, -0.5f,
                        -0.6f, -1.1f,
                        1f, 1f,
                        0.9f, 0.7f,
                        1.2f, 0.5f,
                        0.6f, 1.1f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                0.5,
                99L);

        List<Float> trainLabels = flattenLabels(split.train());
        List<Float> validationLabels = flattenLabels(split.validation());
        assertEquals(4, split.train().size());
        assertEquals(4, split.validation().size());
        assertEquals(2, Collections.frequency(trainLabels, 0f));
        assertEquals(2, Collections.frequency(trainLabels, 1f));
        assertEquals(2, Collections.frequency(validationLabels, 0f));
        assertEquals(2, Collections.frequency(validationLabels, 1f));
    }

    @Test
    public void testPositiveWeightHelpersDeriveBceImbalanceWeights() {
        assertEquals(3.0f, DataLoader.binaryPositiveWeight(1, 0, 0, 0), 1e-6f);
        assertEquals(1.0f, DataLoader.binaryPositiveWeight(true, false), 1e-6f);
        assertArrayEquals(new float[] {3.0f, 3.0f, 1.0f}, DataLoader.multiLabelPositiveWeights(new int[][] {
                {1, 0, 1},
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        }), 1e-6f);
    }

    @Test
    public void testClassWeightHelpersDeriveCrossEntropyImbalanceWeights() {
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f},
                DataLoader.classWeights(0, 0, 0, 1),
                1e-6f);
        assertArrayEquals(
                new float[] {4.0f / 9.0f, 4.0f / 3.0f, 1.0f},
                DataLoader.classWeightsFor(3, 0, 0, 0, 1),
                1e-6f);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classWeightsFor(2, 0, 2));
    }

    @Test
    public void testMultiLabelBinaryStratifiedSplitBalancesPerLabelPositives() {
        var split = DataLoader.multiLabelBinaryStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.5f, 0.5f,
                        0.6f, 0.4f,
                        0.2f, 0.8f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 1},
                        {1, 1, 1}
                },
                0.5,
                123L);

        assertEquals(4, split.train().size());
        assertEquals(4, split.validation().size());
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.validation(), 3));
    }

    @Test
    public void testWeightedRandomSamplerWithReplacementFollowsPositiveWeightsDeterministically() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f, 20f, 30f, 40f}, 4, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 0f}, 4, 1));

        var loader = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .weightedRandomSampler(new float[] {0f, 0f, 1f, 0f}, 5, true, 123L)
                .build();

        assertTrue(loader.sampled());
        assertEquals(5, loader.sampleCount());
        assertEquals(3, loader.numBatches());
        assertEquals(List.of(30f, 30f, 30f, 30f, 30f), flattenInputs(loader));
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .weightedRandomSampler(new float[] {0f, 0f, 1f, 0f}, 5, true, 123L)
                .build()));
    }

    @Test
    public void testWeightedRandomSamplerWithoutReplacementDrawsUniquePositiveWeightSamples() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {0f, 1f, 1f, 0f}, 4, 1));

        var loader = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .weightedRandomSampler(new float[] {0f, 4f, 1f, 0f}, 2, false, 77L)
                .build();
        List<Float> drawn = flattenInputs(loader);
        Collections.sort(drawn);

        assertEquals(List.of(2f, 3f), drawn);
    }

    @Test
    public void testWeightedRandomSamplerDefensivelyCopiesWeights() {
        float[] weights = new float[] {0f, 1f};
        WeightedRandomSampler sampler = DataLoader.weightedRandomSampler(weights, 1, true, 5L);
        weights[1] = 0f;

        assertEquals(List.of(1), sampler.sample(2));
        assertArrayEquals(new float[] {0f, 1f}, sampler.sampleWeights(), 1e-6f);
    }

    @Test
    public void testRandomSamplerIsDeterministicAndSupportsReplacement() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1));

        var first = DataLoader.tensorBuilder(dataset)
                .batchSize(5)
                .randomSampler(2026L)
                .build();
        var second = DataLoader.tensorBuilder(dataset)
                .batchSize(5)
                .randomSampler(2026L)
                .build();
        List<Float> firstDraw = flattenInputs(first);
        List<Float> sorted = new ArrayList<>(firstDraw);
        Collections.sort(sorted);

        assertTrue(first.sampled());
        assertEquals(firstDraw, flattenInputs(second));
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f), sorted);

        var single = DataLoader.tensorDataset(
                GradTensor.of(new float[] {7f}, 1, 1),
                GradTensor.of(new float[] {1f}, 1, 1));
        var replacement = DataLoader.tensorBuilder(single)
                .batchSize(2)
                .randomSampler(4, true, 99L)
                .build();

        assertEquals(4, replacement.sampleCount());
        assertEquals(2, replacement.numBatches());
        assertEquals(List.of(7f, 7f, 7f, 7f), flattenInputs(replacement));
    }

    @Test
    public void testSubsetAndSequentialSamplersSelectExpectedRows() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f, 20f, 30f, 40f, 50f}, 5, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 1f, 1f}, 5, 1));

        var subset = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(7L)
                .subsetSampler(4, 0, 2)
                .build();
        var sequential = DataLoader.tensorBuilder(dataset)
                .batchSize(3)
                .sequentialSampler()
                .build();

        assertTrue(subset.sampled());
        assertEquals(3, subset.sampleCount());
        assertEquals(2, subset.numBatches());
        assertEquals(List.of(50f, 10f, 30f), flattenInputs(subset));
        assertArrayEquals(new int[] {4, 0, 2}, DataLoader.subsetSampler(4, 0, 2).indices());
        assertEquals(List.of(10f, 20f, 30f, 40f, 50f), flattenInputs(sequential));
    }

    @Test
    public void testDistributedSamplerPartitionsAndPadsDatasetAcrossRanks() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 1f, 1f, 1f, 1f}, 5, 1));

        var rank0 = DataLoader.tensorBuilder(dataset)
                .batchSize(3)
                .distributedSampler(2, 0, false, false, 7L)
                .build();
        var rank1 = DataLoader.tensorBuilder(dataset)
                .batchSize(3)
                .distributedSampler(2, 1, false, false, 7L)
                .build();

        assertEquals(3, rank0.sampleCount());
        assertEquals(3, rank1.sampleCount());
        assertEquals(List.of(1f, 3f, 5f), flattenInputs(rank0));
        assertEquals(List.of(2f, 4f, 1f), flattenInputs(rank1));

        var dropRank0 = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .distributedSampler(2, 0, false, true, 7L)
                .build();
        var dropRank1 = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .distributedSampler(2, 1, false, true, 7L)
                .build();

        assertEquals(2, dropRank0.sampleCount());
        assertEquals(List.of(1f, 3f), flattenInputs(dropRank0));
        assertEquals(List.of(2f, 4f), flattenInputs(dropRank1));
    }

    @Test
    public void testDistributedSamplerShuffleIsSeedAndEpochDeterministic() {
        DistributedSampler epoch0 = DataLoader.distributedSampler(2, 0, true, true, 42L, 0L);
        DistributedSampler epoch0Again = DataLoader.distributedSampler(2, 0, true, true, 42L, 0L);
        DistributedSampler epoch1 = epoch0.forEpoch(1L);

        assertEquals(epoch0.sample(8), epoch0Again.sample(8));
        assertTrue(!epoch0.sample(8).equals(epoch1.sample(8)));
        assertEquals(1L, epoch1.epoch());

        List<Integer> combined = new ArrayList<>();
        combined.addAll(DataLoader.distributedSampler(2, 0, true, true, 42L, 0L).sample(8));
        combined.addAll(DataLoader.distributedSampler(2, 1, true, true, 42L, 0L).sample(8));
        Collections.sort(combined);

        assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7), combined);
    }

    @Test
    public void testBalancedSampleWeightHelpersReturnPerSampleInverseClassWeights() {
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f},
                DataLoader.classBalancedSampleWeights(0, 0, 0, 1),
                1e-6f);
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f},
                DataLoader.binaryBalancedSampleWeights(0, 0, 0, 1),
                1e-6f);
    }

    @Test
    public void testClassificationLoaderBuildsClassIndexLabels() {
        var loader = DataLoader.classification(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[] {0, 1, 1},
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {0f, 1f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
    }

    @Test
    public void testBinaryLoaderBuildsBceCompatibleColumnLabels() {
        var loader = DataLoader.binary(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[] {1, 0, 1},
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 1}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 0f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 1}, batches.get(1).labels().shape());
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {3, 1}, DataLoader.binaryLabels(true, false, true).shape());
    }

    @Test
    public void testBinaryLoaderBuildsMultiLabelBceCompatibleLabels() {
        var loader = DataLoader.binary(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 0},
                        {1, 1, 0}
                },
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 3}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 0f, 1f, 0f, 1f, 0f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 3}, batches.get(1).labels().shape());
        assertArrayEquals(new float[] {1f, 1f, 0f}, batches.get(1).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {2, 2}, DataLoader.binaryLabels(new boolean[][] {
                {true, false},
                {false, true}
        }).shape());
    }

    @Test
    public void testTensorLoaderCanReshuffleSeededEpochsDeterministically() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1),
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1));
        var loader = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(true)
                .seed(321L)
                .reshuffleEachEpoch()
                .build();
        var replay = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(true)
                .seed(321L)
                .reshuffleEachEpoch()
                .build();

        List<Float> firstEpoch = flattenInputs(loader);
        List<Float> secondEpoch = flattenInputs(loader);

        assertTrue(loader.shuffle());
        assertTrue(loader.reshuffleEachEpoch());
        assertNotEquals(firstEpoch, secondEpoch);
        assertEquals(firstEpoch, flattenInputs(replay));
        assertEquals(secondEpoch, flattenInputs(replay));
        assertEquals(firstEpoch, flattenInputs(loader.epoch(0L)));
        assertEquals(secondEpoch, flattenInputs(loader.epoch(1L)));
        assertThrows(IllegalArgumentException.class, () -> loader.epoch(-1L));
    }

    @Test
    public void testTensorLoaderCanResumeReshuffleFromInitialEpoch() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1),
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1));
        var baseline = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(true)
                .seed(321L)
                .reshuffleEachEpoch()
                .build();
        flattenInputs(baseline);
        flattenInputs(baseline);
        List<Float> thirdEpoch = flattenInputs(baseline);

        var resumed = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(true)
                .seed(321L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();

        assertEquals(2L, resumed.initialEpoch());
        assertEquals(thirdEpoch, flattenInputs(resumed));
        var viewFirst = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(true)
                .seed(321L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();
        assertEquals(thirdEpoch, flattenInputs(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenInputs(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenInputs(viewFirst));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).initialEpoch(-1L));
    }

    @Test
    public void testTensorLoaderRejectsInvalidConfigEarly() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {2f, 4f}, 2, 1));

        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).batchSize(0));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(0.0, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.weightedRandomSampler(new float[] {0f, 0f}, 1, true, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.weightedRandomSampler(new float[] {1f}, 2, false, 1L));
        var mismatchedSampler = DataLoader.tensorBuilder(dataset)
                .weightedRandomSampler(new float[] {1f}, 1, true, 1L)
                .build();
        assertThrows(IllegalArgumentException.class, mismatchedSampler::numBatches);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.randomSampler(0, true, 1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.randomSampler(3, false, 1L).sample(2));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.subsetSampler(-1));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.subsetSampler(2).sample(2));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.distributedSampler(0, 0));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.distributedSampler(2, 2));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f}, 1, 1)));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classificationDataset(
                GradTensor.of(new float[] {1f, 2f}, 1, 2),
                new int[] {0, -1}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f}, 1, 2),
                new int[] {2}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                new int[][] {{0, 1}, {1}}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                new int[][] {{0, 2}, {1, 0}}));
    }

    private static List<Float> flattenInputs(Iterable<DataLoader.Batch> loader) {
        List<Float> values = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            for (float value : batch.inputs().data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static <T> List<T> flattenGeneric(DataLoader<T> loader) {
        List<T> values = new ArrayList<>();
        for (List<T> batch : loader) {
            values.addAll(batch);
        }
        return values;
    }

    private static Dataset.Sample sampleWithInputLength(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = i + 1f;
        }
        return new Dataset.Sample(GradTensor.of(values, length), GradTensor.of(new float[] {1f}, 1));
    }

    private static List<Float> flattenLabels(DataLoader.TensorDataset dataset) {
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            for (float value : dataset.get(i)[1].data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static int[] positiveCounts(DataLoader.TensorDataset dataset, int columns) {
        int[] counts = new int[columns];
        for (int i = 0; i < dataset.size(); i++) {
            float[] labels = dataset.get(i)[1].data();
            assertEquals(columns, labels.length);
            for (int column = 0; column < columns; column++) {
                if (labels[column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }

    private static void assertLengthSpreadWithin(
            int[] lengths,
            List<Integer> order,
            int batchSize,
            int maxSpread) {
        for (int start = 0; start < order.size(); start += batchSize) {
            int end = Math.min(start + batchSize, order.size());
            int min = Integer.MAX_VALUE;
            int max = Integer.MIN_VALUE;
            for (int i = start; i < end; i++) {
                int length = lengths[order.get(i)];
                min = Math.min(min, length);
                max = Math.max(max, length);
            }
            assertTrue(max - min <= maxSpread, "batch length spread was " + (max - min));
        }
    }
}
