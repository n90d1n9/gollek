package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class DataLoaderTest {

    @Test
    void tensorLoaderBatchesRowsWithoutDroppingRemainder() {
        DataLoader.TensorDataLoader loader = DataLoader.tensors(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f}, 5, 1),
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(3, batches.size());
        assertEquals(5, loader.size());
        assertEquals(2, loader.batchSize());
        assertArrayEquals(new long[] {2, 1}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {1, 1}, batches.get(2).inputs().shape());
        assertArrayEquals(new float[] {5f}, batches.get(2).inputs().data(), 1e-6f);
    }

    @Test
    void defaultTensorCollateCanBeReusedExplicitly() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 3, 2),
                GradTensor.of(new float[] {10f, 20f, 30f}, 3));

        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .collateFn(DataLoader.defaultTensorCollate())
                .build();

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 2}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {2}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {10f, 20f}, batches.get(0).labels().data(), 1e-6f);
    }

    @Test
    void batchRejectsInvalidTensorsAtConstruction() {
        GradTensor tensor = GradTensor.of(new float[] {1f, 2f}, 2, 1);

        assertThrows(NullPointerException.class, () -> new DataLoader.Batch(null, tensor));
        assertThrows(NullPointerException.class, () -> new DataLoader.Batch(tensor, null));
        assertThrows(IllegalArgumentException.class, () -> new DataLoader.Batch(GradTensor.scalar(1f), tensor));
        assertThrows(IllegalArgumentException.class, () -> new DataLoader.Batch(
                GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1),
                tensor));
    }

    @Test
    void genericDataLoaderSeedMakesShuffleDeterministicAndSurvivesMap() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
        DataLoader<Integer> first = new DataLoader<>(dataset, 3, true, false, 123L);
        DataLoader<Integer> second = new DataLoader<>(dataset, 3, true, false, 123L);

        assertEquals(flattenGeneric(first), flattenGeneric(second));
        assertEquals(10, first.size());
        assertEquals(3, first.batchSize());
        assertEquals(4, first.numBatches());
        assertTrue(first.shuffle());
        assertFalse(first.dropLast());
        assertFalse(first.reshuffleEachEpoch());
        assertTrue(first.shuffleSeed().isPresent());
        assertEquals(123L, first.shuffleSeed().orElseThrow());
        assertEquals(flattenGeneric(first), flattenGeneric(first));

        DataLoader<String> mapped = first.map(value -> "item-" + value);
        assertEquals(3, mapped.batchSize());
        assertTrue(mapped.shuffle());
        assertFalse(mapped.dropLast());
        assertFalse(mapped.reshuffleEachEpoch());
        assertEquals(123L, mapped.shuffleSeed().orElseThrow());
        assertEquals(
                flattenGeneric(first).stream().map(value -> "item-" + value).toList(),
                flattenGeneric(mapped));
    }

    @Test
    void genericDataLoaderCanReshuffleSeededEpochsDeterministically() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
        DataLoader<Integer> loader = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .build();
        DataLoader<Integer> replay = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .build();

        List<Integer> firstEpoch = flattenGeneric(loader);
        List<Integer> secondEpoch = flattenGeneric(loader);

        assertTrue(loader.shuffle());
        assertTrue(loader.reshuffleEachEpoch());
        assertEquals(123L, loader.shuffleSeed().orElseThrow());
        assertNotEquals(firstEpoch, secondEpoch);
        assertEquals(firstEpoch, flattenGeneric(replay));
        assertEquals(secondEpoch, flattenGeneric(replay));
        assertEquals(firstEpoch, flattenGeneric(loader.epoch(0L)));
        assertEquals(secondEpoch, flattenGeneric(loader.epoch(1L)));
        assertThrows(IllegalArgumentException.class, () -> loader.epoch(-1L));

        DataLoader<String> mapped = loader.map(value -> "item-" + value);
        assertTrue(mapped.reshuffleEachEpoch());
        assertEquals(firstEpoch.stream().map(value -> "item-" + value).toList(), flattenGeneric(mapped));
    }

    @Test
    void genericDataLoaderCanResumeReshuffleFromInitialEpoch() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
        DataLoader<Integer> baseline = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .build();
        flattenGeneric(baseline);
        flattenGeneric(baseline);
        List<Integer> thirdEpoch = flattenGeneric(baseline);

        DataLoader<Integer> resumed = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();

        assertEquals(2L, resumed.initialEpoch());
        assertEquals(thirdEpoch, flattenGeneric(resumed));
        DataLoader<Integer> viewFirst = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();
        assertEquals(thirdEpoch, flattenGeneric(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenGeneric(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenGeneric(viewFirst));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.builder(dataset).initialEpoch(-1L));

        DataLoader.CollatingDataLoader<Integer, Integer> collating = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .collate(batch -> batch.stream().mapToInt(Integer::intValue).sum());
        assertEquals(
                batchSums(thirdEpoch, 3),
                collect(collating.epoch(2L)));
    }

    @Test
    void genericDataLoaderBuilderConfiguresSeededShuffleAndDropLast() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5, 6));
        DataLoader<Integer> first = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(77L)
                .dropLast()
                .build();
        DataLoader<Integer> second = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle()
                .dropLast(true)
                .randomSeed(77L)
                .build();

        assertEquals(flattenGeneric(first), flattenGeneric(second));
        assertEquals(7, first.size());
        assertEquals(3, first.batchSize());
        assertEquals(2, first.numBatches());
        assertTrue(first.shuffle());
        assertTrue(first.dropLast());
        assertEquals(77L, first.shuffleSeed().orElseThrow());
        assertEquals(6, flattenGeneric(first).size());
    }

    @Test
    void loaderPlanExposesStableGenericAndCollatingMetadata() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5, 6));
        DataLoader<Integer> loader = DataLoader.builder(dataset)
                .batchSize(3)
                .shuffle(77L)
                .dropLast()
                .build();

        DataLoaderPlan plan = loader.plan();
        Map<String, Object> metadata = plan.toMetadata("train.loader");

        assertEquals("generic", plan.kind());
        assertEquals(7, plan.datasetSize());
        assertEquals(7, plan.sampleCount());
        assertEquals(3, plan.batchSize());
        assertEquals(2, plan.batchCount());
        assertFalse(plan.sampled());
        assertTrue(plan.shuffle());
        assertTrue(plan.dropLast());
        assertTrue(plan.hasShuffleSeed());
        assertEquals(77L, plan.shuffleSeed());
        assertEquals(1.0, plan.sampleCoverageRatio(), 1e-6);
        assertEquals(6, plan.nominalBatchCapacity());
        assertTrue(plan.summary().contains("batchCount=2"));
        assertEquals("generic", metadata.get("train.loader.kind"));
        assertEquals(2, metadata.get("train.loader.batchCount"));
        assertEquals(77L, metadata.get("train.loader.shuffleSeed"));

        DataLoader.CollatingDataLoader<Integer, Integer> collating = loader.collate(
                batch -> batch.stream().mapToInt(Integer::intValue).sum());
        assertEquals("collating", collating.plan().kind());
        assertEquals(plan.batchCount(), collating.plan().batchCount());
    }

    @Test
    void loaderPlanExposesTensorMetadata() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 3, 2),
                GradTensor.of(new float[] {10f, 20f, 30f}, 3));
        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(123L)
                .dropLast()
                .build();

        DataLoaderPlan plan = loader.plan();

        assertEquals("tensor", plan.kind());
        assertEquals(3, plan.datasetSize());
        assertEquals(3, plan.sampleCount());
        assertEquals(2, plan.batchSize());
        assertEquals(1, plan.batchCount());
        assertFalse(plan.sampled());
        assertTrue(plan.shuffle());
        assertTrue(plan.dropLast());
        assertEquals(123L, plan.shuffleSeed());
        assertEquals(1.0, plan.sampleCoverageRatio(), 1e-6);
        assertEquals(2, plan.nominalBatchCapacity());
    }

    @Test
    void genericDataLoaderBuilderSupportsSamplersAndPreservesThemAcrossMap() {
        Dataset<Integer> dataset = datasetOf(List.of(0, 1, 2, 3, 4, 5));
        DataLoader<Integer> subset = DataLoader.builder(dataset)
                .batchSize(2)
                .shuffle(99L)
                .subsetSampler(4, 1, 4)
                .build();

        assertEquals(6, subset.size());
        assertEquals(3, subset.sampleCount());
        assertEquals(2, subset.numBatches());
        assertTrue(subset.sampled());
        assertTrue(subset.shuffle());
        assertEquals(List.of(4, 1, 4), flattenGeneric(subset));

        DataLoader<String> mapped = subset.map(value -> "v" + value);
        assertEquals(3, mapped.sampleCount());
        assertTrue(mapped.sampled());
        assertEquals(List.of("v4", "v1", "v4"), flattenGeneric(mapped));

        DataLoader<Integer> random = DataLoader.builder(dataset)
                .batchSize(3)
                .randomSampler(4, true, 7L)
                .build();
        List<Integer> expectedRandom = DataLoader.randomSampler(4, true, 7L).sample(dataset.size()).stream()
                .map(dataset::get)
                .toList();
        assertEquals(4, random.sampleCount());
        assertEquals(expectedRandom, flattenGeneric(random));

        DataLoader<Integer> invalidSubset = DataLoader.builder(dataset)
                .subsetSampler(6)
                .build();
        assertThrows(IllegalArgumentException.class, invalidSubset::numBatches);

        IndexSampler invalidSampleCount = new IndexSampler() {
            @Override
            public List<Integer> sample(int datasetSize) {
                return List.of();
            }

            @Override
            public int sampleCount(int datasetSize) {
                return -1;
            }
        };
        DataLoader<Integer> invalidSampler = DataLoader.builder(dataset)
                .sampler(invalidSampleCount)
                .build();
        assertThrows(IllegalArgumentException.class, invalidSampler::numBatches);
    }

    @Test
    void genericDataLoaderCollatesCustomBatchesAfterSampling() {
        record WordBatch(List<String> words, int characterCount) {}

        Dataset<String> dataset = Dataset.of("alpha", "b", "gamma", "delta");
        DataLoader.CollatingDataLoader<String, WordBatch> loader = DataLoader.builder(dataset)
                .batchSize(2)
                .shuffle(101L)
                .subsetSampler(2, 0, 2)
                .collate(batch -> new WordBatch(
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
        assertEquals(101L, loader.shuffleSeed().orElseThrow());
        assertEquals(List.of("gamma", "alpha"), batches.get(0).words());
        assertEquals(10, batches.get(0).characterCount());
        assertEquals(List.of("gamma"), batches.get(1).words());
        assertEquals(5, batches.get(1).characterCount());

        DataLoader<Integer> base = DataLoader.builder(Dataset.of(1, 2, 3, 4))
                .batchSize(3)
                .build();
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
    void genericDataLoaderPrefetchesBatchesWithoutChangingOrder() {
        DataLoader<Integer> loader = DataLoader.builder(Dataset.of(0, 1, 2, 3, 4))
                .batchSize(2)
                .build();

        List<List<Integer>> batches = new ArrayList<>();
        try (PrefetchingIterable<List<Integer>> prefetched = loader.prefetch(2)) {
            DataLoaderPrefetchPlan plan = prefetched.plan();
            assertTrue(plan.enabled());
            assertEquals(2, plan.bufferSize());
            assertEquals(1, plan.workerCount());
            assertEquals(2, plan.maxBufferedItems());
            assertTrue(plan.summary().contains("bufferSize=2"));
            Map<String, Object> metadata = plan.toMetadata("train.loader.prefetch");
            assertEquals(true, metadata.get("train.loader.prefetch.enabled"));
            assertEquals(2, metadata.get("train.loader.prefetch.bufferSize"));
            assertEquals(1, metadata.get("train.loader.prefetch.workerCount"));
            for (List<Integer> batch : prefetched) {
                batches.add(batch);
            }
        }

        assertEquals(List.of(List.of(0, 1), List.of(2, 3), List.of(4)), batches);

        List<List<Integer>> defaultBatches = new ArrayList<>();
        try (PrefetchingIterable<List<Integer>> prefetched = loader.prefetch()) {
            DataLoaderPrefetchPlan plan = prefetched.plan();
            assertEquals(DataLoaderPrefetchPlan.recommended(), plan);
            assertEquals(DataLoaderPrefetchPlan.DEFAULT_BUFFER_SIZE, prefetched.bufferSize());
            for (List<Integer> batch : prefetched) {
                defaultBatches.add(batch);
            }
        }

        assertEquals(batches, defaultBatches);
        assertThrows(IllegalArgumentException.class, () -> loader.prefetch(0));

        DataLoaderPrefetchPlan disabled = PrefetchingIterable.disabledPlan();
        assertFalse(disabled.enabled());
        assertEquals(0, disabled.bufferSize());
        assertEquals(0, disabled.workerCount());
        assertEquals(0, disabled.maxBufferedItems());
        assertEquals(false, disabled.toMetadata("train.loader.prefetch.").get("train.loader.prefetch.enabled"));
    }

    @Test
    void collatingAndTensorLoadersExposePrefetching() {
        DataLoader.CollatingDataLoader<Integer, Integer> sums = DataLoader.builder(Dataset.of(1, 2, 3, 4))
                .batchSize(2)
                .collate(batch -> batch.stream().mapToInt(Integer::intValue).sum());

        List<Integer> collectedSums = new ArrayList<>();
        try (PrefetchingIterable<Integer> prefetched = sums.prefetch()) {
            assertEquals(DataLoaderPrefetchPlan.DEFAULT_BUFFER_SIZE, prefetched.bufferSize());
            for (Integer sum : prefetched) {
                collectedSums.add(sum);
            }
        }
        assertEquals(List.of(3, 7), collectedSums);

        DataLoader.TensorDataLoader tensorLoader = DataLoader.tensors(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {10f, 20f, 30f, 40f}, 4, 1),
                3);

        List<DataLoader.Batch> tensorBatches = new ArrayList<>();
        try (PrefetchingIterable<DataLoader.Batch> prefetched = tensorLoader.prefetch()) {
            assertEquals(DataLoaderPrefetchPlan.recommended(), prefetched.plan());
            for (DataLoader.Batch batch : prefetched) {
                tensorBatches.add(batch);
            }
        }

        assertEquals(2, tensorBatches.size());
        assertArrayEquals(new long[] {3, 1}, tensorBatches.get(0).inputs().shape());
        assertArrayEquals(new long[] {1, 1}, tensorBatches.get(1).inputs().shape());
        assertArrayEquals(new float[] {4f}, tensorBatches.get(1).inputs().data(), 1e-6f);
    }

    @Test
    void tensorDataLoaderCanReshuffleSeededEpochsDeterministically() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1),
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1));
        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(321L)
                .reshuffleEachEpoch()
                .build();
        DataLoader.TensorDataLoader replay = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(321L)
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
    void tensorDataLoaderCanResumeReshuffleFromInitialEpoch() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1),
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 10, 1));
        DataLoader.TensorDataLoader baseline = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(321L)
                .reshuffleEachEpoch()
                .build();
        flattenInputs(baseline);
        flattenInputs(baseline);
        List<Float> thirdEpoch = flattenInputs(baseline);

        DataLoader.TensorDataLoader resumed = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(321L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();

        assertEquals(2L, resumed.initialEpoch());
        assertEquals(thirdEpoch, flattenInputs(resumed));
        DataLoader.TensorDataLoader viewFirst = DataLoader.tensorBuilder(dataset)
                .batchSize(4)
                .shuffle(321L)
                .reshuffleEachEpoch()
                .initialEpoch(2L)
                .build();
        assertEquals(thirdEpoch, flattenInputs(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenInputs(viewFirst.epoch(2L)));
        assertEquals(thirdEpoch, flattenInputs(viewFirst));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).initialEpoch(-1L));
    }

    @Test
    void prefetchingPropagatesSourceIteratorFailures() {
        Iterable<Integer> failingSource = () -> new Iterator<>() {
            private int position;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public Integer next() {
                if (position++ == 0) {
                    return 42;
                }
                throw new IllegalStateException("boom");
            }
        };

        try (PrefetchingIterable<Integer> prefetched = DataLoader.prefetch(failingSource, 1)) {
            PrefetchingIterable.CloseableIterator<Integer> iterator = prefetched.iterator();
            assertEquals(42, iterator.next());
            IllegalStateException error = assertThrows(IllegalStateException.class, iterator::hasNext);
            assertEquals("boom", error.getMessage());
        }
    }

    @Test
    void genericTensorCollatorsBuildTrainingBatchesFromSamplesAndPairs() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f}, 2), GradTensor.of(new float[] {0f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {3f, 4f}, 2), GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f, 6f}, 2), GradTensor.of(new float[] {0f}, 1)));

        DataLoader.CollatingDataLoader<Dataset.Sample, DataLoader.Batch> sampleLoader = DataLoader.builder(samples)
                .batchSize(2)
                .collate(DataLoader.sampleBatchCollate());

        List<DataLoader.Batch> sampleBatches = new ArrayList<>();
        for (DataLoader.Batch batch : sampleLoader) {
            sampleBatches.add(batch);
        }

        assertEquals(2, sampleBatches.size());
        assertArrayEquals(new long[] {2, 2}, sampleBatches.get(0).inputs().shape());
        assertArrayEquals(new long[] {2, 1}, sampleBatches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, sampleBatches.get(0).inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {0f, 1f}, sampleBatches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 2}, sampleBatches.get(1).inputs().shape());

        Dataset<Dataset.Pair<GradTensor, GradTensor>> pairs = Dataset.of(
                new Dataset.Pair<>(GradTensor.of(new float[] {7f, 8f}, 2), GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Pair<>(GradTensor.of(new float[] {9f, 10f}, 2), GradTensor.of(new float[] {0f}, 1)));

        DataLoader.Batch pairBatch = DataLoader.builder(pairs)
                .batchSize(2)
                .collate(DataLoader.tensorPairBatchCollate())
                .iterator()
                .next();

        assertArrayEquals(new long[] {2, 2}, pairBatch.inputs().shape());
        assertArrayEquals(new float[] {7f, 8f, 9f, 10f}, pairBatch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 0f}, pairBatch.labels().data(), 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.sampleBatchCollate().apply(List.of()));
    }

    @Test
    void paddedTensorCollatorsBatchVariableLengthSequenceSamples() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3),
                        GradTensor.of(new float[] {10f, 11f}, 2)),
                new Dataset.Sample(
                        GradTensor.of(new float[] {4f}, 1),
                        GradTensor.of(new float[] {12f, 13f, 14f}, 3)));

        DataLoader.PaddedBatch batch = DataLoader.builder(samples)
                .batchSize(2)
                .collate(DataLoader.paddedSampleBatchCollate(-1f, -9f))
                .iterator()
                .next();

        assertArrayEquals(new long[] {2, 3}, batch.inputs().shape());
        assertArrayEquals(new long[] {2, 3}, batch.labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f, -1f, -1f}, batch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {10f, 11f, -9f, 12f, 13f, 14f}, batch.labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 1f, 1f, 1f, 0f, 0f}, batch.inputMask().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 1f, 0f, 1f, 1f, 1f}, batch.labelMask().data(), 1e-6f);
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

        int[] lengths = batch.inputLengths();
        lengths[0] = 99;
        assertArrayEquals(new int[] {3, 1}, batch.inputLengths());
        assertArrayEquals(batch.inputs().data(), batch.batch().inputs().data(), 1e-6f);
    }

    @Test
    void paddedBatchDefensivelyCopiesConstructorLengths() {
        int[] inputLengths = {3, 1};
        int[] labelLengths = {2, 3};
        DataLoader.PaddedBatch batch = new DataLoader.PaddedBatch(
                GradTensor.ones(2, 3),
                GradTensor.ones(2, 3),
                GradTensor.ones(2, 3),
                GradTensor.ones(2, 3),
                inputLengths,
                labelLengths);

        inputLengths[0] = 0;
        labelLengths[1] = 0;
        assertArrayEquals(new int[] {3, 1}, batch.inputLengths());
        assertArrayEquals(new int[] {2, 3}, batch.labelLengths());
        assertEquals(4, batch.inputPaddingStats().realTokens());
        assertEquals(5, batch.labelPaddingStats().realTokens());
    }

    @Test
    void paddedBatchRejectsMismatchedInputAndLabelBatchSizes() {
        assertThrows(IllegalArgumentException.class, () -> new DataLoader.PaddedBatch(
                GradTensor.ones(2, 3),
                GradTensor.ones(3, 3),
                GradTensor.ones(2, 3),
                GradTensor.ones(3, 3),
                new int[] {3, 1},
                new int[] {2, 3, 1}));
    }

    @Test
    void causalLanguageModelingCollatorPadsLabelsWithIgnoreIndex() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3),
                        GradTensor.of(new float[] {2f, 3f, 4f}, 3)),
                new Dataset.Sample(
                        GradTensor.of(new float[] {5f}, 1),
                        GradTensor.of(new float[] {6f}, 1)));

        DataLoader.PaddedBatch batch = DataLoader.builder(samples)
                .batchSize(2)
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
    void paddedTensorPairCollatorSupportsTrailingFeatureDimensions() {
        Dataset<Dataset.Pair<GradTensor, GradTensor>> pairs = Dataset.of(
                new Dataset.Pair<>(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                        GradTensor.of(new float[] {1f, 0f}, 2)),
                new Dataset.Pair<>(
                        GradTensor.of(new float[] {5f, 6f}, 1, 2),
                        GradTensor.of(new float[] {1f}, 1)));

        DataLoader.PaddedBatch batch = DataLoader.builder(pairs)
                .batchSize(2)
                .collate(DataLoader.paddedTensorPairBatchCollate())
                .iterator()
                .next();

        assertArrayEquals(new long[] {2, 2, 2}, batch.inputs().shape());
        assertArrayEquals(new long[] {2, 2}, batch.labels().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 0f, 0f}, batch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {1f, 0f, 1f, 0f}, batch.labels().data(), 1e-6f);
        assertArrayEquals(new int[] {2, 1}, batch.inputLengths());

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
    void lengthBucketBatchSamplerGroupsSimilarSequenceLengths() {
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
        assertEquals(9, sampler.sampleCount(lengths.length));
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

        assertThrows(IllegalArgumentException.class, () -> DataLoader.lengthBucketBatchSampler(lengths, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> sampler.sample(8));
    }

    @Test
    void sequenceLengthHelpersInferLengthsFromTensorDatasets() {
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
    void genericBuilderCombinesLengthBucketsWithPaddedCollation() {
        Dataset<Dataset.Sample> samples = Dataset.of(
                new Dataset.Sample(GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {5f}, 1),
                        GradTensor.of(new float[] {0f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {6f, 7f, 8f}, 3),
                        GradTensor.of(new float[] {1f}, 1)),
                new Dataset.Sample(GradTensor.of(new float[] {9f, 10f}, 2),
                        GradTensor.of(new float[] {0f}, 1)));

        DataLoader.CollatingDataLoader<Dataset.Sample, DataLoader.PaddedBatch> loader = DataLoader.builder(samples)
                .batchSize(2)
                .lengthBucketBatchSampler(DataLoader.sampleInputLength(), 1, false, false, 99L)
                .collate(DataLoader.paddedSampleBatchCollate());

        List<int[]> batchLengths = new ArrayList<>();
        List<DataLoader.PaddedBatch> batches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : loader) {
            batchLengths.add(batch.inputLengths());
            batches.add(batch);
        }

        assertEquals(2, loader.numBatches());
        assertTrue(loader.sampled());
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
    void tokenBudgetBatchSamplerBuildsVariableSizeSequenceBatches() {
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
        assertEquals(6, sampler.sampleCount(samples.size()));
        assertEquals(3, sampler.batchCount(samples.size()));
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

        DataLoader.CollatingDataLoader<Dataset.Sample, DataLoader.PaddedBatch> tokenBudgeted =
                DataLoader.builder(samples)
                        .tokenBudgetBatchSampler(DataLoader.sampleInputLength(), 10, Integer.MAX_VALUE, false, false, 7L)
                        .collate(DataLoader.paddedSampleBatchCollate());

        List<int[]> batchLengths = new ArrayList<>();
        List<DataLoader.PaddedBatch> variableBatches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : tokenBudgeted) {
            batchLengths.add(batch.inputLengths());
            variableBatches.add(batch);
        }

        assertEquals(3, tokenBudgeted.numBatches());
        assertTrue(tokenBudgeted.sampled());
        assertEquals(6, tokenBudgeted.sampleCount());
        assertArrayEquals(new int[] {1, 2, 3}, batchLengths.get(0));
        assertArrayEquals(new int[] {4, 5}, batchLengths.get(1));
        assertArrayEquals(new int[] {6}, batchLengths.get(2));

        PaddingEfficiencyReport variableReport = DataLoader.paddingEfficiency(variableBatches);
        assertEquals(21, variableReport.inputs().realTokens());
        assertEquals(25, variableReport.inputs().paddedTokens());
        assertEquals(4, variableReport.inputs().paddingTokens());

        List<DataLoader.PaddedBatch> fixedBatches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : DataLoader.builder(samples)
                .batchSize(3)
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
    void causalLanguageModelingDatasetBuildsNextTokenWindows() {
        int[] tokenIds = {10, 20, 30, 40, 50, 60, 70};
        Dataset<Dataset.Sample> dataset = DataLoader.causalLanguageModelingDataset(tokenIds, 3, 2);
        tokenIds[0] = 999;

        assertEquals(2, dataset.size());
        assertArrayEquals(new float[] {10f, 20f, 30f}, dataset.get(0).input().data(), 1e-6f);
        assertArrayEquals(new float[] {20f, 30f, 40f}, dataset.get(0).label().data(), 1e-6f);
        assertArrayEquals(new float[] {30f, 40f, 50f}, dataset.get(1).input().data(), 1e-6f);
        assertArrayEquals(new float[] {40f, 50f, 60f}, dataset.get(1).label().data(), 1e-6f);
        assertArrayEquals(new int[] {3, 3}, DataLoader.sampleInputLengths(dataset));

        DataLoader.CollatingDataLoader<Dataset.Sample, DataLoader.PaddedBatch> loader = DataLoader.builder(dataset)
                .tokenBudgetBatchSampler(DataLoader.sampleInputLength(), 6, Integer.MAX_VALUE, false, false, 11L)
                .collate(DataLoader.paddedSampleBatchCollate());

        List<DataLoader.PaddedBatch> batches = new ArrayList<>();
        for (DataLoader.PaddedBatch batch : loader) {
            batches.add(batch);
        }

        assertEquals(1, loader.numBatches());
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
    void packedCausalLanguageModelingDatasetAddsEosBetweenDocuments() {
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
    void genericDataLoaderBuilderRejectsInvalidBatchSize() {
        Dataset<Integer> dataset = datasetOf(List.of(1, 2, 3));

        assertThrows(IllegalArgumentException.class, () -> DataLoader.builder(dataset).batchSize(0));
    }

    @Test
    void datasetFactoriesDefensivelyCopyAndFeedGenericLoaderBuilder() {
        List<Integer> values = new ArrayList<>(List.of(1, 2, 3, 4));
        Dataset<Integer> fromList = Dataset.from(values);
        values.set(0, 99);
        values.add(5);

        assertEquals(4, fromList.size());
        assertEquals(1, fromList.get(0));
        assertEquals(
                List.of(1, 2, 3, 4),
                flattenGeneric(DataLoader.builder(fromList).batchSize(2).build()));

        Dataset<String> fromVarargs = Dataset.of("alpha", "beta", "gamma");
        assertEquals(3, fromVarargs.size());
        assertEquals("gamma", fromVarargs.get(2));
    }

    @Test
    void datasetTransformsComposeBeforeGenericLoading() {
        Dataset<String> transformed = Dataset.of(1, 2, 3, 4, 5)
                .filter(value -> value % 2 == 1)
                .map(value -> "n=" + value);

        assertEquals(3, transformed.size());
        assertEquals("n=3", transformed.get(1));
        assertEquals(List.of("n=1", "n=3", "n=5"), transformed.toList());
        assertThrows(UnsupportedOperationException.class, () -> transformed.toList().add("n=7"));
        assertEquals(
                List.of("n=1", "n=3", "n=5"),
                flattenGeneric(DataLoader.builder(transformed).batchSize(2).build()));
    }

    @Test
    void datasetSplitIsDeterministicAndFeedsGenericLoader() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        Dataset.Split<Integer> first = source.split(0.6, 12L);
        Dataset.Split<Integer> second = source.split(0.6, 12L);

        assertEquals(6, first.train().size());
        assertEquals(4, first.validation().size());
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(
                first.validation().toList(),
                flattenGeneric(DataLoader.builder(first.validation()).batchSize(2).build()));

        List<Integer> covered = new ArrayList<>();
        covered.addAll(first.train().toList());
        covered.addAll(first.validation().toList());
        Collections.sort(covered);
        assertEquals(source.toList(), covered);
    }

    @Test
    void datasetThreeWaySplitIsDeterministicAndValidatesFractions() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        Dataset.ThreeWaySplit<Integer> first = source.split(0.5, 0.3, 99L);
        Dataset.ThreeWaySplit<Integer> second = source.split(0.5, 0.3, 99L);

        assertEquals(5, first.train().size());
        assertEquals(3, first.validation().size());
        assertEquals(2, first.test().size());
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(first.test().toList(), second.test().toList());

        assertThrows(IllegalArgumentException.class, () -> source.split(1.0, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.split(0.8, 0.2, 1L));
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(1, 2).split(0.5, 0.25, 1L));
    }

    @Test
    void datasetStratifiedSplitKeepsLabelsBalancedAndFeedsGenericLoader() {
        Dataset<LabeledExample> source = Dataset.of(
                new LabeledExample("cat-1", "cat"),
                new LabeledExample("cat-2", "cat"),
                new LabeledExample("cat-3", "cat"),
                new LabeledExample("cat-4", "cat"),
                new LabeledExample("cat-5", "cat"),
                new LabeledExample("cat-6", "cat"),
                new LabeledExample("dog-1", "dog"),
                new LabeledExample("dog-2", "dog"),
                new LabeledExample("dog-3", "dog"),
                new LabeledExample("dog-4", "dog"));

        Dataset.Split<LabeledExample> first = source.stratifiedSplit(LabeledExample::label, 0.5, 42L);
        Dataset.Split<LabeledExample> second = source.stratifiedSplit(LabeledExample::label, 0.5, 42L);

        assertEquals(Map.of("cat", 3, "dog", 2), labelCounts(first.train()));
        assertEquals(Map.of("cat", 3, "dog", 2), labelCounts(first.validation()));
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(
                first.train().toList(),
                flattenGeneric(DataLoader.builder(first.train()).batchSize(2).build()));
    }

    @Test
    void datasetStratifiedThreeWaySplitKeepsLabelsBalanced() {
        Dataset<LabeledExample> source = Dataset.of(
                new LabeledExample("cat-1", "cat"),
                new LabeledExample("cat-2", "cat"),
                new LabeledExample("cat-3", "cat"),
                new LabeledExample("cat-4", "cat"),
                new LabeledExample("cat-5", "cat"),
                new LabeledExample("cat-6", "cat"),
                new LabeledExample("dog-1", "dog"),
                new LabeledExample("dog-2", "dog"),
                new LabeledExample("dog-3", "dog"),
                new LabeledExample("dog-4", "dog"),
                new LabeledExample("dog-5", "dog"),
                new LabeledExample("dog-6", "dog"));

        Dataset.ThreeWaySplit<LabeledExample> first = source.stratifiedSplit(LabeledExample::label, 0.5, 0.25, 99L);
        Dataset.ThreeWaySplit<LabeledExample> second = source.stratifiedSplit(LabeledExample::label, 0.5, 0.25, 99L);

        assertEquals(Map.of("cat", 3, "dog", 3), labelCounts(first.train()));
        assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(first.validation()));
        assertEquals(Map.of("cat", 1, "dog", 1), labelCounts(first.test()));
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(first.test().toList(), second.test().toList());

        assertThrows(IllegalArgumentException.class, () -> source.stratifiedSplit(LabeledExample::label, 1.0, 1L));
        assertThrows(
                IllegalArgumentException.class,
                () -> source.stratifiedSplit(LabeledExample::label, 0.7, 0.3, 1L));
    }

    @Test
    void datasetKFoldIsDeterministicAndCoversEverySamplePerPass() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7);
        List<Dataset.Fold<Integer>> first = source.kFold(3, 31L);
        List<Dataset.Fold<Integer>> second = source.kFold(3, 31L);

        assertEquals(3, first.size());
        assertEquals(0, first.get(0).foldIndex());
        assertEquals(3, first.get(0).foldCount());
        assertEquals(List.of(3, 2, 2), first.stream().map(fold -> fold.validation().size()).toList());
        assertEquals(first.stream().map(fold -> fold.validation().toList()).toList(),
                second.stream().map(fold -> fold.validation().toList()).toList());
        assertValidationCovers(source, first);
        assertEquals(
                first.get(0).train().toList(),
                flattenGeneric(DataLoader.builder(first.get(0).train()).batchSize(2).build()));
    }

    @Test
    void datasetRepeatedKFoldCreatesIndependentFullValidationPasses() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6);
        List<Dataset.Fold<Integer>> folds = source.repeatedKFold(3, 2, 17L);

        assertEquals(6, folds.size());
        assertEquals(0, folds.get(0).foldIndex());
        assertEquals(5, folds.get(5).foldIndex());
        assertEquals(6, folds.get(0).foldCount());
        assertValidationCovers(source, folds.subList(0, 3));
        assertValidationCovers(source, folds.subList(3, 6));
        assertThrows(IllegalArgumentException.class, () -> source.kFold(1, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.kFold(7, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.repeatedKFold(3, 0, 1L));
    }

    @Test
    void datasetStratifiedKFoldBalancesLabelsInEveryValidationFold() {
        Dataset<LabeledExample> source = Dataset.of(
                new LabeledExample("cat-1", "cat"),
                new LabeledExample("cat-2", "cat"),
                new LabeledExample("cat-3", "cat"),
                new LabeledExample("cat-4", "cat"),
                new LabeledExample("cat-5", "cat"),
                new LabeledExample("cat-6", "cat"),
                new LabeledExample("dog-1", "dog"),
                new LabeledExample("dog-2", "dog"),
                new LabeledExample("dog-3", "dog"));

        List<Dataset.Fold<LabeledExample>> first = source.stratifiedKFold(LabeledExample::label, 3, 55L);
        List<Dataset.Fold<LabeledExample>> second = source.stratifiedKFold(LabeledExample::label, 3, 55L);

        assertEquals(3, first.size());
        for (Dataset.Fold<LabeledExample> fold : first) {
            assertEquals(Map.of("cat", 2, "dog", 1), labelCounts(fold.validation()));
            assertEquals(6, fold.train().size());
        }
        assertEquals(first.stream().map(fold -> fold.validation().toList()).toList(),
                second.stream().map(fold -> fold.validation().toList()).toList());
        assertLabeledValidationCovers(source, first);
    }

    @Test
    void datasetRepeatedStratifiedKFoldRunsBalancedPasses() {
        Dataset<LabeledExample> source = Dataset.of(
                new LabeledExample("cat-1", "cat"),
                new LabeledExample("cat-2", "cat"),
                new LabeledExample("cat-3", "cat"),
                new LabeledExample("cat-4", "cat"),
                new LabeledExample("dog-1", "dog"),
                new LabeledExample("dog-2", "dog"),
                new LabeledExample("dog-3", "dog"),
                new LabeledExample("dog-4", "dog"));

        List<Dataset.Fold<LabeledExample>> folds = source.repeatedStratifiedKFold(LabeledExample::label, 2, 2, 57L);

        assertEquals(4, folds.size());
        for (Dataset.Fold<LabeledExample> fold : folds) {
            assertEquals(4, fold.foldCount());
            assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(fold.validation()));
        }
        assertLabeledValidationCovers(source, folds.subList(0, 2));
        assertLabeledValidationCovers(source, folds.subList(2, 4));
        assertThrows(
                IllegalArgumentException.class,
                () -> Dataset.of(
                                new LabeledExample("cat-1", "cat"),
                                new LabeledExample("cat-2", "cat"),
                                new LabeledExample("dog-1", "dog"))
                        .stratifiedKFold(LabeledExample::label, 2, 1L));
    }

    @Test
    void datasetGroupSplitKeepsGroupsIsolatedAndFeedsGenericLoader() {
        Dataset<GroupedExample> source = groupedExamples(4, 2);
        Dataset.Split<GroupedExample> first = source.groupSplit(GroupedExample::group, 0.5, 41L);
        Dataset.Split<GroupedExample> second = source.groupSplit(GroupedExample::group, 0.5, 41L);

        assertNoGroupLeak(first.train(), first.validation());
        assertEquals(source.size(), first.train().size() + first.validation().size());
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(
                first.validation().toList(),
                flattenGeneric(DataLoader.builder(first.validation()).batchSize(2).build()));
    }

    @Test
    void datasetGroupThreeWaySplitKeepsGroupsIsolated() {
        Dataset<GroupedExample> source = groupedExamples(6, 2);
        Dataset.ThreeWaySplit<GroupedExample> first = source.groupSplit(GroupedExample::group, 0.5, 0.25, 43L);
        Dataset.ThreeWaySplit<GroupedExample> second = source.groupSplit(GroupedExample::group, 0.5, 0.25, 43L);

        assertTrue(Collections.disjoint(groupIds(first.train()), groupIds(first.validation())));
        assertTrue(Collections.disjoint(groupIds(first.train()), groupIds(first.test())));
        assertTrue(Collections.disjoint(groupIds(first.validation()), groupIds(first.test())));
        assertEquals(source.size(), first.train().size() + first.validation().size() + first.test().size());
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());
        assertEquals(first.test().toList(), second.test().toList());
    }

    @Test
    void datasetGroupKFoldKeepsGroupsOutOfBothTrainAndValidation() {
        Dataset<GroupedExample> source = groupedExamples(4, 2);
        List<Dataset.Fold<GroupedExample>> first = source.groupKFold(GroupedExample::group, 4, 47L);
        List<Dataset.Fold<GroupedExample>> second = source.groupKFold(GroupedExample::group, 4, 47L);

        assertEquals(4, first.size());
        for (Dataset.Fold<GroupedExample> fold : first) {
            assertNoGroupLeak(fold.train(), fold.validation());
            assertFalse(fold.validation().toList().isEmpty());
        }
        assertEquals(first.stream().map(fold -> fold.validation().toList()).toList(),
                second.stream().map(fold -> fold.validation().toList()).toList());
        assertGroupedValidationCovers(source, first);
    }

    @Test
    void datasetRepeatedGroupKFoldRunsLeakageSafePasses() {
        Dataset<GroupedExample> source = groupedExamples(4, 2);
        List<Dataset.Fold<GroupedExample>> folds = source.repeatedGroupKFold(GroupedExample::group, 2, 2, 49L);

        assertEquals(4, folds.size());
        for (Dataset.Fold<GroupedExample> fold : folds) {
            assertEquals(4, fold.foldCount());
            assertNoGroupLeak(fold.train(), fold.validation());
        }
        assertGroupedValidationCovers(source, folds.subList(0, 2));
        assertGroupedValidationCovers(source, folds.subList(2, 4));
        assertThrows(IllegalArgumentException.class, () -> groupedExamples(1, 2).groupSplit(GroupedExample::group, 0.5, 1L));
        assertThrows(IllegalArgumentException.class, () -> groupedExamples(2, 2).groupSplit(
                GroupedExample::group, 0.5, 0.25, 1L));
        assertThrows(IllegalArgumentException.class, () -> groupedExamples(2, 2).groupKFold(
                GroupedExample::group, 3, 1L));
    }

    @Test
    void datasetTimeSeriesSplitWalksForwardWithoutFutureLeakage() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        List<Dataset.Fold<Integer>> folds = source.timeSeriesSplit(3, 2, 1, 4);

        assertEquals(3, folds.size());
        assertEquals(List.of(2, 3, 4, 5), folds.get(0).train().toList());
        assertEquals(List.of(7, 8), folds.get(0).validation().toList());
        assertEquals(List.of(4, 5, 6, 7), folds.get(1).train().toList());
        assertEquals(List.of(9, 10), folds.get(1).validation().toList());
        assertEquals(List.of(6, 7, 8, 9), folds.get(2).train().toList());
        assertEquals(List.of(11, 12), folds.get(2).validation().toList());
        for (Dataset.Fold<Integer> fold : folds) {
            assertTrue(Collections.max(fold.train().toList()) < Collections.min(fold.validation().toList()));
        }
        assertEquals(
                folds.get(0).validation().toList(),
                flattenGeneric(DataLoader.builder(folds.get(0).validation()).batchSize(2).build()));
    }

    @Test
    void datasetTimeSeriesSplitDefaultUsesExpandingTrainingWindows() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        List<Dataset.Fold<Integer>> folds = source.timeSeriesSplit(3);

        assertEquals(3, folds.size());
        assertEquals(List.of(1, 2, 3, 4), folds.get(0).train().toList());
        assertEquals(List.of(5, 6), folds.get(0).validation().toList());
        assertEquals(List.of(1, 2, 3, 4, 5, 6), folds.get(1).train().toList());
        assertEquals(List.of(7, 8), folds.get(1).validation().toList());
        assertEquals(List.of(1, 2, 3, 4, 5, 6, 7, 8), folds.get(2).train().toList());
        assertEquals(List.of(9, 10), folds.get(2).validation().toList());

        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(0));
        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(2, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(2, 1, 8, 0));
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(1, 2).timeSeriesSplit(2));
    }

    @Test
    void datasetWindowedBuildsForecastingExamplesAndFeedsGenericLoader() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8);
        Dataset<Dataset.Window<Integer>> windows = source.windowed(3, 2, 2);

        assertEquals(2, windows.size());
        assertEquals(List.of(1, 2, 3), windows.get(0).inputs());
        assertEquals(List.of(4, 5), windows.get(0).targets());
        assertEquals(List.of(3, 4, 5), windows.get(1).inputs());
        assertEquals(List.of(6, 7), windows.get(1).targets());
        assertThrows(UnsupportedOperationException.class, () -> windows.get(0).inputs().add(99));
        assertEquals(
                windows.toList(),
                flattenGeneric(DataLoader.builder(windows).batchSize(2).build()));
    }

    @Test
    void datasetWindowedSupportsOneStepForecastsAndValidatesConfig() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6);
        Dataset<Dataset.Window<Integer>> windows = source.windowed(3);

        assertEquals(3, windows.size());
        assertEquals(List.of(1, 2, 3), windows.get(0).inputs());
        assertEquals(List.of(4), windows.get(0).targets());
        assertEquals(List.of(3, 4, 5), windows.get(2).inputs());
        assertEquals(List.of(6), windows.get(2).targets());
        assertEquals(0, Dataset.of(1, 2, 3).windowed(3, 1).size());
        assertThrows(IndexOutOfBoundsException.class, () -> windows.get(3));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(0));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(2, 0));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(2, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new Dataset.Window<>(List.of(), List.of(1)));
    }

    @Test
    void datasetZipEnumerateAndShardBuildSupervisedViews() {
        Dataset<String> inputs = Dataset.of("red", "green", "blue", "yellow");
        Dataset<Integer> labels = Dataset.of(0, 1, 0, 1);
        Dataset<Dataset.Pair<String, Integer>> pairs = inputs.zip(labels);

        assertEquals(4, pairs.size());
        assertEquals(new Dataset.Pair<>("green", 1), pairs.get(1));
        assertEquals(pairs.toList(), Dataset.zip(inputs, labels).toList());
        assertEquals(
                pairs.toList(),
                flattenGeneric(DataLoader.builder(pairs).batchSize(2).build()));

        Dataset<Dataset.Indexed<Dataset.Pair<String, Integer>>> indexed = pairs.enumerate();
        assertEquals(2, indexed.get(2).index());
        assertEquals(new Dataset.Pair<>("blue", 0), indexed.get(2).value());
        assertThrows(IllegalArgumentException.class, () -> new Dataset.Indexed<>(-1, "bad"));
        assertThrows(IllegalArgumentException.class, () -> inputs.zip(Dataset.of(0, 1)));
        assertThrows(NullPointerException.class, () -> inputs.zip(null));
    }

    @Test
    void datasetShardCreatesDeterministicRoundRobinPartitions() {
        Dataset<Integer> source = Dataset.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        assertEquals(List.of(0, 3, 6, 9), source.shard(3, 0).toList());
        assertEquals(List.of(1, 4, 7), source.shard(3, 1).toList());
        assertEquals(List.of(2, 5, 8), source.shard(3, 2).toList());
        assertEquals(List.of(7), source.shard(8, 7).toList());
        assertEquals(List.of(), source.shard(12, 11).toList());
        assertEquals(
                List.of(1, 4, 7),
                flattenGeneric(DataLoader.builder(source.shard(3, 1)).batchSize(2).build()));

        assertThrows(IndexOutOfBoundsException.class, () -> source.shard(3, 1).get(3));
        assertThrows(IllegalArgumentException.class, () -> source.shard(0, 0));
        assertThrows(IllegalArgumentException.class, () -> source.shard(2, -1));
        assertThrows(IllegalArgumentException.class, () -> source.shard(2, 2));
    }

    @Test
    void datasetTakeDropSliceAndRepeatComposeWithoutCopies() {
        Dataset<String> source = Dataset.of("a", "b", "c", "d", "e");

        assertEquals(List.of("a", "b", "c"), source.take(3).toList());
        assertEquals(source.toList(), source.take(99).toList());
        assertEquals(List.of("c", "d", "e"), source.drop(2).toList());
        assertEquals(List.of(), source.drop(99).toList());
        assertEquals(List.of("b", "c", "d"), source.slice(1, 4).toList());
        assertEquals(List.of(), source.slice(2, 2).toList());
        assertEquals(List.of("a", "b", "c", "a", "b", "c"), source.take(3).repeat(2).toList());
        assertEquals(
                List.of("b", "c", "d", "b", "c", "d"),
                flattenGeneric(DataLoader.builder(source.slice(1, 4).repeat(2)).batchSize(4).build()));

        assertThrows(IndexOutOfBoundsException.class, () -> source.take(2).get(2));
        assertThrows(IllegalArgumentException.class, () -> source.take(-1));
        assertThrows(IllegalArgumentException.class, () -> source.drop(-1));
        assertThrows(IllegalArgumentException.class, () -> source.slice(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> source.slice(3, 2));
        assertThrows(IllegalArgumentException.class, () -> source.slice(0, 6));
        assertThrows(IllegalArgumentException.class, () -> source.repeat(-1));
        assertThrows(IllegalArgumentException.class, () -> hugeDataset().repeat(2));
    }

    @Test
    void datasetCacheMemoizesExpensiveTransformsAcrossLoaderPasses() {
        AtomicInteger calls = new AtomicInteger();
        Dataset<Integer> cached = Dataset.of(1, 2, 3)
                .map(value -> {
                    calls.incrementAndGet();
                    return value * 10;
                })
                .cache();

        assertEquals(3, cached.size());
        assertEquals(10, cached.get(0));
        assertEquals(10, cached.get(0));
        assertEquals(1, calls.get());

        assertEquals(List.of(10, 20, 30), flattenGeneric(DataLoader.builder(cached).batchSize(2).build()));
        assertEquals(3, calls.get());
        assertEquals(List.of(10, 20, 30), flattenGeneric(DataLoader.builder(cached).batchSize(2).build()));
        assertEquals(3, calls.get());
        assertEquals(List.of(), Dataset.<Integer>of().cache().toList());
        assertThrows(IndexOutOfBoundsException.class, () -> cached.get(3));
    }

    @Test
    void datasetShuffleAndSamplingViewsAreDeterministicAndLoaderReady() {
        Dataset<Integer> source = Dataset.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        Dataset<Integer> shuffled = source.shuffle(42L);
        Dataset<Integer> sampled = source.sample(4, 42L);
        Dataset<Integer> bootstrap = source.take(3).sampleWithReplacement(8, 7L);

        assertEquals(source.size(), shuffled.size());
        assertEquals(shuffled.toList(), source.shuffle(42L).toList());
        assertEquals(new HashSet<>(source.toList()), new HashSet<>(shuffled.toList()));
        assertEquals(4, sampled.size());
        assertEquals(sampled.toList(), source.sample(4, 42L).toList());
        assertEquals(4, new HashSet<>(sampled.toList()).size());
        assertEquals(shuffled.take(4).toList(), sampled.toList());
        assertEquals(
                sampled.toList(),
                flattenGeneric(DataLoader.builder(sampled).batchSize(3).build()));

        assertEquals(8, bootstrap.size());
        assertEquals(bootstrap.toList(), source.take(3).sampleWithReplacement(8, 7L).toList());
        assertTrue(bootstrap.toList().stream().allMatch(value -> value >= 0 && value <= 2));
        assertTrue(new HashSet<>(bootstrap.toList()).size() < bootstrap.size());
        assertEquals(List.of(), source.sample(0, 1L).toList());
        assertEquals(List.of(), source.sampleWithReplacement(0, 1L).toList());

        assertThrows(IndexOutOfBoundsException.class, () -> sampled.get(4));
        assertThrows(IllegalArgumentException.class, () -> source.sample(-1, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.sample(11, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.sampleWithReplacement(-1, 1L));
        assertThrows(IllegalArgumentException.class, () -> Dataset.<Integer>of().sampleWithReplacement(1, 1L));
    }

    @Test
    void tensorLoaderSeedMakesShuffleDeterministic() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1));
        DataLoader.TensorDataLoader first = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(42L)
                .build();
        DataLoader.TensorDataLoader second = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(42L)
                .build();
        DataLoader.TensorDataLoader shorthand = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(42L)
                .build();

        assertEquals(flattenInputs(first), flattenInputs(second));
        assertEquals(flattenInputs(first), flattenInputs(shorthand));
        assertTrue(shorthand.shuffle());
    }

    @Test
    void tensorDatasetSplitIsDeterministicAndCoversAllSamples() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1));

        DataLoader.TensorDatasetSplit first = dataset.split(0.6, 7L);
        DataLoader.TensorDatasetSplit second = dataset.split(0.6, 7L);

        assertEquals(3, first.train().size());
        assertEquals(2, first.validation().size());
        assertEquals(flattenDataset(first.train()), flattenDataset(second.train()));
        assertEquals(flattenDataset(first.validation()), flattenDataset(second.validation()));
    }

    @Test
    void tensorDatasetSplitBuildsReadyToTrainLoaders() {
        DataLoader.TensorDatasetSplit split = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                11L);

        DataLoader.TensorDataLoader train = split.trainLoader(2, true, 99L);
        DataLoader.TensorDataLoader validation = split.validationLoader(2);

        assertEquals(4, train.size());
        assertEquals(2, validation.size());
        assertEquals(2, train.numBatches());
        assertEquals(1, validation.numBatches());
        assertEquals(flattenInputs(train), flattenInputs(split.trainLoader(2, true, 99L)));
    }

    @Test
    void tensorDatasetThreeWaySplitIsDeterministicAndBuildsReadyToTrainLoaders() {
        DataLoader.TensorDatasetThreeWaySplit first = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f}, 10, 1),
                0.6,
                0.2,
                77L);
        DataLoader.TensorDatasetThreeWaySplit second = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f}, 10, 1),
                0.6,
                0.2,
                77L);

        assertEquals(6, first.train().size());
        assertEquals(2, first.validation().size());
        assertEquals(2, first.test().size());
        assertEquals(flattenDataset(first.train()), flattenDataset(second.train()));
        assertEquals(flattenDataset(first.validation()), flattenDataset(second.validation()));
        assertEquals(flattenDataset(first.test()), flattenDataset(second.test()));
        assertEquals(3, first.trainLoader(2).numBatches());
        assertEquals(1, first.validationLoader(2).numBatches());
        assertEquals(1, first.testLoader(2).numBatches());
        assertEquals(6, first.trainValidation().train().size());
        assertEquals(2, first.trainValidation().validation().size());
    }

    @Test
    void tensorDatasetKFoldIsDeterministicBalancedAndBuildsReadyToTrainLoaders() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1));

        List<DataLoader.TensorDatasetFold> first = dataset.kFold(3, 42L);
        List<DataLoader.TensorDatasetFold> second = DataLoader.kFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                3,
                42L);

        assertEquals(3, first.size());
        assertEquals(List.of(4, 3, 3), first.stream().map(fold -> fold.validation().size()).toList());
        assertEquals(List.of(6, 7, 7), first.stream().map(fold -> fold.train().size()).toList());
        assertEquals(flattenDataset(first.get(0).validation()), flattenDataset(second.get(0).validation()));
        assertEquals(3, first.get(0).foldCount());
        assertEquals(0, first.get(0).foldIndex());
        assertEquals(3, first.get(0).trainLoader(2).numBatches());
        assertEquals(2, first.get(0).validationLoader(2).numBatches());

        List<Float> coveredValidationRows = new ArrayList<>();
        for (DataLoader.TensorDatasetFold fold : first) {
            coveredValidationRows.addAll(flattenDataset(fold.validation()));
        }
        Collections.sort(coveredValidationRows);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f), coveredValidationRows);
    }

    @Test
    void repeatedKFoldRunsMultipleDeterministicFullValidationPasses() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1));

        List<DataLoader.TensorDatasetFold> first = dataset.repeatedKFold(3, 2, 42L);
        List<DataLoader.TensorDatasetFold> second = DataLoader.repeatedKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                3,
                2,
                42L);

        assertEquals(6, first.size());
        assertEquals(flattenDataset(first.get(0).validation()), flattenDataset(second.get(0).validation()));
        assertEquals(6, first.get(0).foldCount());
        assertEquals(List.of(0, 1, 2, 3, 4, 5), first.stream().map(DataLoader.TensorDatasetFold::foldIndex).toList());
        for (int repeat = 0; repeat < 2; repeat++) {
            List<Float> coveredValidationRows = new ArrayList<>();
            for (int fold = 0; fold < 3; fold++) {
                DataLoader.TensorDatasetFold repeatedFold = first.get(repeat * 3 + fold);
                assertEquals(4, repeatedFold.train().size());
                assertEquals(2, repeatedFold.validation().size());
                coveredValidationRows.addAll(flattenDataset(repeatedFold.validation()));
            }
            Collections.sort(coveredValidationRows);
            assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f), coveredValidationRows);
        }
    }

    @Test
    void groupKFoldKeepsGroupsOutOfBothTrainAndValidation() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1));
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50};

        List<DataLoader.TensorDatasetFold> first = dataset.groupKFold(groups, 3, 42L);
        List<DataLoader.TensorDatasetFold> second = DataLoader.groupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                groups,
                3,
                42L);

        assertEquals(3, first.size());
        assertEquals(List.of(4, 4, 2), first.stream().map(fold -> fold.validation().size()).toList());
        assertEquals(flattenDataset(first.get(0).validation()), flattenDataset(second.get(0).validation()));

        List<Float> coveredValidationRows = new ArrayList<>();
        for (DataLoader.TensorDatasetFold fold : first) {
            Set<Integer> trainGroups = groupsForRows(flattenDataset(fold.train()), groups);
            Set<Integer> validationGroups = groupsForRows(flattenDataset(fold.validation()), groups);
            for (int group : validationGroups) {
                assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
            }
            coveredValidationRows.addAll(flattenDataset(fold.validation()));
        }
        Collections.sort(coveredValidationRows);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f), coveredValidationRows);
    }

    @Test
    void classificationStratifiedGroupKFoldBalancesLabelsWithoutLeakingGroups() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        List<DataLoader.TensorDatasetFold> folds = DataLoader.classificationStratifiedGroupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                labels,
                groups,
                3,
                2026L);

        assertEquals(3, folds.size());
        List<Float> coveredValidationRows = new ArrayList<>();
        for (DataLoader.TensorDatasetFold fold : folds) {
            List<Float> validationLabels = flattenLabels(fold.validation());
            Set<Integer> trainGroups = groupsForRows(flattenDataset(fold.train()), groups);
            Set<Integer> validationGroups = groupsForRows(flattenDataset(fold.validation()), groups);
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertEquals(2, Collections.frequency(validationLabels, 0f));
            assertEquals(2, Collections.frequency(validationLabels, 1f));
            for (int group : validationGroups) {
                assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
            }
            coveredValidationRows.addAll(flattenDataset(fold.validation()));
        }
        Collections.sort(coveredValidationRows);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), coveredValidationRows);
    }

    @Test
    void classificationRepeatedStratifiedKFoldPreservesClassDistributionInEveryRepeat() {
        List<DataLoader.TensorDatasetFold> folds = DataLoader.classificationRepeatedStratifiedKFold(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0.8f, 0.2f,
                        0.7f, 0.3f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.2f, 0.8f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                2,
                3,
                2026L);

        assertEquals(6, folds.size());
        assertEquals(6, folds.get(0).foldCount());
        for (DataLoader.TensorDatasetFold fold : folds) {
            List<Float> validationLabels = flattenLabels(fold.validation());
            assertEquals(4, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertEquals(2, Collections.frequency(validationLabels, 0f));
            assertEquals(2, Collections.frequency(validationLabels, 1f));
        }
    }

    @Test
    void timeSeriesSplitWalksForwardWithoutFutureLeakage() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1));

        List<DataLoader.TensorDatasetFold> folds = dataset.timeSeriesSplit(3, 2, 1, 4);

        assertEquals(3, folds.size());
        assertEquals(List.of(2f, 3f, 4f, 5f), flattenDataset(folds.get(0).train()));
        assertEquals(List.of(7f, 8f), flattenDataset(folds.get(0).validation()));
        assertEquals(List.of(4f, 5f, 6f, 7f), flattenDataset(folds.get(1).train()));
        assertEquals(List.of(9f, 10f), flattenDataset(folds.get(1).validation()));
        assertEquals(List.of(6f, 7f, 8f, 9f), flattenDataset(folds.get(2).train()));
        assertEquals(List.of(11f, 12f), flattenDataset(folds.get(2).validation()));
        for (DataLoader.TensorDatasetFold fold : folds) {
            float latestTrain = Collections.max(flattenDataset(fold.train()));
            float earliestValidation = Collections.min(flattenDataset(fold.validation()));
            assertTrue(latestTrain < earliestValidation, "validation contains past rows for fold " + fold.foldIndex());
            assertEquals(1, fold.validationLoader(2).numBatches());
        }
    }

    @Test
    void timeSeriesSplitDefaultUsesExpandingTrainingWindows() {
        List<DataLoader.TensorDatasetFold> folds = DataLoader.timeSeriesSplit(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f}, 10, 1),
                3);

        assertEquals(3, folds.size());
        assertEquals(List.of(1f, 2f, 3f, 4f), flattenDataset(folds.get(0).train()));
        assertEquals(List.of(5f, 6f), flattenDataset(folds.get(0).validation()));
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f), flattenDataset(folds.get(1).train()));
        assertEquals(List.of(7f, 8f), flattenDataset(folds.get(1).validation()));
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), flattenDataset(folds.get(2).train()));
        assertEquals(List.of(9f, 10f), flattenDataset(folds.get(2).validation()));
    }

    @Test
    void classificationStratifiedSplitPreservesLabelsOnBothSides() {
        DataLoader.TensorDatasetSplit first = DataLoader.classificationStratifiedSplit(
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
        DataLoader.TensorDatasetSplit second = DataLoader.classificationStratifiedSplit(
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
    void classificationStratifiedThreeWaySplitPreservesLabelsAcrossHoldouts() {
        DataLoader.TensorDatasetThreeWaySplit split = DataLoader.classificationStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0.8f, 0.2f,
                        0.7f, 0.3f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.2f, 0.8f,
                        0.3f, 0.7f,
                        1f, 1f,
                        0.9f, 0.9f,
                        0.8f, 0.8f,
                        0.7f, 0.7f
                }, 12, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
                0.5,
                0.25,
                2026L);

        assertEquals(6, split.train().size());
        assertEquals(3, split.validation().size());
        assertEquals(3, split.test().size());
        assertEquals(2, Collections.frequency(flattenLabels(split.train()), 0f));
        assertEquals(2, Collections.frequency(flattenLabels(split.train()), 1f));
        assertEquals(2, Collections.frequency(flattenLabels(split.train()), 2f));
        assertEquals(1, Collections.frequency(flattenLabels(split.validation()), 0f));
        assertEquals(1, Collections.frequency(flattenLabels(split.validation()), 1f));
        assertEquals(1, Collections.frequency(flattenLabels(split.validation()), 2f));
        assertEquals(1, Collections.frequency(flattenLabels(split.test()), 0f));
        assertEquals(1, Collections.frequency(flattenLabels(split.test()), 1f));
        assertEquals(1, Collections.frequency(flattenLabels(split.test()), 2f));
    }

    @Test
    void classificationStratifiedKFoldPreservesClassDistributionInEachValidationFold() {
        List<DataLoader.TensorDatasetFold> folds = DataLoader.classificationStratifiedKFold(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0.8f, 0.2f,
                        0.7f, 0.3f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.2f, 0.8f,
                        0.3f, 0.7f,
                        1f, 1f,
                        0.9f, 0.9f,
                        0.8f, 0.8f,
                        0.7f, 0.7f
                }, 12, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
                4,
                2026L);

        assertEquals(4, folds.size());
        for (DataLoader.TensorDatasetFold fold : folds) {
            List<Float> validationLabels = flattenLabels(fold.validation());
            assertEquals(3, fold.validation().size());
            assertEquals(9, fold.train().size());
            assertEquals(1, Collections.frequency(validationLabels, 0f));
            assertEquals(1, Collections.frequency(validationLabels, 1f));
            assertEquals(1, Collections.frequency(validationLabels, 2f));
        }
    }

    @Test
    void binaryStratifiedSplitPreservesLabelsOnBothSides() {
        DataLoader.TensorDatasetSplit split = DataLoader.binaryStratifiedSplit(
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
    void binaryStratifiedThreeWaySplitPreservesLabelsAcrossHoldouts() {
        DataLoader.TensorDatasetThreeWaySplit split = DataLoader.binaryStratifiedSplit(
                GradTensor.of(new float[] {
                        -1f, -1f,
                        -0.8f, -0.7f,
                        -1.2f, -0.5f,
                        -0.6f, -1.1f,
                        -0.9f, -1.2f,
                        -1.1f, -0.8f,
                        1f, 1f,
                        0.9f, 0.7f,
                        1.2f, 0.5f,
                        0.6f, 1.1f,
                        0.8f, 1.2f,
                        1.1f, 0.8f
                }, 12, 2),
                new int[] {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
                0.5,
                0.25,
                99L);

        assertEquals(6, split.train().size());
        assertEquals(4, split.validation().size());
        assertEquals(2, split.test().size());
        assertEquals(3, Collections.frequency(flattenLabels(split.train()), 0f));
        assertEquals(3, Collections.frequency(flattenLabels(split.train()), 1f));
        assertEquals(2, Collections.frequency(flattenLabels(split.validation()), 0f));
        assertEquals(2, Collections.frequency(flattenLabels(split.validation()), 1f));
        assertEquals(1, Collections.frequency(flattenLabels(split.test()), 0f));
        assertEquals(1, Collections.frequency(flattenLabels(split.test()), 1f));
    }

    @Test
    void binaryStratifiedKFoldPreservesLabelsInEachValidationFold() {
        List<DataLoader.TensorDatasetFold> folds = DataLoader.binaryStratifiedKFold(
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
                4,
                99L);

        assertEquals(4, folds.size());
        for (DataLoader.TensorDatasetFold fold : folds) {
            List<Float> validationLabels = flattenLabels(fold.validation());
            assertEquals(2, fold.validation().size());
            assertEquals(6, fold.train().size());
            assertEquals(1, Collections.frequency(validationLabels, 0f));
            assertEquals(1, Collections.frequency(validationLabels, 1f));
        }
    }

    @Test
    void positiveWeightHelpersDeriveBceImbalanceWeights() {
        assertEquals(3.0f, DataLoader.binaryPositiveWeight(1, 0, 0, 0), 1e-6f);
        assertEquals(1.0f, DataLoader.binaryPositiveWeight(true, false), 1e-6f);
        assertArrayEquals(new float[] {3.0f, 3.0f, 1.0f}, DataLoader.multiLabelPositiveWeights(new int[][] {
                {1, 0, 1},
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        }), 1e-6f);
        assertArrayEquals(new float[] {
                2.0f / 3.0f,
                2.0f / 3.0f,
                2.0f / 3.0f,
                2.0f,
                2.0f,
                2.0f / 3.0f
        }, DataLoader.multiLabelBalancedSampleWeights(new int[][] {
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 0}
        }), 1e-6f);
        assertArrayEquals(new float[] {1.0f, 1.0f}, DataLoader.multiLabelBalancedSampleWeights(new boolean[][] {
                {false, false},
                {false, false}
        }), 1e-6f);
    }

    @Test
    void classWeightHelpersDeriveCrossEntropyImbalanceWeights() {
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
    void multiLabelBinaryStratifiedSplitBalancesPerLabelPositives() {
        DataLoader.TensorDatasetSplit split = DataLoader.multiLabelBinaryStratifiedSplit(
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
    void multiLabelBinaryStratifiedThreeWaySplitBalancesPerLabelPositivesAcrossHoldouts() {
        DataLoader.TensorDatasetThreeWaySplit split = DataLoader.multiLabelBinaryStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.9f, 0.9f,
                        1f, 0.5f,
                        0.9f, 0.4f,
                        0.5f, 1f,
                        0.4f, 0.9f,
                        0.6f, 0.6f,
                        0.7f, 0.7f
                }, 12, 2),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 1},
                        {0, 1, 1}
                },
                0.5,
                0.25,
                456L);

        assertEquals(6, split.train().size());
        assertEquals(3, split.validation().size());
        assertEquals(3, split.test().size());
        int[] trainCounts = positiveCounts(split.train(), 3);
        int[] validationCounts = positiveCounts(split.validation(), 3);
        int[] testCounts = positiveCounts(split.test(), 3);
        assertCountsWithin(new int[] {3, 3, 3}, trainCounts, 1);
        assertCountsWithin(new int[] {2, 2, 2}, validationCounts, 1);
        assertCountsWithin(new int[] {1, 1, 1}, testCounts, 1);
    }

    @Test
    void multiLabelBinaryStratifiedGroupSplitKeepsGroupsLeakageSafeAndBalanced() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        DataLoader.TensorDatasetSplit split = DataLoader.multiLabelBinaryStratifiedGroupSplit(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                balancedGroupedMultiLabelRows(),
                groups,
                0.5,
                2028L);

        assertEquals(6, split.train().size());
        assertEquals(6, split.validation().size());
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.validation(), 3));
        Set<Integer> trainGroups = groupsForRows(flattenDataset(split.train()), groups);
        Set<Integer> validationGroups = groupsForRows(flattenDataset(split.validation()), groups);
        for (int group : validationGroups) {
            assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
        }
    }

    @Test
    void multiLabelBinaryStratifiedGroupThreeWaySplitKeepsGroupsLeakageSafeAndBalanced() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        DataLoader.TensorDatasetThreeWaySplit split = DataLoader.multiLabelBinaryStratifiedGroupSplit(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                balancedGroupedMultiLabelRows(),
                groups,
                0.5,
                1.0 / 6.0,
                2028L);

        assertEquals(6, split.train().size());
        assertEquals(2, split.validation().size());
        assertEquals(4, split.test().size());
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {1, 1, 1}, positiveCounts(split.validation(), 3));
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.test(), 3));
        Set<Integer> trainGroups = groupsForRows(flattenDataset(split.train()), groups);
        Set<Integer> validationGroups = groupsForRows(flattenDataset(split.validation()), groups);
        Set<Integer> testGroups = groupsForRows(flattenDataset(split.test()), groups);
        for (int group : validationGroups) {
            assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
        }
        for (int group : testGroups) {
            assertTrue(!trainGroups.contains(group), "group leaked into train and test: " + group);
            assertTrue(!validationGroups.contains(group), "group leaked into validation and test: " + group);
        }
    }

    @Test
    void multiLabelBinaryStratifiedKFoldBalancesPerLabelPositivesInEveryValidationFold() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 1},
                {1, 1, 0},
                {1, 1, 0},
                {1, 0, 1},
                {1, 0, 1},
                {0, 1, 1},
                {0, 1, 1}
        };
        List<DataLoader.TensorDatasetFold> folds = DataLoader.multiLabelBinaryStratifiedKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                labels,
                3,
                2026L);

        assertEquals(3, folds.size());
        List<Float> coveredValidationRows = new ArrayList<>();
        for (DataLoader.TensorDatasetFold fold : folds) {
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertArrayEquals(new int[] {4, 4, 4}, positiveCounts(fold.train(), 3));
            assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(fold.validation(), 3));
            coveredValidationRows.addAll(flattenDataset(fold.validation()));
        }
        Collections.sort(coveredValidationRows);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), coveredValidationRows);
    }

    @Test
    void multiLabelBinaryRepeatedStratifiedKFoldRunsFullBalancedPasses() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 1},
                {1, 1, 0},
                {1, 1, 0},
                {1, 0, 1},
                {1, 0, 1},
                {0, 1, 1},
                {0, 1, 1}
        };
        List<DataLoader.TensorDatasetFold> folds = DataLoader.multiLabelBinaryRepeatedStratifiedKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                labels,
                3,
                2,
                2026L);

        assertEquals(6, folds.size());
        assertEquals(6, folds.get(0).foldCount());
        for (int repeat = 0; repeat < 2; repeat++) {
            List<Float> coveredValidationRows = new ArrayList<>();
            for (int fold = 0; fold < 3; fold++) {
                DataLoader.TensorDatasetFold repeatedFold = folds.get(repeat * 3 + fold);
                assertEquals(8, repeatedFold.train().size());
                assertEquals(4, repeatedFold.validation().size());
                assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(repeatedFold.validation(), 3));
                coveredValidationRows.addAll(flattenDataset(repeatedFold.validation()));
            }
            Collections.sort(coveredValidationRows);
            assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), coveredValidationRows);
        }
    }

    @Test
    void multiLabelBinaryStratifiedGroupKFoldBalancesLabelsWithoutLeakingGroups() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 1},
                {1, 1, 0},
                {1, 1, 0},
                {1, 0, 1},
                {1, 0, 1},
                {0, 1, 1},
                {0, 1, 1}
        };
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        List<DataLoader.TensorDatasetFold> folds = DataLoader.multiLabelBinaryStratifiedGroupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                labels,
                groups,
                3,
                2027L);

        assertEquals(3, folds.size());
        List<Float> coveredValidationRows = new ArrayList<>();
        for (DataLoader.TensorDatasetFold fold : folds) {
            Set<Integer> trainGroups = groupsForRows(flattenDataset(fold.train()), groups);
            Set<Integer> validationGroups = groupsForRows(flattenDataset(fold.validation()), groups);
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(fold.validation(), 3));
            for (int group : validationGroups) {
                assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
            }
            coveredValidationRows.addAll(flattenDataset(fold.validation()));
        }
        Collections.sort(coveredValidationRows);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), coveredValidationRows);
    }

    @Test
    void multiLabelBinaryRepeatedStratifiedGroupKFoldRunsFullLeakageSafePasses() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 1},
                {1, 1, 0},
                {1, 1, 0},
                {1, 0, 1},
                {1, 0, 1},
                {0, 1, 1},
                {0, 1, 1}
        };
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        List<DataLoader.TensorDatasetFold> folds = DataLoader.multiLabelBinaryRepeatedStratifiedGroupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f}, 12, 1),
                labels,
                groups,
                3,
                2,
                2027L);

        assertEquals(6, folds.size());
        assertEquals(6, folds.get(0).foldCount());
        for (int repeat = 0; repeat < 2; repeat++) {
            List<Float> coveredValidationRows = new ArrayList<>();
            for (int fold = 0; fold < 3; fold++) {
                DataLoader.TensorDatasetFold repeatedFold = folds.get(repeat * 3 + fold);
                Set<Integer> trainGroups = groupsForRows(flattenDataset(repeatedFold.train()), groups);
                Set<Integer> validationGroups = groupsForRows(flattenDataset(repeatedFold.validation()), groups);
                assertEquals(8, repeatedFold.train().size());
                assertEquals(4, repeatedFold.validation().size());
                assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(repeatedFold.validation(), 3));
                for (int group : validationGroups) {
                    assertTrue(!trainGroups.contains(group), "group leaked into train and validation: " + group);
                }
                coveredValidationRows.addAll(flattenDataset(repeatedFold.validation()));
            }
            Collections.sort(coveredValidationRows);
            assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), coveredValidationRows);
        }
    }

    @Test
    void weightedRandomSamplerWithReplacementFollowsPositiveWeightsDeterministically() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f, 20f, 30f, 40f}, 4, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 0f}, 4, 1));

        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(dataset)
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
    void weightedRandomSamplerWithoutReplacementDrawsUniquePositiveWeightSamples() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {0f, 1f, 1f, 0f}, 4, 1));

        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .weightedRandomSampler(new float[] {0f, 4f, 1f, 0f}, 2, false, 77L)
                .build();
        List<Float> drawn = flattenInputs(loader);
        Collections.sort(drawn);

        assertEquals(List.of(2f, 3f), drawn);
    }

    @Test
    void weightedRandomSamplerDefensivelyCopiesWeights() {
        float[] weights = new float[] {0f, 1f};
        WeightedRandomSampler sampler = DataLoader.weightedRandomSampler(weights, 1, true, 5L);
        weights[1] = 0f;

        assertEquals(List.of(1), sampler.sample(2));
        assertArrayEquals(new float[] {0f, 1f}, sampler.sampleWeights(), 1e-6f);
    }

    @Test
    void classWeightedClassificationLoaderUsesBalancedSampleWeightsDeterministically() {
        int[] labels = {0, 0, 0, 1};
        DataLoader.TensorDataLoader loader = DataLoader.classWeightedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                3,
                12,
                17L);

        assertTrue(loader.sampled());
        assertEquals(12, loader.sampleCount());
        assertEquals(4, loader.numBatches());
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f},
                DataLoader.classBalancedSampleWeights(labels),
                1e-6f);
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.classWeightedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                3,
                12,
                17L)));
        assertTrue(Collections.frequency(flattenInputs(loader), 4f) >= 1);
    }

    @Test
    void weightedBinaryLoaderUsesBalancedSampleWeightsDeterministically() {
        int[] labels = {0, 0, 0, 1};
        DataLoader.TensorDataLoader loader = DataLoader.weightedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                2,
                10,
                23L);

        assertTrue(loader.sampled());
        assertEquals(10, loader.sampleCount());
        assertEquals(5, loader.numBatches());
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.weightedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                2,
                10,
                23L)));
        assertTrue(flattenLabels(loader).contains(1f));
    }

    @Test
    void stratifiedBatchSamplerPreservesClassProportionsPerBatchDeterministically() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1};
        DataLoader.TensorDataLoader loader = DataLoader.stratifiedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                3,
                2029L);

        assertTrue(loader.sampled());
        assertEquals(12, loader.sampleCount());
        assertEquals(3, loader.numBatches());
        for (DataLoader.Batch batch : loader) {
            List<Float> batchLabels = flattenBatchLabels(batch);
            assertEquals(3, Collections.frequency(batchLabels, 0f));
            assertEquals(1, Collections.frequency(batchLabels, 1f));
        }
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.stratifiedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                3,
                2029L)));
        assertEquals(3, Collections.frequency(flattenLabels(loader), 1f));

        DataLoader.TensorDataLoader reorderedBuilder = DataLoader.tensorBuilder(DataLoader.classificationDataset(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                        labels))
                .stratifiedBatchSampler(labels, 2, 2030L)
                .batchSize(4)
                .build();
        for (DataLoader.Batch batch : reorderedBuilder) {
            List<Float> batchLabels = flattenBatchLabels(batch);
            assertEquals(3, Collections.frequency(batchLabels, 0f));
            assertEquals(1, Collections.frequency(batchLabels, 1f));
        }
    }

    @Test
    void stratifiedBatchSamplerWithoutReplacementKeepsRowsUnique() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1};
        DataLoader.TensorDataLoader loader = DataLoader.stratifiedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                2,
                false,
                2031L);
        List<Float> drawn = flattenInputs(loader);
        Collections.sort(drawn);

        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), drawn);
        assertArrayEquals(new int[] {6, 2}, DataLoader.stratifiedBatchSampler(
                labels, 4, 2, false, 2031L).classCounts());
        for (DataLoader.Batch batch : loader) {
            List<Float> batchLabels = flattenBatchLabels(batch);
            assertEquals(3, Collections.frequency(batchLabels, 0f));
            assertEquals(1, Collections.frequency(batchLabels, 1f));
        }
    }

    @Test
    void classificationDistributionReportSummarizesAndDefensivelyCopiesCounts() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1};
        DataLoader.ClassificationDistributionReport report = DataLoader.classificationDistribution(
                DataLoader.stratifiedClassification(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                        labels,
                        4,
                        2,
                        false,
                        2033L),
                2);

        assertEquals(8, report.sampleCount());
        assertEquals(2, report.batchCount());
        assertArrayEquals(new int[] {6, 2}, report.classCounts());
        assertArrayEquals(new double[] {0.75, 0.25}, report.classFractions(), 1e-6);
        assertArrayEquals(new int[] {3, 1}, report.batchClassCounts().get(0));
        assertArrayEquals(new int[] {3, 1}, report.batchClassCounts().get(1));
        assertEquals(6, report.classCount(0));
        assertEquals(0, report.majorityClassIndex());
        assertEquals(6, report.majorityClassCount());
        assertEquals(1, report.minorityClassIndex());
        assertEquals(2, report.minorityClassCount());
        assertEquals(0, report.missingClassCount());
        assertEquals(3.0, report.imbalanceRatio(), 1e-6);
        assertEquals(
                -((0.75 * Math.log(0.75)) + (0.25 * Math.log(0.25))) / Math.log(2.0),
                report.normalizedEntropy(),
                1e-6);
        assertArrayEquals(new float[] {2.0f / 3.0f, 2.0f}, report.balancedClassWeights(), 1e-6f);
        assertEquals(2.0f / 3.0f, report.balancedClassWeight(0), 1e-6f);

        DataLoader.ClassificationDistributionReport inferred = DataLoader.classificationDistribution(
                DataLoader.stratifiedClassification(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                        labels,
                        4,
                        2,
                        false,
                        2033L));
        assertEquals(2, inferred.numClasses());
        assertArrayEquals(report.classCounts(), inferred.classCounts());

        Map<String, Object> metadata = report.toMetadata("train.distribution");
        assertEquals(8, metadata.get("train.distribution.sampleCount"));
        assertEquals(2, metadata.get("train.distribution.batchCount"));
        assertEquals(2, metadata.get("train.distribution.numClasses"));
        assertEquals(List.of(6, 2), metadata.get("train.distribution.classCounts"));
        assertEquals(List.of(0.75, 0.25), metadata.get("train.distribution.classFractions"));
        assertEquals(List.of(List.of(3, 1), List.of(3, 1)), metadata.get("train.distribution.batchClassCounts"));
        assertEquals(0, metadata.get("train.distribution.majorityClassIndex"));
        assertEquals(6, metadata.get("train.distribution.majorityClassCount"));
        assertEquals(1, metadata.get("train.distribution.minorityClassIndex"));
        assertEquals(2, metadata.get("train.distribution.minorityClassCount"));
        assertEquals(0, metadata.get("train.distribution.missingClassCount"));
        assertEquals(3.0, (double) metadata.get("train.distribution.imbalanceRatio"), 1e-6);
        assertEquals(report.normalizedEntropy(), (double) metadata.get("train.distribution.normalizedEntropy"), 1e-6);
        assertEquals(List.of(2.0f / 3.0f, 2.0f), metadata.get("train.distribution.balancedClassWeights"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("extra", 1));

        DataLoader.ClassificationDistributionReport missing = DataLoader.classificationDistribution(
                DataLoader.classification(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1),
                        new int[] {0, 0, 0},
                        3),
                3);
        assertEquals(2, missing.missingClassCount());
        assertEquals(3.0, missing.imbalanceRatio(), 1e-6);

        DataLoader.ClassificationDistributionReport candidate = DataLoader.classificationDistribution(
                DataLoader.classification(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                        new int[] {0, 1, 1, 1},
                        2),
                2);
        DataLoader.ClassificationDistributionDriftReport drift = report.driftTo(candidate);
        assertArrayEquals(new double[] {0.75, 0.25}, drift.referenceFractions(), 1e-6);
        assertArrayEquals(new double[] {0.25, 0.75}, drift.candidateFractions(), 1e-6);
        assertArrayEquals(new double[] {-0.5, 0.5}, drift.fractionDeltas(), 1e-6);
        assertEquals(0.5, drift.totalVariationDistance(), 1e-6);
        assertEquals(0, drift.maxDeltaClassIndex());
        assertEquals(0.5, drift.maxAbsoluteFractionDelta(), 1e-6);
        assertEquals(List.of(), drift.referenceMissingClasses());
        assertEquals(List.of(), drift.candidateMissingClasses());
        assertEquals(0.5, (double) drift.toMetadata("drift").get("drift.totalVariationDistance"), 1e-6);

        int[] mutableCounts = report.classCounts();
        mutableCounts[0] = 99;
        List<int[]> mutableBatches = report.batchClassCounts();
        mutableBatches.get(0)[0] = 99;

        assertArrayEquals(new int[] {6, 2}, report.classCounts());
        assertArrayEquals(new int[] {3, 1}, report.batchClassCounts().get(0));
    }

    @Test
    void multiLabelDistributionReportSummarizesPositiveAndNegativeCounts() {
        int[][] labels = {
                {1, 0, 1},
                {0, 1, 0},
                {1, 1, 0}
        };
        DataLoader.MultiLabelDistributionReport report = DataLoader.multiLabelDistribution(
                DataLoader.binary(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1),
                        labels,
                        2),
                3);

        assertEquals(3, report.sampleCount());
        assertEquals(2, report.batchCount());
        assertArrayEquals(new int[] {2, 2, 1}, report.positiveCounts());
        assertArrayEquals(new int[] {1, 1, 2}, report.negativeCounts());
        assertArrayEquals(new double[] {2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0}, report.positiveFractions(), 1e-6);
        assertArrayEquals(new int[] {1, 1, 1}, report.batchPositiveCounts().get(0));
        assertArrayEquals(new int[] {1, 1, 0}, report.batchPositiveCounts().get(1));
        assertEquals(2, report.positiveCount(0));
        assertEquals(2, report.negativeCount(2));
        assertEquals(0, report.maxPositiveLabelIndex());
        assertEquals(2, report.maxPositiveCount());
        assertEquals(2.0 / 3.0, report.maxPositiveFraction(), 1e-6);
        assertEquals(2, report.minPositiveLabelIndex());
        assertEquals(1, report.minPositiveCount());
        assertEquals(1.0 / 3.0, report.minPositiveFraction(), 1e-6);
        assertEquals(0, report.zeroPositiveLabelCount());
        assertEquals(0, report.allPositiveLabelCount());
        assertEquals(2.0, report.positiveImbalanceRatio(), 1e-6);
        assertEquals(5.0 / 3.0, report.labelCardinality(), 1e-6);
        assertEquals(5.0 / 9.0, report.labelDensity(), 1e-6);
        assertArrayEquals(new float[] {0.5f, 0.5f, 2.0f}, report.positiveWeights(), 1e-6f);
        assertEquals(2.0f, report.positiveWeight(2), 1e-6f);

        DataLoader.MultiLabelDistributionReport inferred = DataLoader.multiLabelDistribution(
                DataLoader.binary(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1),
                        labels,
                        2));
        assertEquals(3, inferred.labelCount());
        assertArrayEquals(report.positiveCounts(), inferred.positiveCounts());
        assertArrayEquals(report.negativeCounts(), inferred.negativeCounts());

        Map<String, Object> metadata = report.toMetadata("validation.distribution");
        assertEquals(3, metadata.get("validation.distribution.sampleCount"));
        assertEquals(2, metadata.get("validation.distribution.batchCount"));
        assertEquals(3, metadata.get("validation.distribution.labelCount"));
        assertEquals(List.of(2, 2, 1), metadata.get("validation.distribution.positiveCounts"));
        assertEquals(List.of(1, 1, 2), metadata.get("validation.distribution.negativeCounts"));
        assertEquals(
                List.of(2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
                metadata.get("validation.distribution.positiveFractions"));
        assertEquals(
                List.of(List.of(1, 1, 1), List.of(1, 1, 0)),
                metadata.get("validation.distribution.batchPositiveCounts"));
        assertEquals(0, metadata.get("validation.distribution.maxPositiveLabelIndex"));
        assertEquals(2, metadata.get("validation.distribution.maxPositiveCount"));
        assertEquals(2.0 / 3.0, (double) metadata.get("validation.distribution.maxPositiveFraction"), 1e-6);
        assertEquals(2, metadata.get("validation.distribution.minPositiveLabelIndex"));
        assertEquals(1, metadata.get("validation.distribution.minPositiveCount"));
        assertEquals(1.0 / 3.0, (double) metadata.get("validation.distribution.minPositiveFraction"), 1e-6);
        assertEquals(0, metadata.get("validation.distribution.zeroPositiveLabelCount"));
        assertEquals(0, metadata.get("validation.distribution.allPositiveLabelCount"));
        assertEquals(2.0, (double) metadata.get("validation.distribution.positiveImbalanceRatio"), 1e-6);
        assertEquals(5.0 / 3.0, (double) metadata.get("validation.distribution.labelCardinality"), 1e-6);
        assertEquals(5.0 / 9.0, (double) metadata.get("validation.distribution.labelDensity"), 1e-6);
        assertEquals(List.of(0.5f, 0.5f, 2.0f), metadata.get("validation.distribution.positiveWeights"));

        DataLoader.MultiLabelDistributionReport candidate = DataLoader.multiLabelDistribution(
                DataLoader.binary(
                        GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1),
                        new int[][] {
                                {0, 0, 1},
                                {0, 1, 1},
                                {0, 0, 0}
                        },
                        2));
        DataLoader.MultiLabelDistributionDriftReport drift = report.driftTo(candidate);
        assertArrayEquals(
                new double[] {2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0},
                drift.referencePositiveFractions(),
                1e-6);
        assertArrayEquals(new double[] {0.0, 1.0 / 3.0, 2.0 / 3.0}, drift.candidatePositiveFractions(), 1e-6);
        assertArrayEquals(new double[] {-2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0}, drift.positiveFractionDeltas(), 1e-6);
        assertEquals(4.0 / 9.0, drift.meanAbsolutePositiveFractionDelta(), 1e-6);
        assertEquals(0, drift.maxDeltaLabelIndex());
        assertEquals(2.0 / 3.0, drift.maxAbsolutePositiveFractionDelta(), 1e-6);
        assertEquals(5.0 / 3.0, drift.referenceLabelCardinality(), 1e-6);
        assertEquals(1.0, drift.candidateLabelCardinality(), 1e-6);
        assertEquals(-2.0 / 3.0, drift.labelCardinalityDelta(), 1e-6);
        assertEquals(List.of(), drift.referenceZeroPositiveLabels());
        assertEquals(List.of(0), drift.candidateZeroPositiveLabels());
        assertEquals(
                4.0 / 9.0,
                (double) drift.toMetadata("drift").get("drift.meanAbsolutePositiveFractionDelta"),
                1e-6);

        int[] mutableCounts = report.positiveCounts();
        mutableCounts[0] = 99;
        List<int[]> mutableBatches = report.batchPositiveCounts();
        mutableBatches.get(0)[0] = 99;

        assertArrayEquals(new int[] {2, 2, 1}, report.positiveCounts());
        assertArrayEquals(new int[] {1, 1, 1}, report.batchPositiveCounts().get(0));
    }

    @Test
    void multiLabelWeightedBinaryLoaderUsesBalancedSampleWeightsDeterministically() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 0}
        };
        DataLoader.TensorDataLoader loader = DataLoader.multiLabelWeightedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                2,
                6,
                2028L);

        assertTrue(loader.sampled());
        assertEquals(6, loader.sampleCount());
        assertEquals(3, loader.numBatches());
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.multiLabelWeightedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                2,
                6,
                2028L)));
        List<Float> drawn = flattenInputs(loader);
        assertTrue(
                drawn.contains(4f) || drawn.contains(5f),
                "at least one rare-label row should be selected by the weighted draw");
    }

    @Test
    void classBalancedBatchSamplerOversamplesMinorityClassPerBatchDeterministically() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1};
        DataLoader.TensorDataLoader loader = DataLoader.classBalancedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                3,
                2026L);

        assertTrue(loader.sampled());
        assertEquals(12, loader.sampleCount());
        assertEquals(3, loader.numBatches());
        for (DataLoader.Batch batch : loader) {
            List<Float> batchLabels = List.of(
                    batch.labels().data()[0],
                    batch.labels().data()[1],
                    batch.labels().data()[2],
                    batch.labels().data()[3]);
            assertEquals(2, Collections.frequency(batchLabels, 0f));
            assertEquals(2, Collections.frequency(batchLabels, 1f));
        }
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.classBalancedClassification(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                3,
                2026L)));
        assertEquals(6, Collections.frequency(flattenLabels(loader), 1f));

        DataLoader.TensorDataLoader reorderedBuilder = DataLoader.tensorBuilder(DataLoader.classificationDataset(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                        labels))
                .classBalancedBatchSampler(labels, 2, 2027L)
                .batchSize(4)
                .build();
        for (DataLoader.Batch batch : reorderedBuilder) {
            assertEquals(2, Collections.frequency(flattenBatchLabels(batch), 0f));
            assertEquals(2, Collections.frequency(flattenBatchLabels(batch), 1f));
        }
    }

    @Test
    void classBalancedBatchSamplerWithoutReplacementKeepsRowsUnique() {
        int[] labels = {0, 0, 1, 1, 2, 2};
        DataLoader.TensorDataLoader loader = DataLoader.tensorBuilder(DataLoader.classificationDataset(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                        labels))
                .classBalancedBatchSampler(labels, 3, 2, false, 44L)
                .build();

        List<Float> drawn = flattenInputs(loader);
        Collections.sort(drawn);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f), drawn);
        for (DataLoader.Batch batch : loader) {
            List<Float> batchLabels = List.of(
                    batch.labels().data()[0],
                    batch.labels().data()[1],
                    batch.labels().data()[2]);
            assertEquals(1, Collections.frequency(batchLabels, 0f));
            assertEquals(1, Collections.frequency(batchLabels, 1f));
            assertEquals(1, Collections.frequency(batchLabels, 2f));
        }
    }

    @Test
    void multiLabelBalancedBatchSamplerCoversLabelsPerBatchDeterministically() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {1, 1, 1}
        };
        DataLoader.TensorDataLoader loader = DataLoader.multiLabelBalancedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                3,
                4,
                2026L);

        assertTrue(loader.sampled());
        assertEquals(12, loader.sampleCount());
        assertEquals(4, loader.numBatches());
        for (DataLoader.Batch batch : loader) {
            int[] counts = positiveCounts(batch, 3);
            assertTrue(counts[0] >= 1, "missing positive label column 0");
            assertTrue(counts[1] >= 1, "missing positive label column 1");
            assertTrue(counts[2] >= 1, "missing positive label column 2");
        }
        assertEquals(flattenInputs(loader), flattenInputs(DataLoader.multiLabelBalancedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                3,
                4,
                2026L)));

        DataLoader.TensorDataLoader reorderedBuilder = DataLoader.tensorBuilder(DataLoader.binaryDataset(
                        GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                        labels))
                .multiLabelBalancedBatchSampler(labels, 4, 2027L)
                .batchSize(3)
                .build();
        for (DataLoader.Batch batch : reorderedBuilder) {
            int[] counts = positiveCounts(batch, 3);
            assertTrue(counts[0] >= 1);
            assertTrue(counts[1] >= 1);
            assertTrue(counts[2] >= 1);
        }
    }

    @Test
    void multiLabelBalancedBatchSamplerWithoutReplacementKeepsRowsUnique() {
        int[][] labels = {
                {1, 0},
                {0, 1},
                {1, 1},
                {1, 0},
                {0, 1},
                {1, 1}
        };
        DataLoader.TensorDataLoader loader = DataLoader.multiLabelBalancedBinary(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                3,
                2,
                false,
                2027L);

        List<Float> drawn = flattenInputs(loader);
        Collections.sort(drawn);
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f), drawn);
        for (DataLoader.Batch batch : loader) {
            int[] counts = positiveCounts(batch, 2);
            assertTrue(counts[0] >= 1);
            assertTrue(counts[1] >= 1);
        }
    }

    @Test
    void randomSamplerIsDeterministicAndSupportsReplacement() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1));

        DataLoader.TensorDataLoader first = DataLoader.tensorBuilder(dataset)
                .batchSize(5)
                .randomSampler(2026L)
                .build();
        DataLoader.TensorDataLoader second = DataLoader.tensorBuilder(dataset)
                .batchSize(5)
                .randomSampler(2026L)
                .build();
        List<Float> firstDraw = flattenInputs(first);
        List<Float> sorted = new ArrayList<>(firstDraw);
        Collections.sort(sorted);

        assertTrue(first.sampled());
        assertEquals(firstDraw, flattenInputs(second));
        assertEquals(List.of(1f, 2f, 3f, 4f, 5f), sorted);

        DataLoader.TensorDataset single = DataLoader.tensorDataset(
                GradTensor.of(new float[] {7f}, 1, 1),
                GradTensor.of(new float[] {1f}, 1, 1));
        DataLoader.TensorDataLoader replacement = DataLoader.tensorBuilder(single)
                .batchSize(2)
                .randomSampler(4, true, 99L)
                .build();

        assertEquals(4, replacement.sampleCount());
        assertEquals(2, replacement.numBatches());
        assertEquals(List.of(7f, 7f, 7f, 7f), flattenInputs(replacement));
    }

    @Test
    void subsetAndSequentialSamplersSelectExpectedRows() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f, 20f, 30f, 40f, 50f}, 5, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 1f, 1f}, 5, 1));

        DataLoader.TensorDataLoader subset = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(7L)
                .subsetSampler(4, 0, 2)
                .build();
        DataLoader.TensorDataLoader sequential = DataLoader.tensorBuilder(dataset)
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
    void distributedSamplerPartitionsAndPadsDatasetAcrossRanks() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 1f, 1f, 1f, 1f}, 5, 1));

        DataLoader.TensorDataLoader rank0 = DataLoader.tensorBuilder(dataset)
                .batchSize(3)
                .distributedSampler(2, 0, false, false, 7L)
                .build();
        DataLoader.TensorDataLoader rank1 = DataLoader.tensorBuilder(dataset)
                .batchSize(3)
                .distributedSampler(2, 1, false, false, 7L)
                .build();

        assertEquals(3, rank0.sampleCount());
        assertEquals(3, rank1.sampleCount());
        assertEquals(List.of(1f, 3f, 5f), flattenInputs(rank0));
        assertEquals(List.of(2f, 4f, 1f), flattenInputs(rank1));

        DataLoader.TensorDataLoader dropRank0 = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .distributedSampler(2, 0, false, true, 7L)
                .build();
        DataLoader.TensorDataLoader dropRank1 = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .distributedSampler(2, 1, false, true, 7L)
                .build();

        assertEquals(2, dropRank0.sampleCount());
        assertEquals(List.of(1f, 3f), flattenInputs(dropRank0));
        assertEquals(List.of(2f, 4f), flattenInputs(dropRank1));
    }

    @Test
    void distributedSamplerShuffleIsSeedAndEpochDeterministic() {
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
    void balancedSampleWeightHelpersReturnPerSampleInverseClassWeights() {
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
    void classificationLoaderBuildsClassIndexLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.classification(
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
        assertArrayEquals(new long[] {2, 2}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {2}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {0f, 1f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
    }

    @Test
    void binaryLoaderBuildsBceCompatibleColumnLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.binary(
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
    void binaryLoaderBuildsMultiLabelBceCompatibleLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.binary(
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
    void classificationDatasetRejectsInvalidLabels() {
        GradTensor inputs = GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);

        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classificationDataset(inputs, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classificationDataset(inputs, new int[] {0, -1}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[] {0, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 1}, {1}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 2}, {1, 0}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 1}}));
    }

    @Test
    void tensorLoaderRejectsInvalidConfiguration() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f, 2f}, 2, 1));

        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).batchSize(0));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(1.0, 1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(0.8, 0.2, 1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(Double.NaN, 0.1, 1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f, 2f}, 2, 1)).split(0.5, 0.25, 1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.kFold(1, 1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.kFold(3, 1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.repeatedKFold(2, 0, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> dataset.groupKFold(new int[] {1}, 2, 1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.groupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                new int[] {1, 1, 2, 2},
                3,
                1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classificationStratifiedGroupKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                new int[] {0, 0, 1, 1},
                new int[] {1, 1, 2, 2},
                3,
                1L));
        assertThrows(IllegalArgumentException.class, () -> dataset.timeSeriesSplit(0));
        assertThrows(IllegalArgumentException.class, () -> dataset.timeSeriesSplit(2, 1, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> dataset.timeSeriesSplit(2, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classificationStratifiedKFold(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                new int[] {0, 0, 0, 1},
                2,
                1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.weightedRandomSampler(new float[] {0f, 0f}, 1, true, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.weightedRandomSampler(new float[] {1f}, 2, false, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classBalancedBatchSampler(new int[] {0, -1}, 2, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classBalancedBatchSampler(new int[] {0, 1}, 0, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classBalancedBatchSampler(new int[] {0, 1}, 2, 2, false, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.stratifiedBatchSampler(new int[] {0, -1}, 2, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.stratifiedBatchSampler(new int[] {0, 1}, 0, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.stratifiedBatchSampler(new int[] {0, 1}, 2, 2, false, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classificationDistribution(DataLoader.classification(
                        GradTensor.of(new float[] {1f, 2f}, 2, 1), new int[] {0, 1}, 1), 0));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.multiLabelDistribution(DataLoader.binary(
                        GradTensor.of(new float[] {1f, 2f}, 2, 1), new int[][] {{1, 0}, {0, 1}}, 1), 0));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.multiLabelDistribution(DataLoader.binary(
                        GradTensor.of(new float[] {1f, 2f}, 2, 1), new int[][] {{1, 0}, {0, 1}}, 1), 3));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.multiLabelBalancedBatchSampler(new int[][] {{0, 0}, {0, 0}}, 2, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.multiLabelBalancedBatchSampler(new int[][] {{1, 0}, {0, 1}}, 0, 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.multiLabelBalancedBatchSampler(new int[][] {{1, 0}, {0, 1}}, 2, 2, false, 1L));
        DataLoader.TensorDataLoader mismatchedBalancedSampler = DataLoader.tensorBuilder(dataset)
                .classBalancedBatchSampler(new int[] {0}, 2, 1L)
                .build();
        assertThrows(IllegalArgumentException.class, mismatchedBalancedSampler::numBatches);
        DataLoader.TensorDataLoader mismatchedStratifiedSampler = DataLoader.tensorBuilder(dataset)
                .stratifiedBatchSampler(new int[] {0}, 2, 1L)
                .build();
        assertThrows(IllegalArgumentException.class, mismatchedStratifiedSampler::numBatches);
        DataLoader.TensorDataLoader mismatchedMultiLabelBalancedSampler = DataLoader.tensorBuilder(dataset)
                .multiLabelBalancedBatchSampler(new int[][] {{1, 0}}, 2, 1L)
                .build();
        assertThrows(IllegalArgumentException.class, mismatchedMultiLabelBalancedSampler::numBatches);
        DataLoader.TensorDataLoader mismatchedSampler = DataLoader.tensorBuilder(dataset)
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

    private static <T> List<T> flattenGeneric(Iterable<List<T>> loader) {
        List<T> values = new ArrayList<>();
        for (List<T> batch : loader) {
            values.addAll(batch);
        }
        return values;
    }

    private static <T> List<T> collect(Iterable<T> source) {
        List<T> values = new ArrayList<>();
        for (T value : source) {
            values.add(value);
        }
        return values;
    }

    private static List<Integer> batchSums(List<Integer> values, int batchSize) {
        List<Integer> sums = new ArrayList<>();
        for (int start = 0; start < values.size(); start += batchSize) {
            int end = Math.min(start + batchSize, values.size());
            int sum = 0;
            for (int index = start; index < end; index++) {
                sum += values.get(index);
            }
            sums.add(sum);
        }
        return sums;
    }

    private static Dataset.Sample sampleWithInputLength(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = i + 1f;
        }
        return new Dataset.Sample(GradTensor.of(values, length), GradTensor.of(new float[] {1f}, 1));
    }

    private static Map<String, Integer> labelCounts(Dataset<LabeledExample> dataset) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            counts.merge(dataset.get(i).label(), 1, Integer::sum);
        }
        return counts;
    }

    private static <T extends Comparable<? super T>> void assertValidationCovers(
            Dataset<T> source,
            List<Dataset.Fold<T>> folds) {
        List<T> covered = new ArrayList<>();
        for (Dataset.Fold<T> fold : folds) {
            covered.addAll(fold.validation().toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList(), covered);
    }

    private static void assertLabeledValidationCovers(
            Dataset<LabeledExample> source,
            List<Dataset.Fold<LabeledExample>> folds) {
        List<String> covered = new ArrayList<>();
        for (Dataset.Fold<LabeledExample> fold : folds) {
            covered.addAll(fold.validation().toList().stream().map(LabeledExample::id).toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList().stream().map(LabeledExample::id).toList(), covered);
    }

    private static Dataset<GroupedExample> groupedExamples(int groups, int samplesPerGroup) {
        List<GroupedExample> examples = new ArrayList<>();
        for (int group = 1; group <= groups; group++) {
            for (int sample = 1; sample <= samplesPerGroup; sample++) {
                examples.add(new GroupedExample("g" + group + "-s" + sample, "g" + group));
            }
        }
        return Dataset.from(examples);
    }

    private static Set<String> groupIds(Dataset<GroupedExample> dataset) {
        Set<String> groups = new HashSet<>();
        for (int i = 0; i < dataset.size(); i++) {
            groups.add(dataset.get(i).group());
        }
        return groups;
    }

    private static void assertNoGroupLeak(Dataset<GroupedExample> train, Dataset<GroupedExample> validation) {
        assertTrue(Collections.disjoint(groupIds(train), groupIds(validation)));
    }

    private static void assertGroupedValidationCovers(
            Dataset<GroupedExample> source,
            List<Dataset.Fold<GroupedExample>> folds) {
        List<String> covered = new ArrayList<>();
        for (Dataset.Fold<GroupedExample> fold : folds) {
            covered.addAll(fold.validation().toList().stream().map(GroupedExample::id).toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList().stream().map(GroupedExample::id).toList(), covered);
    }

    private static <T> Dataset<T> datasetOf(List<T> values) {
        return new Dataset<>() {
            @Override
            public T get(int index) {
                return values.get(index);
            }

            @Override
            public int size() {
                return values.size();
            }
        };
    }

    private static Dataset<Integer> hugeDataset() {
        return new Dataset<>() {
            @Override
            public Integer get(int index) {
                return index;
            }

            @Override
            public int size() {
                return Integer.MAX_VALUE;
            }
        };
    }

    private static List<Float> flattenDataset(DataLoader.TensorDataset dataset) {
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            for (float value : dataset.get(i)[0].data()) {
                values.add(value);
            }
        }
        return values;
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

    private static List<Float> flattenLabels(DataLoader.TensorDataLoader loader) {
        List<Float> values = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            for (float value : batch.labels().data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static List<Float> flattenBatchLabels(DataLoader.Batch batch) {
        List<Float> values = new ArrayList<>();
        for (float value : batch.labels().data()) {
            values.add(value);
        }
        return values;
    }

    private static Set<Integer> groupsForRows(List<Float> rows, int[] groups) {
        Set<Integer> result = new HashSet<>();
        for (float row : rows) {
            result.add(groups[(int) row - 1]);
        }
        return result;
    }

    private static int[][] balancedGroupedMultiLabelRows() {
        return new int[][] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0}
        };
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

    private static int[] positiveCounts(DataLoader.Batch batch, int columns) {
        int[] counts = new int[columns];
        float[] labels = batch.labels().data();
        assertEquals(0, labels.length % columns);
        for (int row = 0; row < labels.length / columns; row++) {
            for (int column = 0; column < columns; column++) {
                if (labels[row * columns + column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }

    private static void assertCountsWithin(int[] expected, int[] actual, int tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertTrue(
                    Math.abs(expected[i] - actual[i]) <= tolerance,
                    "expected " + Arrays.toString(expected) + " within " + tolerance
                            + ", actual " + Arrays.toString(actual));
        }
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

    private record LabeledExample(String id, String label) {}

    private record GroupedExample(String id, String group) {}
}
