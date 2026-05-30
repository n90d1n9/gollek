package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

public class DatasetTest {

    @TempDir
    Path tempDir;

    @Test
    public void testCsvDatasetWithHeader() throws IOException {
        Path csvFile = tempDir.resolve("test.csv");
        Files.writeString(csvFile, "id,name,value\n1,alpha,10.0\n2,beta,20.0");

        CsvDataset dataset = new CsvDataset(csvFile, ",", true);
        assertEquals(2, dataset.size());

        Map<String, String> row1 = dataset.get(0);
        assertEquals("1", row1.get("id"));
        assertEquals("alpha", row1.get("name"));
        assertEquals("10.0", row1.get("value"));

        List<String> values = dataset.column("value");
        assertEquals(List.of("10.0", "20.0"), values);
    }

    @Test
    public void testCsvDatasetNoHeader() throws IOException {
        Path csvFile = tempDir.resolve("test_no_header.csv");
        Files.writeString(csvFile, "1,alpha,10.0\n2,beta,20.0");

        CsvDataset dataset = new CsvDataset(csvFile, ",", false);
        assertEquals(2, dataset.size());

        Map<String, String> row1 = dataset.get(0);
        assertEquals("1", row1.get("col_0"));
        assertEquals("alpha", row1.get("col_1"));
        assertEquals("10.0", row1.get("col_2"));
    }

    @Test
    public void testTextDataset() throws IOException {
        Path txtFile = tempDir.resolve("test.txt");
        Files.writeString(txtFile, "line 1\nline 2\nline 3");

        TextDataset dataset = new TextDataset(txtFile);
        assertEquals(3, dataset.size());
        assertEquals("line 1", dataset.get(0));
        assertEquals("line 3", dataset.get(2));
    }

    @Test
    public void testInMemoryDatasetFactoriesAreDefensive() {
        List<String> source = new ArrayList<>(List.of("alpha", "beta"));
        Dataset<String> fromList = Dataset.from(source);
        source.set(0, "changed");
        source.add("gamma");

        assertEquals(2, fromList.size());
        assertEquals("alpha", fromList.get(0));
        assertEquals("beta", fromList.get(1));

        Dataset<Integer> fromVarargs = Dataset.of(1, 2, 3);
        assertEquals(3, fromVarargs.size());
        assertEquals(2, fromVarargs.get(1));
    }

    @Test
    public void testDatasetTransformsComposeAndReturnSafeLists() {
        Dataset<String> transformed = Dataset.of(1, 2, 3, 4)
                .filter(value -> value >= 2)
                .map(value -> "v" + value);

        assertEquals(3, transformed.size());
        assertEquals("v2", transformed.get(0));
        assertEquals(List.of("v2", "v3", "v4"), transformed.toList());
        assertThrows(UnsupportedOperationException.class, () -> transformed.toList().add("v5"));
    }

    @Test
    public void testDatasetSplitsAreDeterministic() {
        Dataset<String> source = Dataset.of("a", "b", "c", "d", "e", "f");
        Dataset.Split<String> first = source.split(0.5, 42L);
        Dataset.Split<String> second = source.split(0.5, 42L);

        assertEquals(3, first.train().size());
        assertEquals(3, first.validation().size());
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());

        Dataset.ThreeWaySplit<String> threeWay = source.split(0.5, 0.25, 42L);
        assertEquals(3, threeWay.train().size());
        assertEquals(2, threeWay.validation().size());
        assertEquals(1, threeWay.test().size());

        assertThrows(IllegalArgumentException.class, () -> source.split(0.0, 42L));
        assertThrows(IllegalArgumentException.class, () -> source.split(0.5, 0.5, 42L));
    }

    @Test
    public void testDatasetStratifiedSplitsAreDeterministic() {
        Dataset<LegacyExample> source = Dataset.of(
                new LegacyExample("cat-1", "cat"),
                new LegacyExample("cat-2", "cat"),
                new LegacyExample("cat-3", "cat"),
                new LegacyExample("cat-4", "cat"),
                new LegacyExample("dog-1", "dog"),
                new LegacyExample("dog-2", "dog"),
                new LegacyExample("dog-3", "dog"),
                new LegacyExample("dog-4", "dog"));

        Dataset.Split<LegacyExample> first = source.stratifiedSplit(LegacyExample::label, 0.5, 7L);
        Dataset.Split<LegacyExample> second = source.stratifiedSplit(LegacyExample::label, 0.5, 7L);

        assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(first.train()));
        assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(first.validation()));
        assertEquals(first.train().toList(), second.train().toList());
        assertEquals(first.validation().toList(), second.validation().toList());

        Dataset.ThreeWaySplit<LegacyExample> threeWay = source.stratifiedSplit(LegacyExample::label, 0.5, 0.25, 7L);
        assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(threeWay.train()));
        assertEquals(Map.of("cat", 1, "dog", 1), labelCounts(threeWay.validation()));
        assertEquals(Map.of("cat", 1, "dog", 1), labelCounts(threeWay.test()));

        assertThrows(NullPointerException.class, () -> source.stratifiedSplit(null, 0.5, 1L));
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(source.get(0))
                .stratifiedSplit(LegacyExample::label, 0.5, 1L));
    }

    @Test
    public void testDatasetKFoldAndStratifiedKFoldAreDeterministic() {
        Dataset<Integer> numeric = Dataset.of(1, 2, 3, 4, 5, 6);
        List<Dataset.Fold<Integer>> folds = numeric.kFold(3, 11L);
        List<Dataset.Fold<Integer>> repeated = numeric.repeatedKFold(3, 2, 11L);

        assertEquals(3, folds.size());
        assertEquals(List.of(2, 2, 2), folds.stream().map(fold -> fold.validation().size()).toList());
        assertValidationCovers(numeric, folds);
        assertEquals(6, repeated.size());
        assertEquals(6, repeated.get(0).foldCount());
        assertValidationCovers(numeric, repeated.subList(0, 3));
        assertValidationCovers(numeric, repeated.subList(3, 6));

        Dataset<LegacyExample> labeled = Dataset.of(
                new LegacyExample("cat-1", "cat"),
                new LegacyExample("cat-2", "cat"),
                new LegacyExample("cat-3", "cat"),
                new LegacyExample("cat-4", "cat"),
                new LegacyExample("dog-1", "dog"),
                new LegacyExample("dog-2", "dog"),
                new LegacyExample("dog-3", "dog"),
                new LegacyExample("dog-4", "dog"));
        List<Dataset.Fold<LegacyExample>> stratified = labeled.stratifiedKFold(LegacyExample::label, 2, 12L);
        List<Dataset.Fold<LegacyExample>> repeatedStratified =
                labeled.repeatedStratifiedKFold(LegacyExample::label, 2, 2, 12L);

        for (Dataset.Fold<LegacyExample> fold : stratified) {
            assertEquals(Map.of("cat", 2, "dog", 2), labelCounts(fold.validation()));
        }
        assertLegacyValidationCovers(labeled, stratified);
        assertEquals(4, repeatedStratified.size());
        assertEquals(4, repeatedStratified.get(0).foldCount());
        assertLegacyValidationCovers(labeled, repeatedStratified.subList(0, 2));
        assertLegacyValidationCovers(labeled, repeatedStratified.subList(2, 4));

        assertThrows(IllegalArgumentException.class, () -> numeric.kFold(1, 1L));
        assertThrows(IllegalArgumentException.class, () -> numeric.repeatedKFold(2, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(
                        new LegacyExample("cat-1", "cat"),
                        new LegacyExample("dog-1", "dog"))
                .stratifiedKFold(LegacyExample::label, 2, 1L));
    }

    @Test
    public void testDatasetGroupSplitsAvoidLeakage() {
        Dataset<LegacyGroupedExample> source = legacyGroupedExamples(4, 2);
        Dataset.Split<LegacyGroupedExample> split = source.groupSplit(LegacyGroupedExample::group, 0.5, 15L);
        Dataset.ThreeWaySplit<LegacyGroupedExample> threeWay =
                legacyGroupedExamples(6, 2).groupSplit(LegacyGroupedExample::group, 0.5, 0.25, 15L);
        List<Dataset.Fold<LegacyGroupedExample>> folds = source.groupKFold(LegacyGroupedExample::group, 4, 15L);
        List<Dataset.Fold<LegacyGroupedExample>> repeated =
                source.repeatedGroupKFold(LegacyGroupedExample::group, 2, 2, 15L);

        assertNoLegacyGroupLeak(split.train(), split.validation());
        assertTrue(Collections.disjoint(groupIds(threeWay.train()), groupIds(threeWay.validation())));
        assertTrue(Collections.disjoint(groupIds(threeWay.train()), groupIds(threeWay.test())));
        assertTrue(Collections.disjoint(groupIds(threeWay.validation()), groupIds(threeWay.test())));
        for (Dataset.Fold<LegacyGroupedExample> fold : folds) {
            assertNoLegacyGroupLeak(fold.train(), fold.validation());
        }
        assertLegacyGroupedValidationCovers(source, folds);
        assertEquals(4, repeated.size());
        assertLegacyGroupedValidationCovers(source, repeated.subList(0, 2));
        assertLegacyGroupedValidationCovers(source, repeated.subList(2, 4));

        assertThrows(IllegalArgumentException.class, () -> legacyGroupedExamples(1, 2)
                .groupSplit(LegacyGroupedExample::group, 0.5, 1L));
        assertThrows(IllegalArgumentException.class, () -> source.groupKFold(LegacyGroupedExample::group, 5, 1L));
    }

    @Test
    public void testDatasetTimeSeriesSplitsWalkForward() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        List<Dataset.Fold<Integer>> defaultFolds = source.timeSeriesSplit(3);
        List<Dataset.Fold<Integer>> cappedFolds = source.timeSeriesSplit(2, 2, 1, 3);

        assertEquals(List.of(1, 2, 3, 4), defaultFolds.get(0).train().toList());
        assertEquals(List.of(5, 6), defaultFolds.get(0).validation().toList());
        assertEquals(List.of(1, 2, 3, 4, 5, 6), defaultFolds.get(1).train().toList());
        assertEquals(List.of(7, 8), defaultFolds.get(1).validation().toList());
        assertEquals(List.of(1, 2, 3, 4, 5, 6, 7, 8), defaultFolds.get(2).train().toList());
        assertEquals(List.of(9, 10), defaultFolds.get(2).validation().toList());

        assertEquals(List.of(3, 4, 5), cappedFolds.get(0).train().toList());
        assertEquals(List.of(7, 8), cappedFolds.get(0).validation().toList());
        assertEquals(List.of(5, 6, 7), cappedFolds.get(1).train().toList());
        assertEquals(List.of(9, 10), cappedFolds.get(1).validation().toList());
        for (Dataset.Fold<Integer> fold : cappedFolds) {
            assertTrue(Collections.max(fold.train().toList()) < Collections.min(fold.validation().toList()));
        }

        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(0));
        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(2, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> source.timeSeriesSplit(2, 1, 8, 0));
    }

    @Test
    public void testDatasetWindowedBuildsForecastingExamples() {
        Dataset<Integer> source = Dataset.of(1, 2, 3, 4, 5, 6, 7, 8);
        Dataset<Dataset.Window<Integer>> windows = source.windowed(3, 2, 2);
        Dataset<Dataset.Window<Integer>> oneStep = source.windowed(3);

        assertEquals(2, windows.size());
        assertEquals(List.of(1, 2, 3), windows.get(0).inputs());
        assertEquals(List.of(4, 5), windows.get(0).targets());
        assertEquals(List.of(3, 4, 5), windows.get(1).inputs());
        assertEquals(List.of(6, 7), windows.get(1).targets());
        assertThrows(UnsupportedOperationException.class, () -> windows.get(0).targets().add(99));

        assertEquals(5, oneStep.size());
        assertEquals(List.of(1, 2, 3), oneStep.get(0).inputs());
        assertEquals(List.of(4), oneStep.get(0).targets());
        assertEquals(List.of(5, 6, 7), oneStep.get(4).inputs());
        assertEquals(List.of(8), oneStep.get(4).targets());
        assertEquals(0, Dataset.of(1, 2, 3).windowed(3, 1).size());

        assertThrows(IndexOutOfBoundsException.class, () -> windows.get(2));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(0));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(2, 0));
        assertThrows(IllegalArgumentException.class, () -> source.windowed(2, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new Dataset.Window<>(List.of(1), List.of()));
    }

    @Test
    public void testDatasetZipEnumerateAndShardViews() {
        Dataset<String> inputs = Dataset.of("red", "green", "blue", "yellow");
        Dataset<Integer> labels = Dataset.of(0, 1, 0, 1);
        Dataset<Dataset.Pair<String, Integer>> pairs = inputs.zip(labels);

        assertEquals(4, pairs.size());
        assertEquals(new Dataset.Pair<>("green", 1), pairs.get(1));
        assertEquals(pairs.toList(), Dataset.zip(inputs, labels).toList());

        Dataset<Dataset.Indexed<Dataset.Pair<String, Integer>>> indexed = pairs.enumerate();
        assertEquals(2, indexed.get(2).index());
        assertEquals(new Dataset.Pair<>("blue", 0), indexed.get(2).value());

        Dataset<Integer> source = Dataset.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        assertEquals(List.of(0, 3, 6, 9), source.shard(3, 0).toList());
        assertEquals(List.of(1, 4, 7), source.shard(3, 1).toList());
        assertEquals(List.of(2, 5, 8), source.shard(3, 2).toList());
        assertEquals(List.of(), source.shard(12, 11).toList());

        assertThrows(IllegalArgumentException.class, () -> new Dataset.Indexed<>(-1, "bad"));
        assertThrows(IllegalArgumentException.class, () -> inputs.zip(Dataset.of(0, 1)));
        assertThrows(IndexOutOfBoundsException.class, () -> source.shard(3, 1).get(3));
        assertThrows(IllegalArgumentException.class, () -> source.shard(0, 0));
        assertThrows(IllegalArgumentException.class, () -> source.shard(2, 2));
    }

    @Test
    public void testDatasetTakeDropSliceAndRepeatViews() {
        Dataset<String> source = Dataset.of("a", "b", "c", "d", "e");

        assertEquals(List.of("a", "b", "c"), source.take(3).toList());
        assertEquals(source.toList(), source.take(99).toList());
        assertEquals(List.of("c", "d", "e"), source.drop(2).toList());
        assertEquals(List.of(), source.drop(99).toList());
        assertEquals(List.of("b", "c", "d"), source.slice(1, 4).toList());
        assertEquals(List.of(), source.slice(2, 2).toList());
        assertEquals(List.of("a", "b", "c", "a", "b", "c"), source.take(3).repeat(2).toList());
        assertEquals(List.of(), Dataset.<String>of().repeat(3).toList());

        assertThrows(IndexOutOfBoundsException.class, () -> source.drop(5).get(0));
        assertThrows(IllegalArgumentException.class, () -> source.take(-1));
        assertThrows(IllegalArgumentException.class, () -> source.drop(-1));
        assertThrows(IllegalArgumentException.class, () -> source.slice(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> source.slice(3, 2));
        assertThrows(IllegalArgumentException.class, () -> source.slice(0, 6));
        assertThrows(IllegalArgumentException.class, () -> source.repeat(-1));
        assertThrows(IllegalArgumentException.class, () -> hugeDataset().repeat(2));
    }

    @Test
    public void testDatasetCacheMemoizesLazyTransforms() {
        AtomicInteger calls = new AtomicInteger();
        Dataset<Integer> cached = Dataset.of(1, 2, 3)
                .map(value -> {
                    calls.incrementAndGet();
                    return value * 10;
                })
                .cache();

        assertEquals(3, cached.size());
        assertEquals(20, cached.get(1));
        assertEquals(20, cached.get(1));
        assertEquals(1, calls.get());
        assertEquals(List.of(10, 20, 30), cached.toList());
        assertEquals(3, calls.get());
        assertEquals(List.of(10, 20, 30), cached.toList());
        assertEquals(3, calls.get());
        assertEquals(List.of(), Dataset.<Integer>of().cache().toList());
        assertThrows(IndexOutOfBoundsException.class, () -> cached.get(-1));
    }

    @Test
    public void testDatasetShuffleAndSamplingViewsAreDeterministic() {
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

    private static Map<String, Integer> labelCounts(Dataset<LegacyExample> dataset) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            counts.merge(dataset.get(i).label(), 1, Integer::sum);
        }
        return counts;
    }

    private static void assertValidationCovers(Dataset<Integer> source, List<Dataset.Fold<Integer>> folds) {
        List<Integer> covered = new ArrayList<>();
        for (Dataset.Fold<Integer> fold : folds) {
            covered.addAll(fold.validation().toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList(), covered);
    }

    private static void assertLegacyValidationCovers(
            Dataset<LegacyExample> source,
            List<Dataset.Fold<LegacyExample>> folds) {
        List<String> covered = new ArrayList<>();
        for (Dataset.Fold<LegacyExample> fold : folds) {
            covered.addAll(fold.validation().toList().stream().map(LegacyExample::id).toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList().stream().map(LegacyExample::id).toList(), covered);
    }

    private static Dataset<LegacyGroupedExample> legacyGroupedExamples(int groups, int samplesPerGroup) {
        List<LegacyGroupedExample> examples = new ArrayList<>();
        for (int group = 1; group <= groups; group++) {
            for (int sample = 1; sample <= samplesPerGroup; sample++) {
                examples.add(new LegacyGroupedExample("g" + group + "-s" + sample, "g" + group));
            }
        }
        return Dataset.from(examples);
    }

    private static Set<String> groupIds(Dataset<LegacyGroupedExample> dataset) {
        Set<String> groups = new HashSet<>();
        for (int i = 0; i < dataset.size(); i++) {
            groups.add(dataset.get(i).group());
        }
        return groups;
    }

    private static void assertNoLegacyGroupLeak(
            Dataset<LegacyGroupedExample> train,
            Dataset<LegacyGroupedExample> validation) {
        assertTrue(Collections.disjoint(groupIds(train), groupIds(validation)));
    }

    private static void assertLegacyGroupedValidationCovers(
            Dataset<LegacyGroupedExample> source,
            List<Dataset.Fold<LegacyGroupedExample>> folds) {
        List<String> covered = new ArrayList<>();
        for (Dataset.Fold<LegacyGroupedExample> fold : folds) {
            covered.addAll(fold.validation().toList().stream().map(LegacyGroupedExample::id).toList());
        }
        Collections.sort(covered);
        assertEquals(source.toList().stream().map(LegacyGroupedExample::id).toList(), covered);
    }

    private static Dataset<Integer> hugeDataset() {
        return new Dataset<>() {
            @Override
            public int size() {
                return Integer.MAX_VALUE;
            }

            @Override
            public Integer get(int index) {
                return index;
            }
        };
    }

    private record LegacyExample(String id, String label) {}

    private record LegacyGroupedExample(String id, String group) {}
}
