package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Base interface for all datasets, similar to {@code torch.utils.data.Dataset}.
 */
public interface Dataset<T> {

    static <T> Dataset<T> from(List<? extends T> items) {
        return new InMemoryDataset<>(items);
    }

    static <L, R> Dataset<Pair<L, R>> zip(Dataset<? extends L> left, Dataset<? extends R> right) {
        return GenericDatasetViews.zip(left, right);
    }

    @SafeVarargs
    static <T> Dataset<T> of(T... items) {
        Objects.requireNonNull(items, "items must not be null");
        return from(Arrays.asList(items));
    }

    int size();

    T get(int index);

    default <R> Dataset<R> map(Function<? super T, ? extends R> mapper) {
        return new MappedDataset<>(this, mapper);
    }

    default <U> Dataset<Pair<T, U>> zip(Dataset<? extends U> other) {
        return GenericDatasetViews.zip(this, other);
    }

    default Dataset<Indexed<T>> enumerate() {
        return GenericDatasetViews.enumerate(this);
    }

    default Dataset<T> shard(int shardCount, int shardIndex) {
        return GenericDatasetViews.shard(this, shardCount, shardIndex);
    }

    default Dataset<T> shuffle(long seed) {
        return GenericDatasetViews.shuffle(this, seed);
    }

    default Dataset<T> sample(int count, long seed) {
        return GenericDatasetViews.sample(this, count, seed);
    }

    default Dataset<T> sampleWithReplacement(int count, long seed) {
        return GenericDatasetViews.sampleWithReplacement(this, count, seed);
    }

    default Dataset<T> take(int count) {
        return GenericDatasetViews.take(this, count);
    }

    default Dataset<T> drop(int count) {
        return GenericDatasetViews.drop(this, count);
    }

    default Dataset<T> slice(int fromInclusive, int toExclusive) {
        return GenericDatasetViews.slice(this, fromInclusive, toExclusive);
    }

    default Dataset<T> repeat(int times) {
        return GenericDatasetViews.repeat(this, times);
    }

    default Dataset<T> cache() {
        return GenericDatasetViews.cache(this);
    }

    default Dataset<T> filter(Predicate<? super T> predicate) {
        return new FilteredDataset<>(this, predicate);
    }

    default List<T> toList() {
        List<T> values = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            values.add(get(i));
        }
        return Collections.unmodifiableList(values);
    }

    default Dataset<Window<T>> windowed(int inputSize) {
        return GenericDatasetWindows.windowed(this, inputSize, 1, 1);
    }

    default Dataset<Window<T>> windowed(int inputSize, int targetSize) {
        return GenericDatasetWindows.windowed(this, inputSize, targetSize, 1);
    }

    default Dataset<Window<T>> windowed(int inputSize, int targetSize, int stride) {
        return GenericDatasetWindows.windowed(this, inputSize, targetSize, stride);
    }

    default Split<T> split(double trainFraction, long seed) {
        requireTrainFraction(trainFraction);
        if (size() < 2) {
            throw new IllegalArgumentException("dataset must contain at least two samples for train/validation split");
        }
        List<Integer> indices = shuffledIndices(size(), seed);
        int trainSize = (int) Math.round(size() * trainFraction);
        trainSize = Math.max(1, Math.min(size() - 1, trainSize));
        return new Split<>(
                new IndexedDataset<>(this, indices.subList(0, trainSize)),
                new IndexedDataset<>(this, indices.subList(trainSize, indices.size())));
    }

    default ThreeWaySplit<T> split(double trainFraction, double validationFraction, long seed) {
        int[] counts = threeWayCounts(size(), trainFraction, validationFraction);
        List<Integer> indices = shuffledIndices(size(), seed);
        int validationEnd = counts[0] + counts[1];
        return new ThreeWaySplit<>(
                new IndexedDataset<>(this, indices.subList(0, counts[0])),
                new IndexedDataset<>(this, indices.subList(counts[0], validationEnd)),
                new IndexedDataset<>(this, indices.subList(validationEnd, indices.size())));
    }

    default List<Fold<T>> kFold(int folds, long seed) {
        return GenericDatasetSplits.kFold(this, folds, seed);
    }

    default List<Fold<T>> repeatedKFold(int folds, int repeats, long seed) {
        return GenericDatasetSplits.repeatedKFold(this, folds, repeats, seed);
    }

    default List<Fold<T>> timeSeriesSplit(int splits) {
        return GenericDatasetSplits.timeSeriesSplit(this, splits);
    }

    default List<Fold<T>> timeSeriesSplit(int splits, int validationSize, int gap, int maxTrainSize) {
        return GenericDatasetSplits.timeSeriesSplit(this, splits, validationSize, gap, maxTrainSize);
    }

    default <K> Split<T> groupSplit(
            Function<? super T, ? extends K> groupExtractor,
            double trainFraction,
            long seed) {
        return GenericDatasetSplits.groupSplit(this, groupExtractor, trainFraction, seed);
    }

    default <K> ThreeWaySplit<T> groupSplit(
            Function<? super T, ? extends K> groupExtractor,
            double trainFraction,
            double validationFraction,
            long seed) {
        return GenericDatasetSplits.groupSplit(this, groupExtractor, trainFraction, validationFraction, seed);
    }

    default <K> List<Fold<T>> groupKFold(
            Function<? super T, ? extends K> groupExtractor,
            int folds,
            long seed) {
        return GenericDatasetSplits.groupKFold(this, groupExtractor, folds, seed);
    }

    default <K> List<Fold<T>> repeatedGroupKFold(
            Function<? super T, ? extends K> groupExtractor,
            int folds,
            int repeats,
            long seed) {
        return GenericDatasetSplits.repeatedGroupKFold(this, groupExtractor, folds, repeats, seed);
    }

    default <K> Split<T> stratifiedSplit(
            Function<? super T, ? extends K> labelExtractor,
            double trainFraction,
            long seed) {
        return GenericDatasetSplits.stratifiedSplit(this, labelExtractor, trainFraction, seed);
    }

    default <K> ThreeWaySplit<T> stratifiedSplit(
            Function<? super T, ? extends K> labelExtractor,
            double trainFraction,
            double validationFraction,
            long seed) {
        return GenericDatasetSplits.stratifiedSplit(this, labelExtractor, trainFraction, validationFraction, seed);
    }

    default <K> List<Fold<T>> stratifiedKFold(
            Function<? super T, ? extends K> labelExtractor,
            int folds,
            long seed) {
        return GenericDatasetSplits.stratifiedKFold(this, labelExtractor, folds, seed);
    }

    default <K> List<Fold<T>> repeatedStratifiedKFold(
            Function<? super T, ? extends K> labelExtractor,
            int folds,
            int repeats,
            long seed) {
        return GenericDatasetSplits.repeatedStratifiedKFold(this, labelExtractor, folds, repeats, seed);
    }

    record Sample(GradTensor input, GradTensor label) {}

    record Pair<L, R>(L left, R right) {}

    record Indexed<T>(int index, T value) {
        public Indexed {
            if (index < 0) {
                throw new IllegalArgumentException("index must be non-negative");
            }
        }
    }

    record Window<T>(List<T> inputs, List<T> targets) {
        public Window {
            Objects.requireNonNull(inputs, "inputs must not be null");
            Objects.requireNonNull(targets, "targets must not be null");
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("inputs must not be empty");
            }
            if (targets.isEmpty()) {
                throw new IllegalArgumentException("targets must not be empty");
            }
            inputs = Collections.unmodifiableList(new ArrayList<>(inputs));
            targets = Collections.unmodifiableList(new ArrayList<>(targets));
        }
    }

    record Split<T>(Dataset<T> train, Dataset<T> validation) {
        public Split {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }
    }

    record ThreeWaySplit<T>(Dataset<T> train, Dataset<T> validation, Dataset<T> test) {
        public ThreeWaySplit {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
            Objects.requireNonNull(test, "test must not be null");
        }
    }

    record Fold<T>(int foldIndex, int foldCount, Dataset<T> train, Dataset<T> validation) {
        public Fold {
            if (foldCount < 1) {
                throw new IllegalArgumentException("foldCount must be at least 1");
            }
            if (foldIndex < 0 || foldIndex >= foldCount) {
                throw new IllegalArgumentException("foldIndex must be between 0 and foldCount - 1");
            }
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }
    }

    final class InMemoryDataset<T> implements Dataset<T> {
        private final List<T> items;

        private InMemoryDataset(List<? extends T> items) {
            Objects.requireNonNull(items, "items must not be null");
            this.items = Collections.unmodifiableList(new ArrayList<>(items));
        }

        @Override
        public int size() {
            return items.size();
        }

        @Override
        public T get(int index) {
            return items.get(index);
        }
    }

    final class MappedDataset<S, R> implements Dataset<R> {
        private final Dataset<? extends S> source;
        private final Function<? super S, ? extends R> mapper;

        private MappedDataset(Dataset<? extends S> source, Function<? super S, ? extends R> mapper) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            this.mapper = Objects.requireNonNull(mapper, "mapper must not be null");
        }

        @Override
        public int size() {
            return source.size();
        }

        @Override
        public R get(int index) {
            return mapper.apply(source.get(index));
        }
    }

    final class FilteredDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final List<Integer> indices;

        private FilteredDataset(Dataset<? extends T> source, Predicate<? super T> predicate) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            Objects.requireNonNull(predicate, "predicate must not be null");
            List<Integer> selected = new ArrayList<>();
            for (int i = 0; i < source.size(); i++) {
                if (predicate.test(source.get(i))) {
                    selected.add(i);
                }
            }
            this.indices = Collections.unmodifiableList(selected);
        }

        @Override
        public int size() {
            return indices.size();
        }

        @Override
        public T get(int index) {
            return source.get(indices.get(index));
        }
    }

    final class IndexedDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final List<Integer> indices;

        IndexedDataset(Dataset<? extends T> source, List<Integer> indices) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            Objects.requireNonNull(indices, "indices must not be null");
            this.indices = Collections.unmodifiableList(new ArrayList<>(indices));
        }

        @Override
        public int size() {
            return indices.size();
        }

        @Override
        public T get(int index) {
            return source.get(indices.get(index));
        }
    }

    private static List<Integer> shuffledIndices(int size, long seed) {
        List<Integer> indices = new ArrayList<>(IntStream.range(0, size).boxed().toList());
        Collections.shuffle(indices, new Random(seed));
        return indices;
    }

    private static void requireTrainFraction(double trainFraction) {
        if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
        }
    }

    private static int[] threeWayCounts(int size, double trainFraction, double validationFraction) {
        if (!Double.isFinite(trainFraction) || !Double.isFinite(validationFraction)) {
            throw new IllegalArgumentException("split fractions must be finite");
        }
        if (trainFraction <= 0.0 || validationFraction <= 0.0 || trainFraction + validationFraction >= 1.0) {
            throw new IllegalArgumentException(
                    "trainFraction and validationFraction must be positive and leave a positive test fraction");
        }
        if (size < 3) {
            throw new IllegalArgumentException(
                    "dataset must contain at least three samples for train/validation/test split");
        }
        int trainCount = (int) Math.round(size * trainFraction);
        int validationCount = (int) Math.round(size * validationFraction);
        trainCount = Math.max(1, Math.min(size - 2, trainCount));
        validationCount = Math.max(1, Math.min(size - trainCount - 1, validationCount));
        return new int[] {trainCount, validationCount, size - trainCount - validationCount};
    }
}
