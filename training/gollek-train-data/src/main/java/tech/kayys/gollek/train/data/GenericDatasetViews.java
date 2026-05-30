package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

final class GenericDatasetViews {
    private GenericDatasetViews() {}

    static <L, R> Dataset<Dataset.Pair<L, R>> zip(
            Dataset<? extends L> left,
            Dataset<? extends R> right) {
        Objects.requireNonNull(left, "left must not be null");
        Objects.requireNonNull(right, "right must not be null");
        if (left.size() != right.size()) {
            throw new IllegalArgumentException("datasets must have the same size to zip");
        }
        return new ZippedDataset<>(left, right);
    }

    static <T> Dataset<Dataset.Indexed<T>> enumerate(Dataset<? extends T> source) {
        return new EnumeratedDataset<>(source);
    }

    static <T> Dataset<T> shard(Dataset<? extends T> source, int shardCount, int shardIndex) {
        return new ShardedDataset<>(source, shardCount, shardIndex);
    }

    static <T> Dataset<T> shuffle(Dataset<? extends T> source, long seed) {
        Objects.requireNonNull(source, "source must not be null");
        List<Integer> indices = sequentialIndices(source.size());
        Collections.shuffle(indices, new Random(seed));
        return new IndexedView<>(source, indices);
    }

    static <T> Dataset<T> sample(Dataset<? extends T> source, int count, long seed) {
        Objects.requireNonNull(source, "source must not be null");
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative");
        }
        if (count > source.size()) {
            throw new IllegalArgumentException("count must be less than or equal to dataset size");
        }
        List<Integer> indices = sequentialIndices(source.size());
        Collections.shuffle(indices, new Random(seed));
        return new IndexedView<>(source, indices.subList(0, count));
    }

    static <T> Dataset<T> sampleWithReplacement(Dataset<? extends T> source, int count, long seed) {
        Objects.requireNonNull(source, "source must not be null");
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative");
        }
        if (count > 0 && source.size() == 0) {
            throw new IllegalArgumentException("cannot sample from an empty dataset");
        }
        Random random = new Random(seed);
        List<Integer> indices = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            indices.add(random.nextInt(source.size()));
        }
        return new IndexedView<>(source, indices);
    }

    static <T> Dataset<T> take(Dataset<? extends T> source, int count) {
        Objects.requireNonNull(source, "source must not be null");
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative");
        }
        return new OffsetDataset<>(source, 0, Math.min(count, source.size()));
    }

    static <T> Dataset<T> drop(Dataset<? extends T> source, int count) {
        Objects.requireNonNull(source, "source must not be null");
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative");
        }
        int start = Math.min(count, source.size());
        return new OffsetDataset<>(source, start, source.size() - start);
    }

    static <T> Dataset<T> slice(Dataset<? extends T> source, int fromInclusive, int toExclusive) {
        Objects.requireNonNull(source, "source must not be null");
        if (fromInclusive < 0) {
            throw new IllegalArgumentException("fromInclusive must be non-negative");
        }
        if (toExclusive < fromInclusive) {
            throw new IllegalArgumentException("toExclusive must be greater than or equal to fromInclusive");
        }
        if (toExclusive > source.size()) {
            throw new IllegalArgumentException("toExclusive must be less than or equal to dataset size");
        }
        return new OffsetDataset<>(source, fromInclusive, toExclusive - fromInclusive);
    }

    static <T> Dataset<T> repeat(Dataset<? extends T> source, int times) {
        return new RepeatedDataset<>(source, times);
    }

    static <T> Dataset<T> cache(Dataset<? extends T> source) {
        return new CachedDataset<>(source);
    }

    private static final class ZippedDataset<L, R> implements Dataset<Dataset.Pair<L, R>> {
        private final Dataset<? extends L> left;
        private final Dataset<? extends R> right;

        private ZippedDataset(Dataset<? extends L> left, Dataset<? extends R> right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public int size() {
            return left.size();
        }

        @Override
        public Dataset.Pair<L, R> get(int index) {
            checkIndex(index, size());
            return new Dataset.Pair<>(left.get(index), right.get(index));
        }
    }

    private static final class EnumeratedDataset<T> implements Dataset<Dataset.Indexed<T>> {
        private final Dataset<? extends T> source;

        private EnumeratedDataset(Dataset<? extends T> source) {
            this.source = Objects.requireNonNull(source, "source must not be null");
        }

        @Override
        public int size() {
            return source.size();
        }

        @Override
        public Dataset.Indexed<T> get(int index) {
            checkIndex(index, size());
            return new Dataset.Indexed<>(index, source.get(index));
        }
    }

    private static final class ShardedDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final int shardCount;
        private final int shardIndex;
        private final int size;

        private ShardedDataset(Dataset<? extends T> source, int shardCount, int shardIndex) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            if (shardCount < 1) {
                throw new IllegalArgumentException("shardCount must be at least 1");
            }
            if (shardIndex < 0 || shardIndex >= shardCount) {
                throw new IllegalArgumentException("shardIndex must be between 0 and shardCount - 1");
            }
            this.shardCount = shardCount;
            this.shardIndex = shardIndex;
            this.size = source.size() <= shardIndex ? 0 : ((source.size() - 1 - shardIndex) / shardCount) + 1;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public T get(int index) {
            checkIndex(index, size);
            return source.get(shardIndex + (index * shardCount));
        }
    }

    private static final class IndexedView<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final List<Integer> indices;

        private IndexedView(Dataset<? extends T> source, List<Integer> indices) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            this.indices = Collections.unmodifiableList(new ArrayList<>(indices));
        }

        @Override
        public int size() {
            return indices.size();
        }

        @Override
        public T get(int index) {
            checkIndex(index, indices.size());
            return source.get(indices.get(index));
        }
    }

    private static final class OffsetDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final int offset;
        private final int size;

        private OffsetDataset(Dataset<? extends T> source, int offset, int size) {
            this.source = source;
            this.offset = offset;
            this.size = size;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public T get(int index) {
            checkIndex(index, size);
            return source.get(offset + index);
        }
    }

    private static final class RepeatedDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final int size;

        private RepeatedDataset(Dataset<? extends T> source, int times) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            if (times < 0) {
                throw new IllegalArgumentException("times must be non-negative");
            }
            long repeatedSize = (long) source.size() * times;
            if (repeatedSize > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("repeated dataset is too large");
            }
            this.size = (int) repeatedSize;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public T get(int index) {
            checkIndex(index, size);
            return source.get(index % source.size());
        }
    }

    private static final class CachedDataset<T> implements Dataset<T> {
        private final Dataset<? extends T> source;
        private final Object[] values;
        private final boolean[] loaded;

        private CachedDataset(Dataset<? extends T> source) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            this.values = new Object[source.size()];
            this.loaded = new boolean[source.size()];
        }

        @Override
        public int size() {
            return values.length;
        }

        @Override
        @SuppressWarnings("unchecked")
        public synchronized T get(int index) {
            checkIndex(index, values.length);
            if (!loaded[index]) {
                values[index] = source.get(index);
                loaded[index] = true;
            }
            return (T) values[index];
        }
    }

    private static void checkIndex(int index, int size) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index " + index + " out of bounds for size " + size);
        }
    }

    private static List<Integer> sequentialIndices(int size) {
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        return indices;
    }
}
