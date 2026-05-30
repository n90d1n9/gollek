package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class GenericDatasetWindows {
    private GenericDatasetWindows() {
    }

    static <T> Dataset<Dataset.Window<T>> windowed(
            Dataset<T> source,
            int inputSize,
            int targetSize,
            int stride) {
        Objects.requireNonNull(source, "source must not be null");
        requireWindowConfig(inputSize, targetSize, stride);
        return new WindowedDataset<>(source, inputSize, targetSize, stride);
    }

    private static void requireWindowConfig(int inputSize, int targetSize, int stride) {
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be at least 1, got: " + inputSize);
        }
        if (targetSize < 1) {
            throw new IllegalArgumentException("targetSize must be at least 1, got: " + targetSize);
        }
        if (stride < 1) {
            throw new IllegalArgumentException("stride must be at least 1, got: " + stride);
        }
    }

    private static final class WindowedDataset<T> implements Dataset<Dataset.Window<T>> {
        private final Dataset<T> source;
        private final int inputSize;
        private final int targetSize;
        private final int stride;
        private final int size;

        private WindowedDataset(Dataset<T> source, int inputSize, int targetSize, int stride) {
            this.source = source;
            this.inputSize = inputSize;
            this.targetSize = targetSize;
            this.stride = stride;
            int availableStartSpan = source.size() - inputSize - targetSize;
            this.size = availableStartSpan < 0 ? 0 : (availableStartSpan / stride) + 1;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public Dataset.Window<T> get(int index) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException("index " + index + " out of bounds for size " + size);
            }
            int start = index * stride;
            return new Dataset.Window<>(
                    GenericDatasetWindows.slice(source, start, inputSize),
                    GenericDatasetWindows.slice(source, start + inputSize, targetSize));
        }
    }

    private static <T> List<T> slice(Dataset<T> source, int start, int length) {
        List<T> values = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            values.add(source.get(start + i));
        }
        return values;
    }
}
