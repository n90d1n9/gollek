package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

final class TensorCollators {
    private TensorCollators() {
    }

    static DataLoader.CollateFn defaultPairCollate() {
        return TensorCollators::collateFirstInputAndLabel;
    }

    static Function<List<Dataset.Sample>, DataLoader.Batch> sampleBatchCollate() {
        return TensorCollators::collateSamples;
    }

    static Function<List<Dataset.Pair<GradTensor, GradTensor>>, DataLoader.Batch> tensorPairBatchCollate() {
        return TensorCollators::collatePairs;
    }

    static Function<List<Dataset.Sample>, DataLoader.PaddedBatch> paddedSampleBatchCollate(
            float inputPadValue,
            float labelPadValue) {
        return batch -> collatePaddedSamples(batch, inputPadValue, labelPadValue);
    }

    static Function<List<Dataset.Pair<GradTensor, GradTensor>>, DataLoader.PaddedBatch> paddedTensorPairBatchCollate(
            float inputPadValue,
            float labelPadValue) {
        return batch -> collatePaddedPairs(batch, inputPadValue, labelPadValue);
    }

    private static DataLoader.Batch collateFirstInputAndLabel(
            List<Integer> indices,
            DataLoader.TensorDatasetAdapter dataset) {
        int numSamples = dataset.size(indices);
        int numTensors = dataset.get(0).length;
        if (numTensors < 2) {
            throw new IllegalStateException("TensorDataset must have at least 2 tensors (input and label)");
        }

        GradTensor[] inputSamples = new GradTensor[numSamples];
        GradTensor[] labelSamples = new GradTensor[numSamples];

        for (int i = 0; i < numSamples; i++) {
            GradTensor[] sample = dataset.get(indices.get(i));
            inputSamples[i] = sample[0];
            labelSamples[i] = sample[1];
        }

        return new DataLoader.Batch(
                stackBatch("inputs", inputSamples),
                stackBatch("labels", labelSamples));
    }

    private static DataLoader.Batch collateSamples(List<Dataset.Sample> batch) {
        requireBatch(batch);
        GradTensor[] inputSamples = new GradTensor[batch.size()];
        GradTensor[] labelSamples = new GradTensor[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            Dataset.Sample sample = Objects.requireNonNull(batch.get(i), "batch sample must not be null");
            inputSamples[i] = Objects.requireNonNull(sample.input(), "sample input tensor must not be null");
            labelSamples[i] = Objects.requireNonNull(sample.label(), "sample label tensor must not be null");
        }

        return new DataLoader.Batch(
                stackBatch("inputs", inputSamples),
                stackBatch("labels", labelSamples));
    }

    private static DataLoader.Batch collatePairs(List<Dataset.Pair<GradTensor, GradTensor>> batch) {
        requireBatch(batch);
        GradTensor[] inputSamples = new GradTensor[batch.size()];
        GradTensor[] labelSamples = new GradTensor[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            Dataset.Pair<GradTensor, GradTensor> pair =
                    Objects.requireNonNull(batch.get(i), "batch pair must not be null");
            inputSamples[i] = Objects.requireNonNull(pair.left(), "pair input tensor must not be null");
            labelSamples[i] = Objects.requireNonNull(pair.right(), "pair label tensor must not be null");
        }

        return new DataLoader.Batch(
                stackBatch("inputs", inputSamples),
                stackBatch("labels", labelSamples));
    }

    private static DataLoader.PaddedBatch collatePaddedSamples(
            List<Dataset.Sample> batch,
            float inputPadValue,
            float labelPadValue) {
        requireBatch(batch);
        GradTensor[] inputSamples = new GradTensor[batch.size()];
        GradTensor[] labelSamples = new GradTensor[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            Dataset.Sample sample = Objects.requireNonNull(batch.get(i), "batch sample must not be null");
            inputSamples[i] = Objects.requireNonNull(sample.input(), "sample input tensor must not be null");
            labelSamples[i] = Objects.requireNonNull(sample.label(), "sample label tensor must not be null");
        }

        PaddedTensor inputs = padFirstDimension("inputs", inputSamples, inputPadValue);
        PaddedTensor labels = padFirstDimension("labels", labelSamples, labelPadValue);
        return new DataLoader.PaddedBatch(
                inputs.values(),
                labels.values(),
                inputs.mask(),
                labels.mask(),
                inputs.lengths(),
                labels.lengths());
    }

    private static DataLoader.PaddedBatch collatePaddedPairs(
            List<Dataset.Pair<GradTensor, GradTensor>> batch,
            float inputPadValue,
            float labelPadValue) {
        requireBatch(batch);
        GradTensor[] inputSamples = new GradTensor[batch.size()];
        GradTensor[] labelSamples = new GradTensor[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            Dataset.Pair<GradTensor, GradTensor> pair =
                    Objects.requireNonNull(batch.get(i), "batch pair must not be null");
            inputSamples[i] = Objects.requireNonNull(pair.left(), "pair input tensor must not be null");
            labelSamples[i] = Objects.requireNonNull(pair.right(), "pair label tensor must not be null");
        }

        PaddedTensor inputs = padFirstDimension("inputs", inputSamples, inputPadValue);
        PaddedTensor labels = padFirstDimension("labels", labelSamples, labelPadValue);
        return new DataLoader.PaddedBatch(
                inputs.values(),
                labels.values(),
                inputs.mask(),
                labels.mask(),
                inputs.lengths(),
                labels.lengths());
    }

    private static void requireBatch(List<?> batch) {
        Objects.requireNonNull(batch, "batch must not be null");
        if (batch.isEmpty()) {
            throw new IllegalArgumentException("batch must not be empty");
        }
    }

    private static GradTensor stackBatch(String name, GradTensor[] tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException(name + " batch must not be empty");
        }
        for (GradTensor tensor : tensors) {
            Objects.requireNonNull(tensor, name + " tensor must not be null");
        }
        return GradTensor.stack(0, tensors);
    }

    private static PaddedTensor padFirstDimension(String name, GradTensor[] tensors, float padValue) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException(name + " batch must not be empty");
        }

        long[] baseShape = Objects.requireNonNull(tensors[0], name + " tensor must not be null").shape();
        if (baseShape.length == 0) {
            throw new IllegalArgumentException(name + " tensors must have rank >= 1 for padding");
        }

        int trailingSize = trailingSize(baseShape, name);
        int[] lengths = new int[tensors.length];
        int maxLength = 0;

        for (int i = 0; i < tensors.length; i++) {
            GradTensor tensor = Objects.requireNonNull(tensors[i], name + " tensor must not be null");
            long[] shape = tensor.shape();
            validatePaddingShape(name, baseShape, shape);
            lengths[i] = Math.toIntExact(shape[0]);
            maxLength = Math.max(maxLength, lengths[i]);
        }

        long[] paddedShape = new long[baseShape.length + 1];
        paddedShape[0] = tensors.length;
        paddedShape[1] = maxLength;
        System.arraycopy(baseShape, 1, paddedShape, 2, baseShape.length - 1);

        float[] padded = new float[Math.toIntExact(numel(paddedShape))];
        if (padValue != 0.0f) {
            java.util.Arrays.fill(padded, padValue);
        }
        float[] mask = new float[tensors.length * maxLength];

        for (int row = 0; row < tensors.length; row++) {
            float[] source = tensors[row].data();
            for (int position = 0; position < lengths[row]; position++) {
                int sourceOffset = position * trailingSize;
                int targetOffset = ((row * maxLength) + position) * trailingSize;
                System.arraycopy(source, sourceOffset, padded, targetOffset, trailingSize);
                mask[row * maxLength + position] = 1.0f;
            }
        }

        return new PaddedTensor(
                GradTensor.of(padded, paddedShape),
                GradTensor.of(mask, tensors.length, maxLength),
                lengths);
    }

    private static void validatePaddingShape(String name, long[] baseShape, long[] shape) {
        if (shape.length != baseShape.length) {
            throw new IllegalArgumentException(name + " tensors must have same rank");
        }
        for (int dim = 1; dim < shape.length; dim++) {
            if (shape[dim] != baseShape[dim]) {
                throw new IllegalArgumentException(name + " tensors must match on trailing dimensions");
            }
        }
    }

    private static int trailingSize(long[] shape, String name) {
        long size = 1L;
        for (int dim = 1; dim < shape.length; dim++) {
            size = Math.multiplyExact(size, shape[dim]);
        }
        try {
            return Math.toIntExact(size);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(name + " trailing dimensions are too large", e);
        }
    }

    private static long numel(long[] shape) {
        long size = 1L;
        for (long dim : shape) {
            size = Math.multiplyExact(size, dim);
        }
        return size;
    }

    private record PaddedTensor(GradTensor values, GradTensor mask, int[] lengths) {}
}
