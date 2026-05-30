package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;

final class TensorDatasetSamples {
    private TensorDatasetSamples() {
    }

    static List<GradTensor[]> fromTuples(GradTensor[]... samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        List<GradTensor[]> copied = new ArrayList<>(samples.length);
        for (int index = 0; index < samples.length; index++) {
            copied.add(copyTuple(samples[index], "sample " + index));
        }
        return List.copyOf(copied);
    }

    static List<GradTensor[]> fromList(List<GradTensor[]> samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        List<GradTensor[]> copied = new ArrayList<>(samples.size());
        for (int index = 0; index < samples.size(); index++) {
            copied.add(copyTuple(samples.get(index), "sample " + index));
        }
        return List.copyOf(copied);
    }

    static List<GradTensor[]> fromBatchedPair(GradTensor inputs, GradTensor targets) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(targets, "targets must not be null");
        long[] inputShape = inputs.shape();
        long[] targetShape = targets.shape();
        if (inputShape.length == 0 || targetShape.length == 0) {
            throw new IllegalArgumentException("inputs and targets must include a batch dimension");
        }
        if (inputShape[0] != targetShape[0]) {
            throw new IllegalArgumentException("inputs and targets must have same batch dimension");
        }

        int sampleCount = Math.toIntExact(inputShape[0]);
        int inputStride = sampleStride(inputShape);
        int targetStride = sampleStride(targetShape);
        long[] inputSampleShape = Arrays.copyOfRange(inputShape, 1, inputShape.length);
        long[] targetSampleShape = Arrays.copyOfRange(targetShape, 1, targetShape.length);
        float[] inputData = inputs.data();
        float[] targetData = targets.data();
        List<GradTensor[]> samples = new ArrayList<>(sampleCount);

        for (int index = 0; index < sampleCount; index++) {
            int inputOffset = index * inputStride;
            int targetOffset = index * targetStride;
            samples.add(new GradTensor[] {
                    GradTensor.of(
                            Arrays.copyOfRange(inputData, inputOffset, inputOffset + inputStride),
                            inputSampleShape),
                    GradTensor.of(
                            Arrays.copyOfRange(targetData, targetOffset, targetOffset + targetStride),
                            targetSampleShape)
            });
        }
        return List.copyOf(samples);
    }

    private static GradTensor[] copyTuple(GradTensor[] sample, String sampleName) {
        Objects.requireNonNull(sample, sampleName + " must not be null");
        GradTensor[] copied = Arrays.copyOf(sample, sample.length);
        for (int tensor = 0; tensor < copied.length; tensor++) {
            Objects.requireNonNull(copied[tensor], sampleName + " tensor " + tensor + " must not be null");
        }
        return copied;
    }

    private static int sampleStride(long[] shape) {
        long stride = 1L;
        for (int dimension = 1; dimension < shape.length; dimension++) {
            stride = Math.multiplyExact(stride, shape[dimension]);
        }
        return Math.toIntExact(stride);
    }
}
