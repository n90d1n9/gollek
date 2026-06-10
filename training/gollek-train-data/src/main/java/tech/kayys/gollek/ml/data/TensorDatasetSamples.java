package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class TensorDatasetSamples {
    private final List<GradTensor[]> samples;

    private TensorDatasetSamples(List<GradTensor[]> samples) {
        this.samples = samples;
    }

    static TensorDatasetSamples of(GradTensor[]... samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        List<GradTensor[]> copies = new ArrayList<>(samples.length);
        for (int i = 0; i < samples.length; i++) {
            copies.add(copySample(samples[i], "samples[" + i + "]"));
        }
        return new TensorDatasetSamples(List.copyOf(copies));
    }

    static TensorDatasetSamples fromBatchedTensors(GradTensor inputs, GradTensor targets) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(targets, "targets must not be null");
        if (inputs.shape().length == 0 || targets.shape().length == 0) {
            throw new IllegalArgumentException("inputs and targets must include a batch dimension");
        }
        if (inputs.shape()[0] != targets.shape()[0]) {
            throw new IllegalArgumentException("inputs and targets must have same batch dimension");
        }

        int sampleCount = Math.toIntExact(inputs.shape()[0]);
        long[] inputShape = inputs.shape();
        long[] targetShape = targets.shape();
        int inputStride = sampleStride(inputShape);
        int targetStride = sampleStride(targetShape);
        long[] inputSampleShape = Arrays.copyOfRange(inputShape, 1, inputShape.length);
        long[] targetSampleShape = Arrays.copyOfRange(targetShape, 1, targetShape.length);
        float[] inputData = inputs.data();
        float[] targetData = targets.data();
        List<GradTensor[]> samples = new ArrayList<>(sampleCount);

        for (int i = 0; i < sampleCount; i++) {
            int inputOffset = i * inputStride;
            int targetOffset = i * targetStride;
            samples.add(new GradTensor[] {
                GradTensor.of(
                        Arrays.copyOfRange(inputData, inputOffset, inputOffset + inputStride),
                        inputSampleShape),
                GradTensor.of(
                        Arrays.copyOfRange(targetData, targetOffset, targetOffset + targetStride),
                        targetSampleShape)
            });
        }
        return new TensorDatasetSamples(List.copyOf(samples));
    }

    int size() {
        return samples.size();
    }

    GradTensor[] get(int index) {
        return Arrays.copyOf(samples.get(index), samples.get(index).length);
    }

    private static int sampleStride(long[] shape) {
        int stride = 1;
        for (int d = 1; d < shape.length; d++) {
            stride = Math.multiplyExact(stride, Math.toIntExact(shape[d]));
        }
        return stride;
    }

    private static GradTensor[] copySample(GradTensor[] sample, String name) {
        Objects.requireNonNull(sample, name + " must not be null");
        GradTensor[] copy = Arrays.copyOf(sample, sample.length);
        for (int i = 0; i < copy.length; i++) {
            Objects.requireNonNull(copy[i], name + "[" + i + "] must not be null");
        }
        return copy;
    }
}
