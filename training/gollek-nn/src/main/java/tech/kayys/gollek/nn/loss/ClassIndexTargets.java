package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;

final class ClassIndexTargets {

    private ClassIndexTargets() {
    }

    static float[] requireVectorData(GradTensor targets, int batch, String name) {
        long[] shape = targets.shape();
        if (shape.length != 1 || shape[0] != batch) {
            throw new IllegalArgumentException(
                    name + " must be a 1D class-index tensor with shape [" + batch + "], got shape: "
                            + Arrays.toString(shape));
        }
        return targets.data();
    }

    static int require(float value, int numClasses, int sampleIndex) {
        return require(value, numClasses, "sample " + sampleIndex);
    }

    static int require(float value, int numClasses, String location) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    "target class index at " + location + " must be finite, got: " + value);
        }
        int target = (int) value;
        if (value != target) {
            throw new IllegalArgumentException(
                    "target class index at " + location + " must be an integer, got: " + value);
        }
        if (target < 0 || target >= numClasses) {
            throw new IndexOutOfBoundsException(
                    "target class index at " + location + " " + target
                            + " out of range [0, " + (numClasses - 1) + "]");
        }
        return target;
    }
}
