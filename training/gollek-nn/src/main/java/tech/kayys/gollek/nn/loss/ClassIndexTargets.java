package tech.kayys.gollek.ml.nn.loss;

final class ClassIndexTargets {

    private ClassIndexTargets() {
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
