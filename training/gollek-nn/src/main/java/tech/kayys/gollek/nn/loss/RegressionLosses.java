package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;

final class RegressionLosses {
    private RegressionLosses() {
    }

    static int requireSameFiniteNonEmpty(GradTensor predictions, GradTensor targets, String lossName) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (!Arrays.equals(predictionShape, targetShape)) {
            throw new IllegalArgumentException(
                    lossName + " predictions and targets shapes must match, got: "
                            + Arrays.toString(predictionShape) + " vs " + Arrays.toString(targetShape));
        }

        float[] predictionData = predictions.data();
        float[] targetData = targets.data();
        if (predictionData.length == 0) {
            throw new IllegalArgumentException(lossName + " requires at least one element");
        }
        for (int i = 0; i < predictionData.length; i++) {
            if (!Float.isFinite(predictionData[i])) {
                throw new IllegalArgumentException(
                        lossName + " predictions must be finite, got " + predictionData[i] + " at index " + i);
            }
            if (!Float.isFinite(targetData[i])) {
                throw new IllegalArgumentException(
                        lossName + " targets must be finite, got " + targetData[i] + " at index " + i);
            }
        }
        return predictionData.length;
    }
}
