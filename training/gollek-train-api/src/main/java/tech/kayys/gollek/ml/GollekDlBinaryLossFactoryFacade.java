package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.BinaryFocalWithLogitsLoss;

/**
 * Binary and multilabel logit loss helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlBinaryLossFactoryFacade extends GollekDlZeroInflatedLossFactoryFacade {
    protected GollekDlBinaryLossFactoryFacade() {
    }

    public static BCEWithLogitsLoss bceWithLogitsLoss() {
        return new BCEWithLogitsLoss();
    }

    public static BCEWithLogitsLoss bceWithLogitsLoss(float positiveWeight) {
        return new BCEWithLogitsLoss(positiveWeight);
    }

    public static BCEWithLogitsLoss bceWithLogitsLoss(float[] positiveWeights) {
        return new BCEWithLogitsLoss(positiveWeights);
    }

    public static BCEWithLogitsLoss binaryCrossEntropyWithLogits() {
        return bceWithLogitsLoss();
    }

    public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float positiveWeight) {
        return bceWithLogitsLoss(positiveWeight);
    }

    public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float[] positiveWeights) {
        return bceWithLogitsLoss(positiveWeights);
    }

    public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss() {
        return new BinaryFocalWithLogitsLoss();
    }

    public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(float gamma, float alpha) {
        return new BinaryFocalWithLogitsLoss(gamma, alpha);
    }

    public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
            float gamma,
            float alpha,
            float positiveWeight) {
        return new BinaryFocalWithLogitsLoss(gamma, alpha, positiveWeight);
    }

    public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
            float gamma,
            float alpha,
            float[] positiveWeights) {
        return new BinaryFocalWithLogitsLoss(gamma, alpha, positiveWeights);
    }
}
