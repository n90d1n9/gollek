package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.CausalLanguageModelingLoss;
import tech.kayys.gollek.ml.nn.loss.FocalLoss;

/**
 * Classification loss construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlClassificationLossFactoryFacade extends GollekDlSchedulerFactoryFacade {
    protected GollekDlClassificationLossFactoryFacade() {
    }

    public static CrossEntropyLoss crossEntropy() {
        return new CrossEntropyLoss();
    }

    public static CrossEntropyLoss crossEntropy(float[] classWeights) {
        return new CrossEntropyLoss(classWeights);
    }

    public static CausalLanguageModelingLoss causalLanguageModelingLoss() {
        return new CausalLanguageModelingLoss();
    }

    public static CausalLanguageModelingLoss causalLanguageModelingLoss(float ignoreIndex) {
        return new CausalLanguageModelingLoss(ignoreIndex);
    }

    public static FocalLoss focalLoss() {
        return new FocalLoss();
    }

    public static FocalLoss focalLoss(float gamma) {
        return new FocalLoss(gamma);
    }

    public static FocalLoss focalLoss(float gamma, float alpha) {
        return new FocalLoss(gamma, alpha);
    }

    public static FocalLoss focalLoss(float gamma, float[] classWeights) {
        return new FocalLoss(gamma, classWeights);
    }
}
