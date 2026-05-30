package tech.kayys.gollek.ml;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.BinaryFocalWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.CausalLanguageModelingLoss;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.FocalLoss;
import tech.kayys.gollek.ml.nn.loss.GaussianNllLoss;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.L1Loss;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.NegativeBinomialNllLoss;
import tech.kayys.gollek.ml.nn.loss.PinballLoss;
import tech.kayys.gollek.ml.nn.loss.PoissonNllLoss;
import tech.kayys.gollek.ml.nn.loss.PredictionIntervalLoss;
import tech.kayys.gollek.ml.nn.loss.SmoothL1Loss;
import tech.kayys.gollek.ml.nn.loss.TweedieNllLoss;
import tech.kayys.gollek.ml.nn.loss.ZeroInflatedNegativeBinomialNllLoss;
import tech.kayys.gollek.ml.nn.loss.ZeroInflatedPoissonNllLoss;

class GollekDLLossFactoryTest {

    @Test
    void exposesClassificationLossFactoriesOnGollekFacade() {
        assertInstanceOf(CrossEntropyLoss.class, Gollek.DL.crossEntropy());
        assertInstanceOf(CrossEntropyLoss.class, Gollek.DL.crossEntropy(new float[] {1.0f, 2.0f}));
        assertInstanceOf(CausalLanguageModelingLoss.class, Gollek.DL.causalLanguageModelingLoss());
        assertInstanceOf(CausalLanguageModelingLoss.class, Gollek.DL.causalLanguageModelingLoss(-1.0f));
        assertInstanceOf(FocalLoss.class, Gollek.DL.focalLoss());
        assertInstanceOf(FocalLoss.class, Gollek.DL.focalLoss(2.0f));
        assertInstanceOf(FocalLoss.class, Gollek.DL.focalLoss(2.0f, 0.25f));
        assertInstanceOf(FocalLoss.class, Gollek.DL.focalLoss(2.0f, new float[] {1.0f, 2.0f}));
    }

    @Test
    void exposesRegressionLossFactoriesOnGollekFacade() {
        assertInstanceOf(MSELoss.class, Gollek.DL.mseLoss());
        assertInstanceOf(L1Loss.class, Gollek.DL.l1Loss());
        assertInstanceOf(HuberLoss.class, Gollek.DL.huberLoss());
        assertInstanceOf(HuberLoss.class, Gollek.DL.huberLoss(1.5f));
        assertInstanceOf(SmoothL1Loss.class, Gollek.DL.smoothL1Loss());
        assertInstanceOf(SmoothL1Loss.class, Gollek.DL.smoothL1Loss(0.75f));
        assertInstanceOf(PinballLoss.class, Gollek.DL.pinballLoss());
        assertInstanceOf(PinballLoss.class, Gollek.DL.quantileLoss(0.9));
        assertInstanceOf(PinballLoss.class, Gollek.DL.quantileLoss(0.1, 0.5, 0.9));
        assertInstanceOf(PinballLoss.class, Gollek.DL.predictionIntervalLoss(0.1, 0.9));
        assertInstanceOf(PredictionIntervalLoss.class, Gollek.DL.intervalScoreLoss());
        assertInstanceOf(PredictionIntervalLoss.class, Gollek.DL.winklerLoss(0.2, 12.0));
        assertInstanceOf(GaussianNllLoss.class, Gollek.DL.gaussianNllLoss());
        assertInstanceOf(GaussianNllLoss.class, Gollek.DL.heteroscedasticGaussianLoss(true));
    }

    @Test
    void exposesCountAndZeroInflatedLossFactoriesOnGollekFacade() {
        assertInstanceOf(PoissonNllLoss.class, Gollek.DL.poissonNllLoss());
        assertInstanceOf(PoissonNllLoss.class, Gollek.DL.poissonNllLoss(true, true, 1e-8));
        assertInstanceOf(PoissonNllLoss.class, Gollek.DL.countNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Gollek.DL.tweedieNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Gollek.DL.compoundPoissonGammaLoss(1.5));
        assertInstanceOf(NegativeBinomialNllLoss.class, Gollek.DL.negativeBinomialNllLoss());
        assertInstanceOf(NegativeBinomialNllLoss.class, Gollek.DL.overdispersedCountNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Gollek.DL.zeroInflatedPoissonNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Gollek.DL.zipNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Gollek.DL.excessZeroCountNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Gollek.DL.zeroInflatedNegativeBinomialNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Gollek.DL.zinbNllLoss());
        assertInstanceOf(
                ZeroInflatedNegativeBinomialNllLoss.class,
                Gollek.DL.excessZeroOverdispersedCountNllLoss());
    }

    @Test
    void exposesBinaryLogitLossFactoriesOnGollekFacade() {
        assertInstanceOf(BCEWithLogitsLoss.class, Gollek.DL.bceWithLogitsLoss());
        assertInstanceOf(BCEWithLogitsLoss.class, Gollek.DL.bceWithLogitsLoss(3.0f));
        assertInstanceOf(BCEWithLogitsLoss.class, Gollek.DL.bceWithLogitsLoss(new float[] {2.0f, 3.0f}));
        assertInstanceOf(BCEWithLogitsLoss.class, Gollek.DL.binaryCrossEntropyWithLogits());
        assertInstanceOf(BCEWithLogitsLoss.class, Gollek.DL.binaryCrossEntropyWithLogits(2.0f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Gollek.DL.binaryFocalWithLogitsLoss());
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Gollek.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Gollek.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, 3.0f));
        assertInstanceOf(
                BinaryFocalWithLogitsLoss.class,
                Gollek.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, new float[] {2.0f, 3.0f}));
    }
}
