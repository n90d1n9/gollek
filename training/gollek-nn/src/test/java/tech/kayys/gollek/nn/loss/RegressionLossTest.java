package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RegressionLossTest {

    @Test
    void mseAndL1BackpropagateExpectedGradients() {
        float[] predData = {1.5f, -0.5f, 0.25f};
        GradTensor targets = GradTensor.of(new float[] {0.5f, -1.5f, 0.25f}, 3);

        GradTensor msePred = GradTensor.of(predData.clone(), 3).requiresGrad(true);
        new MSELoss().compute(msePred, targets).backward();
        assertArrayEquals(new float[] {2.0f / 3.0f, 2.0f / 3.0f, 0.0f}, msePred.grad().data(), 1e-6f);

        GradTensor l1Pred = GradTensor.of(predData.clone(), 3).requiresGrad(true);
        new L1Loss().compute(l1Pred, targets).backward();
        assertArrayEquals(new float[] {1.0f / 3.0f, 1.0f / 3.0f, 0.0f}, l1Pred.grad().data(), 1e-6f);
    }

    @Test
    void smoothL1UsesStandardBetaScaledFormulaAndGradients() {
        SmoothL1Loss loss = new SmoothL1Loss(0.5f);
        float[] predData = {0.25f, 1.0f, -2.0f};
        GradTensor pred = GradTensor.of(predData.clone(), 3).requiresGrad(true);
        GradTensor targets = GradTensor.zeros(3);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals((0.0625f + 0.75f + 1.75f) / 3.0f, output.item(), 1e-6f);
        assertArrayEquals(
                finiteDifferenceGradient(loss::compute, predData, targets.data()),
                pred.grad().data(),
                2e-3f);
    }

    @Test
    void huberLossBackpropagatesFiniteDifferenceGradients() {
        HuberLoss loss = new HuberLoss(0.75f);
        float[] predData = {0.25f, 1.5f, -1.0f};
        GradTensor pred = GradTensor.of(predData.clone(), 3).requiresGrad(true);
        GradTensor targets = GradTensor.zeros(3);

        loss.compute(pred, targets).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss::compute, predData, targets.data()),
                pred.grad().data(),
                2e-3f);
    }

    @Test
    void pinballLossBackpropagatesAsymmetricQuantileGradients() {
        PinballLoss loss = new PinballLoss(0.9f);
        GradTensor pred = GradTensor.of(new float[] {1.0f, 3.0f, 2.0f}, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {2.0f, 1.0f, 2.0f}, 3);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals((0.9f + 0.2f) / 3.0f, output.item(), 1e-6f);
        assertArrayEquals(new float[] {-0.9f / 3.0f, 0.1f / 3.0f, 0.0f}, pred.grad().data(), 1e-6f);
    }

    @Test
    void pinballLossSupportsMultipleQuantileOutputsPerTarget() {
        PinballLoss loss = new PinballLoss(0.1f, 0.5f, 0.9f);
        GradTensor pred = GradTensor.of(
                new float[] {
                        0.0f, 0.0f, 0.0f,
                        2.0f, 2.0f, 2.0f
                },
                2, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f, 1.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(0.5f, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        -0.1f / 6.0f, -0.5f / 6.0f, -0.9f / 6.0f,
                        0.9f / 6.0f, 0.5f / 6.0f, 0.1f / 6.0f
                },
                pred.grad().data(),
                1e-6f);
    }

    @Test
    void predictionIntervalLossScoresWidthMissesAndCrossedBounds() {
        PredictionIntervalLoss loss = new PredictionIntervalLoss(0.2f, 12.0f);
        GradTensor pred = GradTensor.of(
                new float[] {
                        0.0f, 2.0f,
                        2.0f, 3.0f,
                        5.0f, 4.0f
                },
                3, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f, 1.0f, 4.5f}, 3);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(34.0f / 3.0f, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        -1.0f / 3.0f, 1.0f / 3.0f,
                        9.0f / 3.0f, 1.0f / 3.0f,
                        21.0f / 3.0f, -21.0f / 3.0f
                },
                pred.grad().data(),
                1e-6f);
    }

    @Test
    void gaussianNllLossBackpropagatesMeanAndLogVarianceGradients() {
        GaussianNllLoss loss = new GaussianNllLoss(false);
        GradTensor pred = GradTensor.of(
                new float[] {
                        1.0f, 0.0f,
                        3.0f, 0.6931472f
                },
                2, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {2.0f, 1.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(0.9232868f, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        -0.5f, 0.0f,
                        0.5f, -0.25f
                },
                pred.grad().data(),
                1e-6f);
    }

    @Test
    void gaussianNllLossSupportsConstantAndClippedVarianceRange() {
        GaussianNllLoss loss = new GaussianNllLoss(true, -1.0f, 1.0f);
        GradTensor pred = GradTensor.of(new float[] {0.0f, -2.0f}, 1, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        float expected = 0.5f * (-1.0f + (float) Math.E) + 0.5f * (float) Math.log(2.0 * Math.PI);
        assertEquals(expected, output.item(), 1e-6f);
        assertArrayEquals(new float[] {-2.7182817f, 0.0f}, pred.grad().data(), 1e-6f);
    }

    @Test
    void poissonNllLossBackpropagatesLogRateGradients() {
        PoissonNllLoss loss = new PoissonNllLoss();
        GradTensor pred = GradTensor.of(new float[] {0.0f, 0.6931472f}, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f, 3.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(0.46027923f, output.item(), 1e-6f);
        assertArrayEquals(new float[] {0.0f, -0.5f}, pred.grad().data(), 1e-6f);
    }

    @Test
    void poissonNllLossSupportsRateInputsAndFullStirlingTerm() {
        PoissonNllLoss loss = new PoissonNllLoss(false, true, 1.0e-6f);
        GradTensor pred = GradTensor.of(new float[] {2.0f, 0.0f}, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {3.0f, 0.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        float stirling = 3.0f * (float) Math.log(3.0)
                - 3.0f
                + 0.5f * (float) Math.log(2.0 * Math.PI * 3.0);
        float expected = (2.0f - 3.0f * (float) Math.log(2.000001f) + stirling) / 2.0f;
        assertEquals(expected, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {(1.0f - 3.0f / 2.000001f) / 2.0f, 0.5f},
                pred.grad().data(),
                1e-6f);
    }

    @Test
    void tweedieNllLossBackpropagatesCompoundPoissonGammaGradients() {
        TweedieNllLoss loss = new TweedieNllLoss(1.5);
        GradTensor pred = GradTensor.of(new float[] {0.0f, 1.3862944f}, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f, 2.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(5.0f, output.item(), 1e-6f);
        assertArrayEquals(new float[] {0.0f, 0.5f}, pred.grad().data(), 1e-6f);
    }

    @Test
    void tweedieNllLossSupportsRawMeanGammaPower() {
        TweedieNllLoss loss = new TweedieNllLoss(2.0, false, 1.0e-6);
        GradTensor pred = GradTensor.of(new float[] {2.0f}, 1).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {4.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        float mean = 2.000001f;
        assertEquals((float) Math.log(mean) + 4.0f / mean, output.item(), 1e-6f);
        assertArrayEquals(new float[] {1.0f / mean - 4.0f / (mean * mean)}, pred.grad().data(), 1e-6f);
    }

    @Test
    void negativeBinomialNllLossBackpropagatesOverdispersedCountGradients() {
        NegativeBinomialNllLoss loss = new NegativeBinomialNllLoss(true, true);
        GradTensor pred = GradTensor.of(new float[] {(float) Math.log(2.0), 0.0f}, 1, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {2.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(1.9095426f, output.item(), 1e-6f);
        assertArrayEquals(new float[] {0.0f, -0.4013877f}, pred.grad().data(), 2e-5f);
    }

    @Test
    void negativeBinomialNllLossSupportsRawMeanAndInverseDispersionInputs() {
        NegativeBinomialNllLoss loss = new NegativeBinomialNllLoss(false, false, 1.0e-6);
        GradTensor pred = GradTensor.of(new float[] {1.0f, 2.0f}, 1, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {0.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        float mean = 1.000001f;
        float inverseDispersion = 2.000001f;
        assertEquals(2.0f * (float) Math.log((mean + inverseDispersion) / inverseDispersion),
                output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        inverseDispersion / (mean + inverseDispersion),
                        (float) (-Math.log(inverseDispersion) - 1.0
                                + Math.log(mean + inverseDispersion)
                                + inverseDispersion / (mean + inverseDispersion))
                },
                pred.grad().data(),
                2e-5f);
    }

    @Test
    void zeroInflatedPoissonNllLossBackpropagatesZeroAndPositiveCountBranches() {
        ZeroInflatedPoissonNllLoss loss = new ZeroInflatedPoissonNllLoss();
        GradTensor pred = GradTensor.of(
                new float[] {
                        0.0f, 0.0f,
                        (float) Math.log(2.0), 0.0f
                },
                2, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 2.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(0.8433691f, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        0.13447072f, -0.11552929f,
                        0.0f, 0.25f
                },
                pred.grad().data(),
                1e-6f);
    }

    @Test
    void zeroInflatedPoissonNllLossSupportsRawRatesAndFullCountConstant() {
        ZeroInflatedPoissonNllLoss loss = new ZeroInflatedPoissonNllLoss(false, true, 1.0e-6);
        GradTensor pred = GradTensor.of(new float[] {2.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {3.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        float rate = 2.000001f;
        float expected = (float) (Math.log(2.0) - 3.0 * Math.log(rate) + rate + Math.log(6.0));
        assertEquals(expected, output.item(), 1e-6f);
        assertArrayEquals(new float[] {1.0f - 3.0f / rate, 0.5f}, pred.grad().data(), 1e-6f);
    }

    @Test
    void zeroInflatedNegativeBinomialNllLossBackpropagatesMixtureAndOverdispersionGradients() {
        ZeroInflatedNegativeBinomialNllLoss loss = new ZeroInflatedNegativeBinomialNllLoss();
        GradTensor pred = GradTensor.of(
                new float[] {
                        (float) Math.log(2.0), 0.0f, 0.0f,
                        (float) Math.log(2.0), 0.0f, 0.0f
                },
                2, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 2.0f}, 2);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(1.1575038f, output.item(), 1e-6f);
        assertArrayEquals(
                new float[] {
                        0.08333333f, 0.0539932f, -0.125f,
                        0.0f, -0.20069385f, 0.25f
                },
                pred.grad().data(),
                2e-5f);
    }

    @Test
    void zeroInflatedNegativeBinomialNllLossSupportsRawInputsAndFullCountConstant() {
        ZeroInflatedNegativeBinomialNllLoss loss = new ZeroInflatedNegativeBinomialNllLoss(false, true);
        GradTensor pred = GradTensor.of(new float[] {2.0f, 1.0f, 0.0f}, 1, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {2.0f}, 1);

        GradTensor output = loss.compute(pred, targets);
        output.backward();

        assertEquals(2.6026897f, output.item(), 1e-6f);
        assertArrayEquals(new float[] {0.0f, -0.4013877f, 0.5f}, pred.grad().data(), 2e-5f);
    }

    @Test
    void regressionLossesRejectInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new PinballLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new PinballLoss(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new PinballLoss(1.0f));
        assertThrows(IllegalArgumentException.class, () -> PinballLoss.interval(0.9f, 0.1f));
        assertThrows(IllegalArgumentException.class, () -> new PredictionIntervalLoss(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new PredictionIntervalLoss(1.0f));
        assertThrows(IllegalArgumentException.class, () -> new PredictionIntervalLoss(0.1f, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new GaussianNllLoss(false, 1.0f, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new GaussianNllLoss(false, Float.NaN, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new PoissonNllLoss(true, false, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new TweedieNllLoss(0.9));
        assertThrows(IllegalArgumentException.class, () -> new TweedieNllLoss(2.1));
        assertThrows(IllegalArgumentException.class, () -> new TweedieNllLoss(1.5, true, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NegativeBinomialNllLoss(true, false, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new ZeroInflatedPoissonNllLoss(true, false, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedNegativeBinomialNllLoss(true, false, 0.0));

        GradTensor values = GradTensor.ones(2);
        GradTensor mismatch = GradTensor.ones(2, 1);
        GradTensor empty = GradTensor.zeros(0);
        GradTensor nonFinitePrediction = GradTensor.of(new float[] {Float.POSITIVE_INFINITY}, 1);
        GradTensor nonFiniteTarget = GradTensor.of(new float[] {Float.NaN}, 1);
        GradTensor finiteOne = GradTensor.ones(1);

        assertAllRegressionLossesReject(values, mismatch);
        assertAllRegressionLossesReject(empty, empty);
        assertAllRegressionLossesReject(nonFinitePrediction, finiteOne);
        assertAllRegressionLossesReject(finiteOne, nonFiniteTarget);
        assertThrows(IllegalArgumentException.class,
                () -> new PinballLoss(0.1f, 0.9f).compute(GradTensor.ones(2), GradTensor.ones(2)));
        assertThrows(IllegalArgumentException.class,
                () -> new PredictionIntervalLoss().compute(GradTensor.ones(2, 2), GradTensor.ones(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new PredictionIntervalLoss().compute(
                        GradTensor.of(new float[] {Float.POSITIVE_INFINITY, 1.0f}, 1, 2),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new GaussianNllLoss().compute(GradTensor.ones(2, 2), GradTensor.ones(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new GaussianNllLoss().compute(
                        GradTensor.of(new float[] {Float.NaN, 0.0f}, 1, 2),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new PoissonNllLoss().compute(GradTensor.ones(1), GradTensor.of(new float[] {-1.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new PoissonNllLoss(false).compute(
                        GradTensor.of(new float[] {-0.1f}, 1),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new TweedieNllLoss().compute(GradTensor.ones(1), GradTensor.of(new float[] {-1.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new TweedieNllLoss(1.5, false).compute(
                        GradTensor.of(new float[] {-0.1f}, 1),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new NegativeBinomialNllLoss().compute(GradTensor.ones(1, 2), GradTensor.ones(1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new NegativeBinomialNllLoss().compute(GradTensor.ones(1, 2),
                        GradTensor.of(new float[] {-1.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new NegativeBinomialNllLoss(false).compute(
                        GradTensor.of(new float[] {-0.1f, 1.0f}, 1, 2),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedPoissonNllLoss().compute(GradTensor.ones(1, 2), GradTensor.ones(1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedPoissonNllLoss().compute(GradTensor.ones(1, 2),
                        GradTensor.of(new float[] {-1.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedPoissonNllLoss(false).compute(
                        GradTensor.of(new float[] {-0.1f, 1.0f}, 1, 2),
                        GradTensor.ones(1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedNegativeBinomialNllLoss().compute(
                        GradTensor.ones(1, 3),
                        GradTensor.ones(1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedNegativeBinomialNllLoss().compute(
                        GradTensor.ones(1, 3),
                        GradTensor.of(new float[] {-1.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedNegativeBinomialNllLoss(false).compute(
                        GradTensor.of(new float[] {-0.1f, 1.0f, 0.0f}, 1, 3),
                        GradTensor.ones(1)));
    }

    private static void assertAllRegressionLossesReject(GradTensor predictions, GradTensor targets) {
        assertThrows(IllegalArgumentException.class, () -> new MSELoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new L1Loss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new PinballLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new PredictionIntervalLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new GaussianNllLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new PoissonNllLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new TweedieNllLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class,
                () -> new NegativeBinomialNllLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedPoissonNllLoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class,
                () -> new ZeroInflatedNegativeBinomialNllLoss().compute(predictions, targets));
    }

    private static float[] finiteDifferenceGradient(
            LossComputer loss,
            float[] predData,
            float[] targetData) {
        float[] grad = new float[predData.length];
        float eps = 1e-3f;
        for (int i = 0; i < predData.length; i++) {
            float[] plus = predData.clone();
            float[] minus = predData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.compute(
                            GradTensor.of(plus, plus.length),
                            GradTensor.of(targetData.clone(), targetData.length)).item()
                    - loss.compute(
                            GradTensor.of(minus, minus.length),
                            GradTensor.of(targetData.clone(), targetData.length)).item()) / (2.0f * eps);
        }
        return grad;
    }

    @FunctionalInterface
    private interface LossComputer {
        GradTensor compute(GradTensor predictions, GradTensor targets);
    }
}
