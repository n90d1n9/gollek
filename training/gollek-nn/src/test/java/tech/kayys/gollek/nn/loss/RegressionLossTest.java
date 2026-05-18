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
    void regressionLossesRejectInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(0.0f));

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
    }

    private static void assertAllRegressionLossesReject(GradTensor predictions, GradTensor targets) {
        assertThrows(IllegalArgumentException.class, () -> new MSELoss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new L1Loss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new SmoothL1Loss().compute(predictions, targets));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss().compute(predictions, targets));
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
