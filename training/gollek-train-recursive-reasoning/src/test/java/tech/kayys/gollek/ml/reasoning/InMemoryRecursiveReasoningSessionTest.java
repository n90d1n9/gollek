package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class InMemoryRecursiveReasoningSessionTest {
    @Test
    void fitProducesSummaryAndReport() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .parallelSamples(2)
                .supervisionSteps(2)
                .transitionsPerSupervisionStep(1)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("arc-agi", latent, config);

        InMemoryRecursiveReasoningSession session = new InMemoryRecursiveReasoningSession(
                context,
                latent,
                (previous, ignored) -> new RecursiveReasoningTransitionResult(
                        new RecursiveReasoningState(
                                "state-" + previous.sampleIndex() + "-" + (previous.transitionIndex() + 1),
                                previous.supervisionStep(),
                                previous.transitionIndex() + 1,
                                previous.sampleIndex(),
                                previous.latentState(),
                                Map.of()),
                        -0.25,
                        (double) previous.sampleIndex()));

        assertFalse(session.isStopped());
        TrainingSummary summary = session.fit();
        RecursiveReasoningReport report = session.report();

        assertNotNull(summary);
        assertEquals("generative-recursive-reasoning", report.familyId());
        assertEquals("arc-agi", report.taskId());
        assertEquals(2, report.summary().exploredTrajectoryCount());
        assertEquals(1, report.selectedTrajectoryIndex());
        assertEquals(1.0, report.selectedRewardScore());
        assertEquals(2, session.activeSampleCount());
        assertEquals(1, session.currentTransitionIndex());
        assertEquals(1, session.currentSupervisionStep());
    }

    @Test
    void stoppedSessionRejectsFit() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder().build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("sudoku", latent, config);
        InMemoryRecursiveReasoningSession session = new InMemoryRecursiveReasoningSession(
                context,
                latent,
                (previous, ignored) -> new RecursiveReasoningTransitionResult(previous, 0.0, 0.0));

        session.stop();
        assertTrue(session.isStopped());
        assertThrows(IllegalStateException.class, session::fit);
    }

    private static final class DummyTensor implements Tensor {
        @Override
        public Shape shape() { return new Shape(1, 1); }
        @Override
        public DeviceType device() { return DeviceType.CPU; }
        @Override
        public DType dtype() { return DType.F32; }
        @Override
        public ComputeBackend backend() { return null; }
        @Override
        public Tensor add(Tensor other) { return this; }
        @Override
        public Tensor sub(Tensor other) { return this; }
        @Override
        public Tensor mul(Tensor other) { return this; }
        @Override
        public Tensor mul(float scalar) { return this; }
        @Override
        public Tensor div(float scalar) { return this; }
        @Override
        public Tensor matmul(Tensor other) { return this; }
        @Override
        public Tensor reshape(long... newShape) { return this; }
        @Override
        public Tensor softmax() { return this; }
        @Override
        public Tensor slice(long[] offsets, long[] sizes) { return this; }
        @Override
        public Tensor pow(float exponent) { return this; }
        @Override
        public Tensor mean() { return this; }
        @Override
        public Tensor abs() { return this; }
        @Override
        public Tensor crossEntropy(Tensor target) { return this; }
        @Override
        public Tensor binaryCrossEntropy(Tensor target) { return this; }
        @Override
        public Tensor div(Tensor other) { return this; }
        @Override
        public Tensor add(float scalar) { return this; }
        @Override
        public Tensor zerosLike() { return this; }
        @Override
        public Tensor sqrt() { return this; }
        @Override
        public Tensor cast(DType dtype) { return this; }
        @Override
        public Tensor to(DeviceType device) { return this; }
        @Override
        public float item() { return 0f; }
        @Override
        public void backward() {}
        @Override
        public Tensor grad() { return this; }
        @Override
        public void setGrad(Tensor grad) {}
        @Override
        public boolean requiresGrad() { return false; }
        @Override
        public void setRequiresGrad(boolean requiresGrad) {}
        @Override
        public Tensor relu() { return this; }
        @Override
        public Tensor sigmoid() { return this; }
        @Override
        public Tensor tanh() { return this; }
        @Override
        public Tensor log() { return this; }
        @Override
        public Tensor exp() { return this; }
        @Override
        public Tensor silu() { return this; }
        @Override
        public Tensor flatten() { return this; }
        @Override
        public Tensor unsqueeze(int dim) { return this; }
        @Override
        public Tensor squeeze() { return this; }
        @Override
        public Tensor transpose() { return this; }
        @Override
        public Tensor transpose(int dim0, int dim1) { return this; }
        @Override
        public long numel() { return 1L; }
    }
}
