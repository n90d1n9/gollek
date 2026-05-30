package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

class RecursiveReasoningContractsTest {
    @Test
    void configBuilderNormalizesMinimums() {
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .transitionsPerSupervisionStep(0)
                .supervisionSteps(0)
                .parallelSamples(0)
                .build();

        assertEquals(1, config.transitionsPerSupervisionStep());
        assertEquals(1, config.supervisionSteps());
        assertEquals(1, config.parallelSamples());
    }

    @Test
    void stateRequiresNonNegativeIndicesAndFrozenMetadata() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningState state = new RecursiveReasoningState(
                "s0",
                0,
                0,
                0,
                latent,
                Map.of("hypothesis", "a"));

        assertEquals("a", state.metadata().get("hypothesis"));
        assertThrows(UnsupportedOperationException.class, () -> state.metadata().put("x", "y"));
        assertThrows(IllegalArgumentException.class,
                () -> new RecursiveReasoningState("bad", -1, 0, 0, latent, Map.of()));
    }

    @Test
    void transitionContractCanReturnScoredSample() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .parallelSamples(4)
                .transitionsPerSupervisionStep(8)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("nqueens", latent, config);
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, latent, Map.of());

        StochasticLatentTransition transition = (prev, ctx) -> new RecursiveReasoningTransitionResult(
                new RecursiveReasoningState(
                        "next",
                        prev.supervisionStep(),
                        prev.transitionIndex() + 1,
                        prev.sampleIndex(),
                        prev.latentState(),
                        Map.of("taskId", ctx.taskId())),
                -0.25,
                0.9);

        RecursiveReasoningTransitionResult result = transition.sample(previous, context);
        assertEquals("next", result.nextState().stateId());
        assertEquals("nqueens", result.nextState().metadata().get("taskId"));
        assertEquals(-0.25, result.logProbability());
        assertEquals(0.9, result.rewardScore());
    }

    @Test
    void rolloutSummaryNormalizesMetadata() {
        RecursiveReasoningRolloutSummary summary =
                new RecursiveReasoningRolloutSummary(4, 2, "best", Map.of("selector", "lprm"));

        assertEquals(4, summary.exploredTrajectoryCount());
        assertEquals("best", summary.selectedStateId());
        assertEquals("lprm", summary.metadata().get("selector"));
        assertNotNull(summary.metadata());
        assertTrue(summary.completedTrajectoryCount() <= summary.exploredTrajectoryCount());
    }

    private static final class DummyTensor implements Tensor {
        @Override
        public Shape shape() {
            return new Shape(1, 1);
        }

        @Override
        public DeviceType device() {
            return DeviceType.CPU;
        }

        @Override
        public DType dtype() {
            return DType.F32;
        }

        @Override
        public ComputeBackend backend() {
            return null;
        }

        @Override
        public Tensor add(Tensor other) {
            return this;
        }

        @Override
        public Tensor sub(Tensor other) {
            return this;
        }

        @Override
        public Tensor mul(Tensor other) {
            return this;
        }

        @Override
        public Tensor mul(float scalar) {
            return this;
        }

        @Override
        public Tensor div(float scalar) {
            return this;
        }

        @Override
        public Tensor matmul(Tensor other) {
            return this;
        }

        @Override
        public Tensor reshape(long... newShape) {
            return this;
        }

        @Override
        public Tensor softmax() {
            return this;
        }

        @Override
        public Tensor slice(long[] offsets, long[] sizes) {
            return this;
        }

        @Override
        public Tensor pow(float exponent) {
            return this;
        }

        @Override
        public Tensor mean() {
            return this;
        }

        @Override
        public Tensor abs() {
            return this;
        }

        @Override
        public Tensor crossEntropy(Tensor target) {
            return this;
        }

        @Override
        public Tensor binaryCrossEntropy(Tensor target) {
            return this;
        }

        @Override
        public Tensor div(Tensor other) {
            return this;
        }

        @Override
        public Tensor add(float scalar) {
            return this;
        }

        @Override
        public Tensor zerosLike() {
            return this;
        }

        @Override
        public Tensor sqrt() {
            return this;
        }

        @Override
        public Tensor cast(DType dtype) {
            return this;
        }

        @Override
        public Tensor to(DeviceType device) {
            return this;
        }

        @Override
        public float item() {
            return 0f;
        }

        @Override
        public void backward() {
        }

        @Override
        public Tensor grad() {
            return this;
        }

        @Override
        public void setGrad(Tensor grad) {
        }

        @Override
        public boolean requiresGrad() {
            return false;
        }

        @Override
        public void setRequiresGrad(boolean requiresGrad) {
        }

        @Override
        public Tensor relu() {
            return this;
        }

        @Override
        public Tensor sigmoid() {
            return this;
        }

        @Override
        public Tensor tanh() {
            return this;
        }

        @Override
        public Tensor log() {
            return this;
        }

        @Override
        public Tensor exp() {
            return this;
        }

        @Override
        public Tensor silu() {
            return this;
        }

        @Override
        public Tensor flatten() {
            return this;
        }

        @Override
        public Tensor unsqueeze(int dim) {
            return this;
        }

        @Override
        public Tensor squeeze() {
            return this;
        }

        @Override
        public Tensor transpose() {
            return this;
        }

        @Override
        public Tensor transpose(int dim0, int dim1) {
            return this;
        }

        @Override
        public long numel() {
            return 1L;
        }
    }
}
