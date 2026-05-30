package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class ReferenceRecursiveReasoningRolloutExecutorTest {
    @Test
    void executesConfiguredParallelRolloutAndTracksAllStates() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .parallelSamples(3)
                .supervisionSteps(2)
                .transitionsPerSupervisionStep(2)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("sudoku", latent, config);

        ReferenceRecursiveReasoningRolloutExecutor executor = new ReferenceRecursiveReasoningRolloutExecutor();
        RecursiveReasoningRolloutResult result = executor.execute(latent, context, (previous, ignored) -> {
            int nextTransition = previous.transitionIndex() + 1;
            return new RecursiveReasoningTransitionResult(
                    new RecursiveReasoningState(
                            "s" + previous.sampleIndex() + "-t" + previous.supervisionStep() + "-" + nextTransition,
                            previous.supervisionStep(),
                            nextTransition,
                            previous.sampleIndex(),
                            previous.latentState(),
                            Map.of("from", previous.stateId())),
                    -0.5,
                    (double) previous.sampleIndex());
        });

        assertEquals(3, result.trajectories().size());
        assertEquals(5, result.selectedTrajectory().states().size());
        assertEquals(3, result.summary().exploredTrajectoryCount());
        assertEquals("reward-then-logprob", result.summary().metadata().get("selector"));
        assertNotNull(result.selectedTrajectory().finalState());
        assertEquals(1, result.selectedTrajectory().finalState().supervisionStep());
    }

    @Test
    void prefersHigherRewardThenHigherLogProbability() {
        Tensor latent = new DummyTensor();
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .parallelSamples(2)
                .supervisionSteps(1)
                .transitionsPerSupervisionStep(1)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("graph-coloring", latent, config);

        ReferenceRecursiveReasoningRolloutExecutor executor = new ReferenceRecursiveReasoningRolloutExecutor();
        RecursiveReasoningRolloutResult result = executor.execute(latent, context, (previous, ignored) -> {
            double reward = previous.sampleIndex() == 0 ? 0.2 : 0.9;
            double logProbability = previous.sampleIndex() == 0 ? -0.1 : -0.7;
            return new RecursiveReasoningTransitionResult(
                    new RecursiveReasoningState(
                            "candidate-" + previous.sampleIndex(),
                            previous.supervisionStep(),
                            previous.transitionIndex() + 1,
                            previous.sampleIndex(),
                            previous.latentState(),
                            Map.of()),
                    logProbability,
                    reward);
        });

        assertEquals(1, result.selectedTrajectoryIndex());
        assertEquals("candidate-1", result.selectedTrajectory().finalState().stateId());
        assertTrue((Integer) result.summary().metadata().get("selectedSampleIndex") >= 0);
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
