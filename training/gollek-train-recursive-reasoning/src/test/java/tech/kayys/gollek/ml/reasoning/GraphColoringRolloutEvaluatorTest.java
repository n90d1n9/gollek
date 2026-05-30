package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class GraphColoringRolloutEvaluatorTest {
    @Test
    void evaluatesRolloutFinalStatesAsGraphColoringCandidates() {
        GraphColoringProblem problem = triangle();
        int[] validTokens = GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens();
        int[] duplicateTokens = validTokens.clone();
        int[] invalidTokens = {1, 1, 0};
        RecursiveReasoningRolloutResult rollout = rollout(
                state(0, "state-0", validTokens),
                state(1, "state-1", duplicateTokens),
                state(2, "state-2", invalidTokens));

        GraphColoringRolloutEvaluationReport report = GraphColoringRolloutEvaluator.evaluate(
                problem,
                rollout,
                DiscreteStateTokenDecoder.fromMetadata("tokens"),
                6);

        assertEquals(3, report.evaluations().size());
        assertEquals(1, report.selectedTrajectoryIndex());
        assertEquals("state-1", report.selectedEvaluation().stateId());
        assertTrue(report.selectedEvaluation().valid());
        assertFalse(report.evaluations().get(2).valid());
        assertEquals(1, report.evaluations().get(2).evaluation().edgeConflictCount());
        assertEquals(1, report.evaluations().get(2).evaluation().uncoloredNodeCount());
        assertEquals(3, report.coverage().candidateCount());
        assertEquals(2, report.coverage().validCandidateCount());
        assertEquals(1, report.coverage().uniqueValidSolutionCount());
        assertEquals(1, report.coverage().duplicateValidCandidateCount());
        assertEquals(1.0 / 6.0, report.coverage().coverageRate(), 1e-9);
        assertEquals("state-1", report.metadata().get("selectedStateId"));

        validTokens[0] = 9;
        int[] exposed = report.selectedEvaluation().tokens();
        exposed[1] = 9;
        assertArrayEquals(GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens(), report.selectedEvaluation().tokens());
    }

    private static RecursiveReasoningRolloutResult rollout(RecursiveReasoningState... finalStates) {
        List<RecursiveReasoningTrajectory> trajectories = java.util.Arrays.stream(finalStates)
                .map(state -> new RecursiveReasoningTrajectory(
                        state.sampleIndex(),
                        List.of(state),
                        -state.sampleIndex(),
                        (double) state.sampleIndex()))
                .toList();
        RecursiveReasoningState selected = finalStates.length == 1 ? finalStates[0] : finalStates[1];
        return new RecursiveReasoningRolloutResult(
                trajectories,
                finalStates.length == 1 ? 0 : 1,
                new RecursiveReasoningRolloutSummary(
                        trajectories.size(),
                        trajectories.size(),
                        selected.stateId(),
                        Map.of("selector", "test")));
    }

    private static RecursiveReasoningState state(int sampleIndex, String stateId, int[] tokens) {
        return new RecursiveReasoningState(
                stateId,
                1,
                2,
                sampleIndex,
                new DummyTensor(),
                Map.of("tokens", tokens));
    }

    private static GraphColoringProblem triangle() {
        return GraphColoringProblem.ofEdges(
                3,
                3,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2),
                GraphColoringEdge.of(0, 2));
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
