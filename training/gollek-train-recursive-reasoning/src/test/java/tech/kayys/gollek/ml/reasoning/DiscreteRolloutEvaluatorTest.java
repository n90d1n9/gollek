package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class DiscreteRolloutEvaluatorTest {
    @Test
    void evaluatesCollectedTokenPredictionsAndSummarizesCoverage() {
        int[] first = {1, 2};
        int[] second = {3, 4};
        int[] duplicate = {1, 2};
        int[] invalid = {9, 9};
        RecursiveReasoningRolloutResult rollout = rollout(
                state(0, "state-0", first),
                state(1, "state-1", second),
                state(2, "state-2", duplicate),
                state(3, "state-3", invalid));

        DiscreteRolloutEvaluationReport report = DiscreteRolloutEvaluator.evaluate(
                rollout,
                DiscreteStateTokenDecoder.fromMetadata("tokens"),
                DiscreteRolloutEvaluatorTest::score,
                4);
        first[0] = 7;
        int[] exposed = report.evaluations().getFirst().tokens();
        exposed[1] = 7;

        assertEquals(4, report.evaluations().size());
        assertEquals(1, report.selectedTrajectoryIndex());
        assertEquals("state-1", report.selectedEvaluation().stateId());
        assertTrue(report.selectedEvaluation().valid());
        assertFalse(report.evaluations().get(3).valid());
        assertArrayEquals(new int[] {1, 2}, report.evaluations().getFirst().tokens());
        assertEquals(4, report.coverage().candidateCount());
        assertEquals(3, report.coverage().validCandidateCount());
        assertEquals(2, report.coverage().uniqueValidCandidateCount());
        assertEquals(1, report.coverage().duplicateValidCandidateCount());
        assertEquals(0.75, report.coverage().validRate(), 1e-9);
        assertEquals(0.5, report.coverage().coverageRate(), 1e-9);
        assertEquals(4, report.metadata().get("knownSolutionCount"));
        assertEquals("state-1", report.metadata().get("selectedStateId"));
    }

    @Test
    void canEvaluateAnAlreadyCollectedTokenReport() {
        RecursiveReasoningRolloutResult rollout = rollout(
                state(0, "state-0", new int[] {1}),
                state(1, "state-1", new int[] {9}));
        DiscreteRolloutTokenReport tokenReport = DiscreteRolloutTokenCollector.collectFinalStateTokens(
                rollout,
                DiscreteStateTokenDecoder.fromMetadata("tokens"));

        DiscreteRolloutEvaluationReport report = DiscreteRolloutEvaluator.evaluate(tokenReport, prediction ->
                prediction.tokens()[0] == 1
                        ? DiscreteTokenEvaluation.valid("one")
                        : DiscreteTokenEvaluation.invalid(1));

        assertEquals(2, report.coverage().candidateCount());
        assertEquals(1, report.coverage().validCandidateCount());
        assertTrue(Double.isNaN(report.coverage().coverageRate()));
    }

    @Test
    void rejectsNullCandidateEvaluation() {
        RecursiveReasoningRolloutResult rollout = rollout(state(0, "state-0", new int[] {1}));

        assertThrows(
                NullPointerException.class,
                () -> DiscreteRolloutEvaluator.evaluate(
                        rollout,
                        DiscreteStateTokenDecoder.fromMetadata("tokens"),
                        ignored -> null));
    }

    private static DiscreteTokenEvaluation score(DiscreteTrajectoryTokenPrediction prediction) {
        int[] tokens = prediction.tokens();
        if (tokens[0] == 9) {
            return DiscreteTokenEvaluation.invalid(2, Map.of("reason", "sentinel"));
        }
        return DiscreteTokenEvaluation.valid(Arrays.toString(tokens), Map.of("length", tokens.length));
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
