package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class GramDeepSupervisionTrainerTest {
    @Test
    void fitsPosteriorTransitionsWithTruncatedDeepSupervisionObjective() {
        ScalarTensor initial = new ScalarTensor(0.0f);
        RecursiveReasoningConfig rolloutConfig = RecursiveReasoningConfig.builder()
                .supervisionSteps(2)
                .transitionsPerSupervisionStep(2)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("nqueens", initial, rolloutConfig);
        RecursiveReasoningState initialState = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());
        GramObjectiveConfig objectiveConfig = GramObjectiveConfig.builder()
                .reduction(GramObjectiveReduction.SUM)
                .klBeta(0.1)
                .latentProcessRewardWeight(0.5)
                .adaptiveComputationWeight(0.25)
                .build();
        GramDeepSupervisionTrainer trainer = new GramDeepSupervisionTrainer(
                newTransition(),
                input -> new GramTerminalLossTerms(
                        input.terminalState().latentState().item(),
                        input.supervisionStep(),
                        2.0,
                        Map.of("terminal", input.terminalState().stateId())),
                objectiveConfig);

        GramDeepSupervisionTrainingResult result = trainer.fit(
                initialState,
                context,
                new GramTrainingTarget(new ScalarTensor(42.0f), Map.of("kind", "synthetic")));

        assertEquals(4, result.transitions().size());
        assertEquals(5, result.trajectory().states().size());
        assertEquals(8.0f, result.trajectory().finalState().latentState().item(), 1e-6f);
        assertEquals(1, result.trajectory().finalState().supervisionStep());
        assertEquals(2, result.trajectory().finalState().transitionIndex());
        assertEquals(1.0, result.objective().klDivergence(), 1e-9);
        assertEquals(12.0, result.objective().reconstructionNll(), 1e-9);
        assertEquals(13.6, result.objective().totalLoss(), 1e-9);
        assertEquals(0, result.summary().bestValidationEpoch());
        assertEquals(4, result.summary().metadata().get("transitionCount"));
        assertEquals("synthetic", ((Map<?, ?>) result.metadata().get("targetMetadata")).get("kind"));
        assertEquals(
                "final-transition",
                result.objective().supervisionSteps().getFirst().metadata().get("klAggregation"));
    }

    @Test
    void canAggregateFullStepKlInsteadOfTruncatedFinalKl() {
        ScalarTensor initial = new ScalarTensor(0.0f);
        RecursiveReasoningConfig rolloutConfig = RecursiveReasoningConfig.builder()
                .supervisionSteps(1)
                .transitionsPerSupervisionStep(3)
                .build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("sudoku", initial, rolloutConfig);
        RecursiveReasoningState initialState = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());
        GramObjectiveConfig objectiveConfig = GramObjectiveConfig.builder()
                .reduction(GramObjectiveReduction.SUM)
                .klBeta(1.0)
                .truncatedSurrogate(false)
                .build();
        GramDeepSupervisionTrainer trainer = new GramDeepSupervisionTrainer(
                newTransition(),
                input -> new GramTerminalLossTerms(0.0, 0.0, 0.0, Map.of()),
                objectiveConfig);

        GramDeepSupervisionTrainingResult result = trainer.fit(
                initialState,
                context,
                new GramTrainingTarget(new ScalarTensor(1.0f), Map.of()));

        assertEquals(1.5, result.objective().klDivergence(), 1e-9);
        assertEquals(1.5, result.objective().totalLoss(), 1e-9);
        assertEquals(
                "all-transitions",
                result.objective().supervisionSteps().getFirst().metadata().get("klAggregation"));
    }

    @Test
    void targetRequiresEmbedding() {
        assertThrows(NullPointerException.class, () -> new GramTrainingTarget(null, Map.of()));
    }

    private GramVariationalTransition newTransition() {
        GramDeterministicTransition deterministic = (previous, ignored) -> previous.latentState().add(1.0f);
        GramTransitionDistributionHead prior = input -> new GramLatentGaussian(
                input.deterministicProposal(),
                new ScalarTensor(0.0f));
        GramTransitionDistributionHead posterior = input -> new GramLatentGaussian(
                input.deterministicProposal().add(1.0f),
                new ScalarTensor(0.0f));
        GramNoiseSampler noise = (distribution, input) -> new ScalarTensor(0.0f);
        GramNextStateFactory nextState = (previous, context, sample) -> new RecursiveReasoningState(
                "raw-" + sample.latentState().item(),
                99,
                99,
                previous.sampleIndex(),
                sample.latentState(),
                Map.of("raw", true));
        return new GramVariationalTransition(deterministic, prior, posterior, noise, nextState);
    }

    private record ScalarTensor(float value, Shape shape) implements Tensor {
        private ScalarTensor(float value) {
            this(value, new Shape(1));
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
            return new ScalarTensor(value + other.item(), shape);
        }

        @Override
        public Tensor sub(Tensor other) {
            return new ScalarTensor(value - other.item(), shape);
        }

        @Override
        public Tensor mul(Tensor other) {
            return new ScalarTensor(value * other.item(), shape);
        }

        @Override
        public Tensor mul(float scalar) {
            return new ScalarTensor(value * scalar, shape);
        }

        @Override
        public Tensor div(float scalar) {
            return new ScalarTensor(value / scalar, shape);
        }

        @Override
        public Tensor matmul(Tensor other) {
            return mul(other);
        }

        @Override
        public Tensor reshape(long... newShape) {
            return new ScalarTensor(value, new Shape(newShape));
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
            return new ScalarTensor((float) Math.pow(value, exponent), shape);
        }

        @Override
        public Tensor mean() {
            return this;
        }

        @Override
        public Tensor abs() {
            return new ScalarTensor(Math.abs(value), shape);
        }

        @Override
        public Tensor crossEntropy(Tensor target) {
            return sub(target).abs();
        }

        @Override
        public Tensor binaryCrossEntropy(Tensor target) {
            return crossEntropy(target);
        }

        @Override
        public Tensor div(Tensor other) {
            return new ScalarTensor(value / other.item(), shape);
        }

        @Override
        public Tensor add(float scalar) {
            return new ScalarTensor(value + scalar, shape);
        }

        @Override
        public Tensor zerosLike() {
            return new ScalarTensor(0.0f, shape);
        }

        @Override
        public Tensor sqrt() {
            return new ScalarTensor((float) Math.sqrt(value), shape);
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
            return value;
        }

        @Override
        public void backward() {
        }

        @Override
        public Tensor grad() {
            return zerosLike();
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
            return new ScalarTensor(Math.max(0.0f, value), shape);
        }

        @Override
        public Tensor sigmoid() {
            return new ScalarTensor((float) (1.0 / (1.0 + Math.exp(-value))), shape);
        }

        @Override
        public Tensor tanh() {
            return new ScalarTensor((float) Math.tanh(value), shape);
        }

        @Override
        public Tensor log() {
            return new ScalarTensor((float) Math.log(value), shape);
        }

        @Override
        public Tensor exp() {
            return new ScalarTensor((float) Math.exp(value), shape);
        }

        @Override
        public Tensor silu() {
            return mul(sigmoid());
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
            return shape.numel();
        }
    }
}
