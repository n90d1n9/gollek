package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.random.GaussianNoise;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class GramVariationalTransitionTest {
    @Test
    void samplesPosteriorWithKlAndBuildsNextState() {
        ScalarTensor initial = new ScalarTensor(1.0f);
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder().build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("nqueens", initial, config);
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());
        GramVariationalTransition transition = newTransition();

        GramVariationalTransitionResult result = transition.samplePosterior(previous, context, new ScalarTensor(99.0f));

        assertEquals(3.0f, result.nextState().latentState().item(), 1e-6f);
        assertEquals(0.5f, result.sample().klDivergence().item(), 1e-6f);
        assertEquals(GramTransitionMode.POSTERIOR, result.sample().mode());
        assertEquals("POSTERIOR", result.sample().metadata().get("mode"));
        assertEquals(0.0, result.logProbability(), 1e-9);
    }

    @Test
    void exposesPriorTransitionForReferenceRollouts() {
        ScalarTensor initial = new ScalarTensor(1.0f);
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder().build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("sudoku", initial, config);
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());

        RecursiveReasoningTransitionResult result = newTransition()
                .asPriorTransition()
                .sample(previous, context);

        assertEquals(2.0f, result.nextState().latentState().item(), 1e-6f);
        assertEquals("next-1", result.nextState().stateId());
    }

    @Test
    void priorSampleDoesNotRequirePosteriorOrKl() {
        ScalarTensor initial = new ScalarTensor(1.0f);
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder().build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("arc", initial, config);
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());

        GramVariationalTransitionResult result = newTransition().samplePrior(previous, context);

        assertNull(result.sample().posterior());
        assertNull(result.sample().klDivergence());
        assertEquals(GramTransitionMode.PRIOR, result.sample().mode());
    }

    @Test
    void posteriorInputRequiresTargetEmbedding() {
        ScalarTensor initial = new ScalarTensor(1.0f);
        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder().build();
        RecursiveReasoningContext context = new RecursiveReasoningContext("bad", initial, config);
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, initial, Map.of());

        assertThrows(
                NullPointerException.class,
                () -> GramTransitionInput.posterior(previous, context, initial, null));
    }

    @Test
    void seededGaussianNoiseIsDeterministicAndReusable() {
        assertArrayEquals(GaussianNoise.floats(8, 7L), GaussianNoise.floats(8, 7L));
        assertThrows(IllegalArgumentException.class, () -> GaussianNoise.floats(-1, 7L));
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
                "next-" + (previous.transitionIndex() + 1),
                previous.supervisionStep(),
                previous.transitionIndex() + 1,
                previous.sampleIndex(),
                sample.latentState(),
                Map.of("mode", sample.mode().name()));
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
