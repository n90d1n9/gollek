package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class DiscreteTokenProjectingNextStateFactoryTest {
    @Test
    void attachesProjectedTokensUsingDefaultMetadataKeys() {
        DummyTensor tensor = new DummyTensor();
        RecursiveReasoningContext context = context(tensor);
        RecursiveReasoningState previous = state("root", tensor);
        GramTransitionSample sample = priorSample(tensor);
        int[] tokens = {1, 2, 3};
        DiscreteTokenProjectingNextStateFactory factory = new DiscreteTokenProjectingNextStateFactory(
                delegate(),
                (proposed, ignoredContext, ignoredSample) -> new DiscreteTokenProjection(
                        tokens,
                        "argmax",
                        Map.of("source", "test")));

        RecursiveReasoningState next = factory.nextState(previous, context, sample);
        tokens[0] = 9;

        assertEquals("metadata", next.metadata().get("existing"));
        assertEquals("argmax", next.metadata().get(DiscreteStateTokens.PROJECTION_MODE_METADATA_KEY));
        assertEquals("test", ((Map<?, ?>) next.metadata()
                .get(DiscreteStateTokens.PROJECTION_METADATA_KEY)).get("source"));
        assertArrayEquals(new int[] {1, 2, 3}, DiscreteStateTokens.defaultDecoder().decodeTokens(next));
    }

    @Test
    void graphColoringWrapperUsesGraphColoringMetadataKeys() {
        DummyTensor tensor = new DummyTensor();
        RecursiveReasoningContext context = context(tensor);
        RecursiveReasoningState previous = state("root", tensor);
        GramTransitionSample sample = priorSample(tensor);
        GraphColoringProblem problem = GraphColoringProblem.ofEdges(
                3,
                3,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2),
                GraphColoringEdge.of(0, 2));
        int[] tokens = GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens();
        GraphColoringTokenProjectingNextStateFactory factory = new GraphColoringTokenProjectingNextStateFactory(
                delegate(),
                problem,
                (proposed, ignoredContext, ignoredSample, ignoredProblem) -> new GraphColoringTokenProjectionResult(
                        tokens,
                        GraphColoringTokenProjectionMode.NODE_COLOR_ARGMAX,
                        Map.of("source", "graph-head")));

        RecursiveReasoningState next = factory.nextState(previous, context, sample);
        tokens[0] = 9;

        assertEquals(
                "NODE_COLOR_ARGMAX",
                next.metadata().get(GraphColoringStateTokens.PROJECTION_MODE_METADATA_KEY));
        assertEquals("graph-head", ((Map<?, ?>) next.metadata()
                .get(GraphColoringStateTokens.PROJECTION_METADATA_KEY)).get("source"));
        assertArrayEquals(
                GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens(),
                GraphColoringStateTokens.defaultDecoder().decodeTokens(next));
    }

    @Test
    void rejectsNullProjectionFromPredictionHead() {
        DummyTensor tensor = new DummyTensor();
        DiscreteTokenProjectingNextStateFactory factory = new DiscreteTokenProjectingNextStateFactory(
                delegate(),
                (proposed, ignoredContext, ignoredSample) -> null);

        assertThrows(
                NullPointerException.class,
                () -> factory.nextState(state("root", tensor), context(tensor), priorSample(tensor)));
    }

    private static GramNextStateFactory delegate() {
        return (previous, context, sample) -> new RecursiveReasoningState(
                "next",
                previous.supervisionStep(),
                previous.transitionIndex() + 1,
                previous.sampleIndex(),
                sample.latentState(),
                Map.of("existing", "metadata"));
    }

    private static RecursiveReasoningContext context(Tensor tensor) {
        return new RecursiveReasoningContext("structured", tensor, RecursiveReasoningConfig.builder().build());
    }

    private static RecursiveReasoningState state(String stateId, Tensor tensor) {
        return new RecursiveReasoningState(stateId, 0, 0, 0, tensor, Map.of());
    }

    private static GramTransitionSample priorSample(Tensor tensor) {
        GramLatentGaussian gaussian = new GramLatentGaussian(tensor, tensor);
        return new GramTransitionSample(
                GramTransitionMode.PRIOR,
                gaussian,
                null,
                gaussian,
                tensor,
                tensor,
                null,
                Map.of());
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
