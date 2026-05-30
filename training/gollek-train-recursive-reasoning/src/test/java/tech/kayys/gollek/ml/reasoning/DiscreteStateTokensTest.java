package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class DiscreteStateTokensTest {
    @Test
    void attachesAndDecodesDefaultTokenMetadata() {
        RecursiveReasoningState state = state(Map.of("keep", "yes"));
        int[] tokens = {1, 2, 3};

        RecursiveReasoningState updated = DiscreteStateTokens.attach(state, tokens);
        tokens[0] = 9;

        assertEquals("yes", updated.metadata().get("keep"));
        assertArrayEquals(new int[] {1, 2, 3}, DiscreteStateTokens.defaultDecoder().decodeTokens(updated));
        int[] exposed = DiscreteStateTokens.defaultDecoder().decodeTokens(updated);
        exposed[1] = 9;
        assertArrayEquals(new int[] {1, 2, 3}, DiscreteStateTokens.defaultDecoder().decodeTokens(updated));
    }

    @Test
    void decodesCommonNumericMetadataShapes() {
        assertArrayEquals(new int[] {1, 2}, decode(Map.of("tokens", new int[] {1, 2})));
        assertArrayEquals(new int[] {3, 4}, decode(Map.of("tokens", new long[] {3L, 4L})));
        assertArrayEquals(new int[] {5, 6}, decode(Map.of("tokens", new Number[] {5, 6L})));
        assertArrayEquals(new int[] {7, 8}, decode(Map.of("tokens", List.of(7, 8L))));
    }

    @Test
    void expectedLengthDecoderRejectsWrongShape() {
        RecursiveReasoningState state = state(Map.of("tokens", List.of(1, 2, 3)));

        assertArrayEquals(
                new int[] {1, 2, 3},
                DiscreteStateTokenDecoder.fromMetadata("tokens", 3).decodeTokens(state));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteStateTokenDecoder.fromMetadata("tokens", 2).decodeTokens(state));
    }

    @Test
    void rejectsMissingAndNonNumericMetadata() {
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteStateTokenDecoder.fromMetadata("missing").decodeTokens(state(Map.of())));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteStateTokenDecoder.fromMetadata("tokens").decodeTokens(
                        state(Map.of("tokens", List.of(1, "bad")))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteStateTokenDecoder.fromMetadata("tokens").decodeTokens(
                        state(Map.of("tokens", "bad"))));
    }

    @Test
    void attachesProjectionMetadataWithCustomKeys() {
        RecursiveReasoningState updated = DiscreteStateTokens.attachProjection(
                state(Map.of()),
                "tokens",
                new int[] {0, 1},
                "mode",
                "row-argmax",
                "projection",
                Map.of("source", "test"));

        assertArrayEquals(new int[] {0, 1}, DiscreteStateTokenDecoder.fromMetadata("tokens").decodeTokens(updated));
        assertEquals("row-argmax", updated.metadata().get("mode"));
        assertEquals("test", ((Map<?, ?>) updated.metadata().get("projection")).get("source"));
    }

    private static int[] decode(Map<String, Object> metadata) {
        return DiscreteStateTokenDecoder.fromMetadata("tokens").decodeTokens(state(metadata));
    }

    private static RecursiveReasoningState state(Map<String, Object> metadata) {
        return new RecursiveReasoningState("state", 0, 0, 0, new DummyTensor(), metadata);
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
