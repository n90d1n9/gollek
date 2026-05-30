package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class NQueensTokenProjectorTest {
    @Test
    void cellArgmaxProjectsPerCellTokenLogits() {
        int[] expectedTokens = NQueensSolution.ofColumns(1, 3, 0, 2).toTokens();

        NQueensTokenProjectionResult result = NQueensTokenProjector.cellArgmax(
                4,
                3,
                logitsForTokens(expectedTokens, 3));

        assertArrayEquals(expectedTokens, result.tokens());
        assertEquals(NQueensTokenProjectionMode.CELL_ARGMAX, result.mode());
        assertEquals(16, result.metadata().get("cellCount"));
        result.tokens()[0] = 9;
        assertArrayEquals(expectedTokens, result.tokens());
    }

    @Test
    void rowQueenArgmaxProjectsExactlyOneQueenPerRow() {
        float[] logits = lowLogits(4, 3);
        setQueenScore(logits, 4, 3, 0, 1, 4.0f);
        setQueenScore(logits, 4, 3, 1, 3, 4.0f);
        setQueenScore(logits, 4, 3, 2, 0, 4.0f);
        setQueenScore(logits, 4, 3, 3, 2, 4.0f);
        setQueenScore(logits, 4, 3, 0, 0, 3.0f);

        NQueensTokenProjectionResult result = NQueensTokenProjector.rowQueenArgmax(
                NQueensProblem.empty(4),
                3,
                logits);

        assertArrayEquals(NQueensSolution.ofColumns(1, 3, 0, 2).toTokens(), result.tokens());
        assertTrue(NQueensBenchmark.evaluateTokens(NQueensProblem.empty(4), result.tokens()).valid());
        assertEquals(NQueensTokenProjectionMode.ROW_QUEEN_ARGMAX, result.mode());
    }

    @Test
    void rowQueenArgmaxCanRespectFixedQueens() {
        NQueensProblem problem = NQueensProblem.ofFixedColumns(-1, 2, -1, -1);
        float[] logits = lowLogits(4, 3);
        setQueenScore(logits, 4, 3, 0, 1, 4.0f);
        setQueenScore(logits, 4, 3, 1, 0, 9.0f);
        setQueenScore(logits, 4, 3, 1, 2, -2.0f);
        setQueenScore(logits, 4, 3, 2, 0, 4.0f);
        setQueenScore(logits, 4, 3, 3, 2, 4.0f);

        NQueensTokenProjectionResult result = NQueensTokenProjector.rowQueenArgmaxRespectFixed(problem, 3, logits);

        assertArrayEquals(NQueensSolution.ofColumns(1, 2, 0, 2).toTokens(), result.tokens());
        assertEquals(NQueensTokenProjectionMode.ROW_QUEEN_ARGMAX_RESPECT_FIXED, result.mode());
        assertEquals(1, result.metadata().get("forcedFixedQueenCount"));
    }

    @Test
    void rejectsInvalidProjectionInputs() {
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenProjector.cellArgmax(0, 3, new float[] {}));
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenProjector.cellArgmax(2, 2, new float[8]));
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenProjector.cellArgmax(2, 3, new float[4]));

        float[] logits = new float[12];
        logits[3] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenProjector.cellArgmax(2, 3, logits));
    }

    @Test
    void projectingNextStateFactoryAttachesDefaultMetadataForRolloutEvaluation() {
        DummyTensor tensor = new DummyTensor();
        RecursiveReasoningContext context = new RecursiveReasoningContext(
                "nqueens",
                tensor,
                RecursiveReasoningConfig.builder().build());
        RecursiveReasoningState previous = new RecursiveReasoningState("root", 0, 0, 0, tensor, Map.of());
        GramTransitionSample sample = priorSample(tensor);
        int[] expectedTokens = NQueensSolution.ofColumns(1, 3, 0, 2).toTokens();
        GramNextStateFactory delegate = (ignoredPrevious, ignoredContext, transitionSample) ->
                new RecursiveReasoningState(
                        "next",
                        0,
                        1,
                        0,
                        transitionSample.latentState(),
                        Map.of("existing", "metadata"));
        NQueensTokenPredictionHead head = (proposed, ignoredContext, ignoredSample, problem) ->
                new NQueensTokenProjectionResult(
                        expectedTokens,
                        NQueensTokenProjectionMode.CELL_ARGMAX,
                        Map.of("source", "test-head"));
        NQueensTokenProjectingNextStateFactory factory = new NQueensTokenProjectingNextStateFactory(
                delegate,
                NQueensProblem.empty(4),
                head);

        RecursiveReasoningState next = factory.nextState(previous, context, sample);
        expectedTokens[0] = 9;

        assertEquals("metadata", next.metadata().get("existing"));
        assertEquals("CELL_ARGMAX", next.metadata().get(NQueensStateTokens.PROJECTION_MODE_METADATA_KEY));
        assertEquals("test-head", ((Map<?, ?>) next.metadata()
                .get(NQueensStateTokens.PROJECTION_METADATA_KEY)).get("source"));
        assertArrayEquals(
                NQueensSolution.ofColumns(1, 3, 0, 2).toTokens(),
                NQueensStateTokens.defaultDecoder().decodeTokens(next, NQueensProblem.empty(4)));
    }

    private static float[] logitsForTokens(int[] tokens, int vocabSize) {
        float[] logits = new float[tokens.length * vocabSize];
        Arrays.fill(logits, -4.0f);
        for (int cell = 0; cell < tokens.length; cell++) {
            logits[cell * vocabSize + tokens[cell]] = 4.0f;
        }
        return logits;
    }

    private static float[] lowLogits(int size, int vocabSize) {
        float[] logits = new float[size * size * vocabSize];
        Arrays.fill(logits, -4.0f);
        return logits;
    }

    private static void setQueenScore(
            float[] logits,
            int size,
            int vocabSize,
            int row,
            int column,
            float score) {
        logits[(row * size + column) * vocabSize + NQueensProblem.QUEEN_TOKEN] = score;
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
