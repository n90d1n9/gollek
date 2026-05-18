package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ReferenceByteLatentModelTest {

    @Test
    void computesDeterministicForwardPass() {
        ReferenceByteLatentModel model = new ReferenceByteLatentModel(
                new ByteLatentModelSpec(256, 64, 2, 4, 16));
        ByteSequenceWindowBatch batch = new ByteSequenceWindowBatch(
                new int[][] {{10, 20, 0}},
                new int[][] {{11, 21, 0}},
                new boolean[][] {{true, true, false}},
                0,
                3);

        ByteLatentForwardPass pass = model.forward(batch);

        assertArrayEquals(new int[] {26, 38, 20}, pass.predictedTokenIds()[0]);
        assertEquals(2, pass.tokenCount());
        assertEquals(((15.0d / 255.0d) + (17.0d / 255.0d)) / 2.0d, pass.meanLoss());
        assertEquals("reference-byte-latent", pass.metadata().get("modelType"));
        assertEquals(ByteLatentModelFamily.FAMILY_ID, pass.metadata().get("familyId"));
    }

    @Test
    void predictsNextTokenFromPrompt() {
        ReferenceByteLatentModel model = new ReferenceByteLatentModel(
                new ByteLatentModelSpec(256, 64, 2, 4, 16));

        int nextToken = model.predictNextToken(new int[] {10, 20});

        assertEquals(38, nextToken);
    }

    @Test
    void generatesDeterministicContinuation() {
        ReferenceByteLatentModel model = new ReferenceByteLatentModel(
                new ByteLatentModelSpec(256, 64, 2, 4, 16));

        ByteLatentGenerationResult result = model.generate(new int[] {10, 20}, 3);

        assertArrayEquals(new int[] {10, 20}, result.promptTokenIds());
        assertArrayEquals(new int[] {38, 58, 80}, result.generatedTokenIds());
        assertArrayEquals(new int[] {10, 20, 38, 58, 80}, result.combinedTokenIds());
        assertEquals(2, result.metadata().get("promptLength"));
        assertEquals(3, result.metadata().get("generatedLength"));
        assertEquals("ReferenceByteLatentModel", result.metadata().get("modelClass"));
    }

    @Test
    void truncatesPromptToModelMaxSequenceLength() {
        ReferenceByteLatentModel model = new ReferenceByteLatentModel(
                new ByteLatentModelSpec(256, 64, 2, 4, 3));

        ByteLatentGenerationResult result = model.generate(new int[] {1, 2, 3, 4, 5}, 1);

        assertArrayEquals(new int[] {3, 4, 5}, result.promptTokenIds());
        assertArrayEquals(new int[] {25}, result.generatedTokenIds());
    }
}
