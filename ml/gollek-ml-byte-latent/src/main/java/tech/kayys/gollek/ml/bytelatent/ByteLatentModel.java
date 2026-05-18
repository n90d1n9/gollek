package tech.kayys.gollek.ml.bytelatent;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Minimal executable forward-pass contract for byte-latent language models.
 */
public interface ByteLatentModel {
    ByteLatentModelSpec spec();

    ByteLatentForwardPass forward(ByteSequenceWindowBatch batch);

    default String familyId() {
        return ByteLatentModelFamily.FAMILY_ID;
    }

    default int predictNextToken(int[] promptTokenIds) {
        int[] normalizedPrompt = normalizePrompt(promptTokenIds);
        ByteSequenceWindowBatch batch = buildPromptBatch(normalizedPrompt);
        ByteLatentForwardPass pass = forward(batch);
        return pass.predictedTokenIds()[0][normalizedPrompt.length - 1];
    }

    default ByteLatentGenerationResult generate(int[] promptTokenIds, int maxNewTokens) {
        int[] normalizedPrompt = normalizePrompt(promptTokenIds);
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0 but was " + maxNewTokens);
        }
        int[] generated = new int[maxNewTokens];
        int[] context = Arrays.copyOf(normalizedPrompt, normalizedPrompt.length);
        for (int i = 0; i < maxNewTokens; i++) {
            int nextToken = predictNextToken(context);
            generated[i] = nextToken;
            context = Arrays.copyOf(context, context.length + 1);
            context[context.length - 1] = nextToken;
        }
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("familyId", familyId());
        metadata.put("modelClass", getClass().getSimpleName());
        metadata.put("promptLength", normalizedPrompt.length);
        metadata.put("generatedLength", generated.length);
        metadata.put("maxSequenceLength", spec().maxByteSequenceLength());
        return new ByteLatentGenerationResult(normalizedPrompt, generated, metadata);
    }

    private int[] normalizePrompt(int[] promptTokenIds) {
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("promptTokenIds must not be null or empty");
        }
        int maxLength = spec().maxByteSequenceLength();
        int start = Math.max(0, promptTokenIds.length - maxLength);
        int[] normalized = Arrays.copyOfRange(promptTokenIds, start, promptTokenIds.length);
        for (int tokenId : normalized) {
            if (tokenId < 0) {
                throw new IllegalArgumentException("prompt token ids must be >= 0 but found " + tokenId);
            }
        }
        return normalized;
    }

    private ByteSequenceWindowBatch buildPromptBatch(int[] promptTokenIds) {
        int sequenceLength = promptTokenIds.length;
        int[][] inputIds = new int[][] {Arrays.copyOf(promptTokenIds, sequenceLength)};
        int[][] targetIds = new int[][] {Arrays.copyOf(promptTokenIds, sequenceLength)};
        boolean[][] attentionMask = new boolean[1][sequenceLength];
        Arrays.fill(attentionMask[0], true);
        return new ByteSequenceWindowBatch(inputIds, targetIds, attentionMask, 0, sequenceLength);
    }
}
