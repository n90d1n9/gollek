package tech.kayys.gollek.ml.reasoning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Shared helpers for projecting logits shaped [itemCount, vocabSize].
 */
public final class DiscreteTokenProjector {
    private DiscreteTokenProjector() {
    }

    public static DiscreteTokenProjectionResult argmax(
            int itemCount,
            int vocabSize,
            float[] logits) {
        float[] input = requireFiniteLogits(itemCount, vocabSize, logits);
        int[] tokens = new int[itemCount];
        for (int item = 0; item < itemCount; item++) {
            tokens[item] = argmaxTokenAt(input, item, vocabSize);
        }
        return new DiscreteTokenProjectionResult(
                tokens,
                itemCount,
                vocabSize,
                metadata(itemCount, vocabSize, "argmax"));
    }

    public static int argmaxTokenAt(
            float[] logits,
            int itemIndex,
            int vocabSize) {
        Objects.requireNonNull(logits, "logits must not be null");
        requireVocabSize(vocabSize);
        int totalItems = requireLogitMatrix(logits, vocabSize);
        requireItemIndex(itemIndex, totalItems);
        int offset = Math.multiplyExact(itemIndex, vocabSize);
        int bestToken = 0;
        float bestScore = logits[offset];
        for (int token = 1; token < vocabSize; token++) {
            float score = logits[offset + token];
            if (score > bestScore) {
                bestScore = score;
                bestToken = token;
            }
        }
        return bestToken;
    }

    public static int bestItemForToken(
            float[] logits,
            int startItem,
            int itemCount,
            int vocabSize,
            int tokenIndex) {
        Objects.requireNonNull(logits, "logits must not be null");
        requireVocabSize(vocabSize);
        if (tokenIndex < 0 || tokenIndex >= vocabSize) {
            throw new IllegalArgumentException(
                    "tokenIndex must be in [0, " + vocabSize + ") but was " + tokenIndex);
        }
        if (itemCount < 1) {
            throw new IllegalArgumentException("itemCount must be >= 1 but was " + itemCount);
        }
        int totalItems = requireLogitMatrix(logits, vocabSize);
        requireItemIndex(startItem, totalItems);
        requireItemIndex(startItem + itemCount - 1, totalItems);

        int bestItem = startItem;
        float bestScore = logits[startItem * vocabSize + tokenIndex];
        for (int item = startItem + 1; item < startItem + itemCount; item++) {
            float score = logits[item * vocabSize + tokenIndex];
            if (score > bestScore) {
                bestScore = score;
                bestItem = item;
            }
        }
        return bestItem;
    }

    public static float[] requireFiniteLogits(
            int itemCount,
            int vocabSize,
            float[] logits) {
        if (itemCount < 1) {
            throw new IllegalArgumentException("itemCount must be >= 1 but was " + itemCount);
        }
        requireVocabSize(vocabSize);
        Objects.requireNonNull(logits, "logits must not be null");
        int expected = Math.multiplyExact(itemCount, vocabSize);
        if (logits.length != expected) {
            throw new IllegalArgumentException(
                    "logits length must be " + expected + " for itemCount " + itemCount
                            + " and vocabSize " + vocabSize + " but was " + logits.length);
        }
        for (int i = 0; i < logits.length; i++) {
            if (!Float.isFinite(logits[i])) {
                throw new IllegalArgumentException("logits must be finite but index " + i + " was " + logits[i]);
            }
        }
        return logits.clone();
    }

    private static void requireVocabSize(int vocabSize) {
        if (vocabSize < 1) {
            throw new IllegalArgumentException("vocabSize must be >= 1 but was " + vocabSize);
        }
    }

    private static int requireLogitMatrix(float[] logits, int vocabSize) {
        if (logits.length == 0 || logits.length % vocabSize != 0) {
            throw new IllegalArgumentException(
                    "logits length must be a non-empty multiple of vocabSize but was "
                            + logits.length + " for vocabSize " + vocabSize);
        }
        return logits.length / vocabSize;
    }

    private static void requireItemIndex(int itemIndex, int itemCount) {
        if (itemIndex < 0 || itemIndex >= itemCount) {
            throw new IllegalArgumentException(
                    "itemIndex must be in [0, " + itemCount + ") but was " + itemIndex);
        }
    }

    private static Map<String, Object> metadata(
            int itemCount,
            int vocabSize,
            String projection) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("projection", projection);
        metadata.put("itemCount", itemCount);
        metadata.put("vocabSize", vocabSize);
        return metadata;
    }
}
