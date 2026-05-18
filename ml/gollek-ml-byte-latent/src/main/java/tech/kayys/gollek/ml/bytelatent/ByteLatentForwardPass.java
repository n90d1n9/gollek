package tech.kayys.gollek.ml.bytelatent;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight forward-pass result for byte-latent training and inference.
 */
public record ByteLatentForwardPass(
        int[][] predictedTokenIds,
        double meanLoss,
        int tokenCount,
        Map<String, Object> metadata) {

    public ByteLatentForwardPass {
        predictedTokenIds = deepCopy(predictedTokenIds, "predictedTokenIds");
        metadata = Map.copyOf(Objects.requireNonNull(metadata, "metadata must not be null"));
        if (Double.isNaN(meanLoss) || Double.isInfinite(meanLoss) || meanLoss < 0.0d) {
            throw new IllegalArgumentException("meanLoss must be finite and >= 0 but was " + meanLoss);
        }
        if (tokenCount < 0) {
            throw new IllegalArgumentException("tokenCount must be >= 0 but was " + tokenCount);
        }
    }

    @Override
    public int[][] predictedTokenIds() {
        return deepCopy(predictedTokenIds, "predictedTokenIds");
    }

    private static int[][] deepCopy(int[][] values, String label) {
        Objects.requireNonNull(values, label + " must not be null");
        int[][] copy = new int[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = Arrays.copyOf(
                    Objects.requireNonNull(values[i], label + "[" + i + "] must not be null"),
                    values[i].length);
        }
        return copy;
    }
}
