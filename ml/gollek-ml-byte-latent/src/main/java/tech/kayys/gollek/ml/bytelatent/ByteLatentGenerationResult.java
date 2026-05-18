package tech.kayys.gollek.ml.bytelatent;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal byte-native generation result for prompt continuation.
 */
public record ByteLatentGenerationResult(
        int[] promptTokenIds,
        int[] generatedTokenIds,
        Map<String, Object> metadata) {

    public ByteLatentGenerationResult {
        promptTokenIds = copy(promptTokenIds, "promptTokenIds");
        generatedTokenIds = copy(generatedTokenIds, "generatedTokenIds");
        metadata = Map.copyOf(Objects.requireNonNull(metadata, "metadata must not be null"));
    }

    @Override
    public int[] promptTokenIds() {
        return copy(promptTokenIds, "promptTokenIds");
    }

    @Override
    public int[] generatedTokenIds() {
        return copy(generatedTokenIds, "generatedTokenIds");
    }

    public int[] combinedTokenIds() {
        int[] combined = Arrays.copyOf(promptTokenIds, promptTokenIds.length + generatedTokenIds.length);
        System.arraycopy(generatedTokenIds, 0, combined, promptTokenIds.length, generatedTokenIds.length);
        return combined;
    }

    public String generatedText() {
        return new String(asBytes(generatedTokenIds), StandardCharsets.UTF_8);
    }

    public String combinedText() {
        return new String(asBytes(combinedTokenIds()), StandardCharsets.UTF_8);
    }

    private static int[] copy(int[] values, String label) {
        Objects.requireNonNull(values, label + " must not be null");
        return Arrays.copyOf(values, values.length);
    }

    private static byte[] asBytes(int[] tokenIds) {
        byte[] bytes = new byte[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            bytes[i] = (byte) Math.floorMod(tokenIds[i], 256);
        }
        return bytes;
    }
}
