package tech.kayys.gollek.ml.bytelatent;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Encoded latent representation derived from byte sequences.
 */
public record ByteLatentState(
        List<int[]> latentCodes,
        int sourceByteCount,
        Map<String, Object> metadata) {

    public ByteLatentState {
        Objects.requireNonNull(latentCodes, "latentCodes must not be null");
        Objects.requireNonNull(metadata, "metadata must not be null");
        if (sourceByteCount < 0) {
            throw new IllegalArgumentException("sourceByteCount must be >= 0 but was " + sourceByteCount);
        }
        latentCodes = latentCodes.stream()
                .map(code -> {
                    Objects.requireNonNull(code, "latent code must not be null");
                    return code.clone();
                })
                .toList();
        metadata = Map.copyOf(metadata);
    }

    @Override
    public List<int[]> latentCodes() {
        return latentCodes.stream()
                .map(int[]::clone)
                .toList();
    }
}
