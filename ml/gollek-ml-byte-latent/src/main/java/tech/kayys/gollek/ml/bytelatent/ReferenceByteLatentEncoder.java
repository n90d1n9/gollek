package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Deterministic reference encoder that lifts byte-native sequences into simple latent codes.
 */
public final class ReferenceByteLatentEncoder implements ByteLatentEncoder {
    private final ByteLatentModelSpec spec;

    public ReferenceByteLatentEncoder(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    @Override
    public ByteLatentState encode(ByteSequenceBatch batch) {
        Objects.requireNonNull(batch, "batch must not be null");
        List<int[]> latentCodes = batch.sequences().stream()
                .map(this::encodeSequence)
                .toList();
        int sourceByteCount = batch.sequences().stream()
                .mapToInt(sequence -> sequence.length)
                .sum();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("encoderType", "reference-byte-latent");
        metadata.put("familyId", ByteLatentModelFamily.FAMILY_ID);
        metadata.put("latentDimension", spec.latentDimension());
        metadata.put("batchSize", batch.batchSize());
        metadata.put("sourceByteCount", sourceByteCount);
        return new ByteLatentState(latentCodes, sourceByteCount, metadata);
    }

    private int[] encodeSequence(byte[] sequence) {
        int[] latent = new int[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            latent[i] = Byte.toUnsignedInt(sequence[i]) + spec.latentDimension();
        }
        return latent;
    }
}
