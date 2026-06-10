package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Deterministic reference decoder that projects latent codes back into byte-native sequences.
 */
public final class ReferenceByteLatentDecoder implements ByteLatentDecoder {
    private final ByteLatentModelSpec spec;

    public ReferenceByteLatentDecoder(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    @Override
    public ByteSequenceBatch decode(ByteLatentState state) {
        Objects.requireNonNull(state, "state must not be null");
        List<byte[]> sequences = state.latentCodes().stream()
                .map(this::decodeSequence)
                .toList();
        return new ByteSequenceBatch(sequences);
    }

    public Map<String, Object> decodeMetadata(ByteLatentState state) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("decoderType", "reference-byte-latent");
        metadata.put("familyId", ByteLatentModelFamily.FAMILY_ID);
        metadata.put("latentDimension", spec.latentDimension());
        metadata.put("sourceByteCount", state.sourceByteCount());
        metadata.put("sequenceCount", state.latentCodes().size());
        return metadata;
    }

    private byte[] decodeSequence(int[] latentCodes) {
        byte[] decoded = new byte[latentCodes.length];
        for (int i = 0; i < latentCodes.length; i++) {
            int value = Math.floorMod(latentCodes[i] - spec.latentDimension(), spec.byteVocabularySize());
            decoded[i] = (byte) value;
        }
        return decoded;
    }
}
