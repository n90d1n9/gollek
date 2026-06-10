package tech.kayys.gollek.ml.bytelatent;

import java.util.Objects;

/**
 * Deterministic output head that maps decoded byte-native values into predicted
 * token ids.
 */
public final class ReferenceByteLatentOutputHead {
    private final ByteLatentModelSpec spec;

    public ReferenceByteLatentOutputHead(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    public int projectTokenId(int decodedToken, int position) {
        return Math.floorMod(decodedToken + spec.attentionHeadCount() + position,
                spec.byteVocabularySize());
    }
}
