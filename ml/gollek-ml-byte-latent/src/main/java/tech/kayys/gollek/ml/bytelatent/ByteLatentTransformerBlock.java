package tech.kayys.gollek.ml.bytelatent;

import java.util.Objects;

/**
 * Architecture placeholder for byte-latent transformer blocks.
 */
public final class ByteLatentTransformerBlock {
    private final ByteLatentModelSpec spec;
    private final boolean causal;

    public ByteLatentTransformerBlock(ByteLatentModelSpec spec, boolean causal) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
        this.causal = causal;
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    public boolean causal() {
        return causal;
    }

    public String familyId() {
        return ByteLatentModelFamily.FAMILY_ID;
    }

    public int transformToken(int tokenId, int position) {
        int layerShift = spec.transformerLayerCount() + spec.attentionHeadCount();
        int positionShift = causal ? position : position * 2;
        return Math.floorMod(tokenId + layerShift + positionShift, spec.byteVocabularySize());
    }
}
