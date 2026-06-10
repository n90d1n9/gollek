package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Deterministic causal mixer that adds lightweight latent-state interaction
 * before transformer-style token updates.
 */
public final class ReferenceByteLatentMixer {
    private final ByteLatentModelSpec spec;

    public ReferenceByteLatentMixer(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    public ByteLatentState mix(ByteLatentState state) {
        Objects.requireNonNull(state, "state must not be null");
        List<int[]> mixedCodes = state.latentCodes().stream()
                .map(this::mixSequence)
                .toList();
        Map<String, Object> metadata = new LinkedHashMap<>(state.metadata());
        metadata.put("mixerType", "reference-byte-latent");
        metadata.put("mixingMode", "causal-residual");
        metadata.put("mixShiftModulus", spec.attentionHeadCount() + 1);
        metadata.put("mixedSequenceCount", mixedCodes.size());
        return new ByteLatentState(mixedCodes, state.sourceByteCount(), metadata);
    }

    private int[] mixSequence(int[] latentCodes) {
        int[] mixed = new int[latentCodes.length];
        int modulus = spec.attentionHeadCount() + 1;
        for (int i = 0; i < latentCodes.length; i++) {
            int residual = i == 0 ? 0 : Math.floorMod(mixed[i - 1], modulus);
            mixed[i] = latentCodes[i] + residual;
        }
        return mixed;
    }
}
