package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Deterministic scalar projection that adds a lightweight latent head before
 * transformer-style token updates.
 */
public final class ReferenceByteLatentProjection {
    private final ByteLatentModelSpec spec;

    public ReferenceByteLatentProjection(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
    }

    public ByteLatentModelSpec spec() {
        return spec;
    }

    public ByteLatentState project(ByteLatentState state) {
        Objects.requireNonNull(state, "state must not be null");
        List<int[]> projectedCodes = state.latentCodes().stream()
                .map(this::projectSequence)
                .toList();
        Map<String, Object> metadata = new LinkedHashMap<>(state.metadata());
        metadata.put("projectionType", "reference-byte-latent");
        metadata.put("projectionMode", "position-head");
        metadata.put("projectionStride", spec.transformerLayerCount());
        metadata.put("projectedSequenceCount", projectedCodes.size());
        return new ByteLatentState(projectedCodes, state.sourceByteCount(), metadata);
    }

    private int[] projectSequence(int[] latentCodes) {
        int[] projected = new int[latentCodes.length];
        for (int i = 0; i < latentCodes.length; i++) {
            projected[i] = latentCodes[i] + (spec.transformerLayerCount() * (i + 1));
        }
        return projected;
    }
}
