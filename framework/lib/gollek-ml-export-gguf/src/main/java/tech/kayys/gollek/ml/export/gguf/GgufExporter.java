package tech.kayys.gollek.ml.export.gguf;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.gguf.GgufMetaValue;
import tech.kayys.gollek.ml.gguf.GgufWriter;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Exports a Gollek {@link NNModule} to GGUF format for llama.cpp / Ollama deployment.
 *
 * <p>Quantization is applied tensor-by-tensor before writing:
 * <ul>
 *   <li>{@code NONE / FP16} — stored as F32 (FP16 is a runtime concern in llama.cpp)</li>
 *   <li>{@code INT8}  — symmetric per-tensor: {@code scale = max(|x|) / 127}</li>
 *   <li>{@code INT4 / NF4} — block-wise Q4_0 (32-element blocks): {@code scale = max(|x|) / 7}</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * GgufExporter.fromModel(model)
 *             .quantization(Quantization.INT8)
 *             .export(Path.of("model.gguf"));
 * }</pre>
 */
public final class GgufExporter {

    /** GGUF file_type values matching llama.cpp conventions. */
    public enum Quantization {
        NONE(0), FP16(1), INT8(8), INT4(2), NF4(20);
        public final long fileType;
        Quantization(long fileType) { this.fileType = fileType; }
    }

    private final NNModule model;
    private final Map<String, Object> metadata;
    private Quantization quantization = Quantization.NONE;

    private GgufExporter(NNModule model, Map<String, Object> metadata) {
        this.model    = model;
        this.metadata = metadata != null ? metadata : Map.of();
    }

    public static GgufExporter fromModel(NNModule model) {
        return new GgufExporter(model, Map.of());
    }

    public static GgufExporter fromModel(NNModule model, Map<String, Object> metadata) {
        return new GgufExporter(model, metadata);
    }

    public GgufExporter quantization(Quantization q) {
        this.quantization = q;
        return this;
    }

    public void export(Path outputPath) throws IOException {
        Map<String, GradTensor> stateDict = model.stateDict();

        Map<String, GradTensor> quantized = new LinkedHashMap<>();
        for (var e : stateDict.entrySet()) {
            quantized.put(e.getKey(), quantize(e.getValue()));
        }

        Map<String, GgufMetaValue> ggufMeta = new LinkedHashMap<>();
        ggufMeta.put("general.architecture",
                GgufMetaValue.ofString(metadata.getOrDefault("architecture", "gollek").toString()));
        ggufMeta.put("general.quantization_version", GgufMetaValue.ofUint32(2));
        ggufMeta.put("general.file_type",            GgufMetaValue.ofUint32(quantization.fileType));
        ggufMeta.put("general.parameter_count",      GgufMetaValue.ofUint32(model.parameterCount()));
        for (var e : metadata.entrySet()) {
            if (!e.getKey().startsWith("general."))
                ggufMeta.put(e.getKey(), GgufMetaValue.ofString(e.getValue().toString()));
        }

        GgufWriter.save(outputPath, quantized, ggufMeta);
    }

    // ── Quantization ──────────────────────────────────────────────────────────

    private GradTensor quantize(GradTensor t) {
        return switch (quantization) {
            case NONE, FP16 -> t;
            case INT8       -> quantizeInt8(t);
            case INT4, NF4  -> quantizeInt4(t);
        };
    }

    /** Symmetric per-tensor INT8. */
    private static GradTensor quantizeInt8(GradTensor t) {
        float[] src = t.data();
        float maxAbs = 0f;
        for (float v : src) maxAbs = Math.max(maxAbs, Math.abs(v));
        if (maxAbs == 0f) return t;
        float scale = maxAbs / 127f;
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++)
            out[i] = Math.max(-127, Math.min(127, Math.round(src[i] / scale)));
        return GradTensor.of(out, t.shape());
    }

    /** Block-wise Q4_0 (32-element blocks). */
    private static GradTensor quantizeInt4(GradTensor t) {
        float[] src = t.data();
        float[] out = new float[src.length];
        for (int b = 0; b < src.length; b += 32) {
            int end = Math.min(b + 32, src.length);
            float maxAbs = 0f;
            for (int i = b; i < end; i++) maxAbs = Math.max(maxAbs, Math.abs(src[i]));
            float scale = maxAbs == 0f ? 1f : maxAbs / 7f;
            for (int i = b; i < end; i++)
                out[i] = Math.max(-8, Math.min(7, Math.round(src[i] / scale)));
        }
        return GradTensor.of(out, t.shape());
    }
}
