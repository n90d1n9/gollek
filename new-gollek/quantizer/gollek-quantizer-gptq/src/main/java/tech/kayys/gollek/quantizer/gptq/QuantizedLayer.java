package tech.kayys.gollek.quantizer.gptq;

import java.lang.foreign.MemorySegment;
import java.util.Optional;

/**
 * Represents a single GPTQ-quantized linear layer.
 *
 * GPTQ stores weights as:
 * qweight [out_features / pack_factor, in_features] - packed INT32
 * qzeros [in_features / group_size, out_features / pack_factor] - packed INT32
 * scales [in_features / group_size, out_features] - FP16
 * g_idx [in_features] - INT32 (optional, act-order)
 * bias [out_features] - FP16/FP32 (optional)
 *
 * Dequantized weight formula (per element w[i,j], group g):
 * w[i,j] = (qweight[i,j] - zeros[g,j]) * scales[g,j]
 */
public class QuantizedLayer {

    private final String name;
    private final GPTQConfig config;

    /** Packed INT32 quantized weights — off-heap via FFM */
    private MemorySegment qweight;

    /** Packed INT32 quantized zero-points — off-heap via FFM */
    private MemorySegment qzeros;

    /** FP16 scales — off-heap via FFM */
    private MemorySegment scales;

    /** INT32 group indices (optional, for act-order) — off-heap via FFM */
    private MemorySegment gIdx;

    /** Bias tensor (optional) — off-heap via FFM */
    private MemorySegment bias;

    // Dimensions
    private int inFeatures;
    private int outFeatures;

    // Shape metadata
    private long[] qweightShape;
    private long[] scalesShape;

    public QuantizedLayer(String name, GPTQConfig config) {
        this.name = name;
        this.config = config;
    }

    // ── Getters / Setters ─────────────────────────────────────────────────────

    public String getName() {
        return name;
    }

    public GPTQConfig getConfig() {
        return config;
    }

    public MemorySegment getQweight() {
        return qweight;
    }

    public void setQweight(MemorySegment qweight) {
        this.qweight = qweight;
    }

    public MemorySegment getQzeros() {
        return qzeros;
    }

    public void setQzeros(MemorySegment qzeros) {
        this.qzeros = qzeros;
    }

    public MemorySegment getScales() {
        return scales;
    }

    public void setScales(MemorySegment scales) {
        this.scales = scales;
    }

    public Optional<MemorySegment> getGIdx() {
        return Optional.ofNullable(gIdx);
    }

    public void setGIdx(MemorySegment gIdx) {
        this.gIdx = gIdx;
    }

    public Optional<MemorySegment> getBias() {
        return Optional.ofNullable(bias);
    }

    public void setBias(MemorySegment bias) {
        this.bias = bias;
    }

    public int getInFeatures() {
        return inFeatures;
    }

    public void setInFeatures(int inFeatures) {
        this.inFeatures = inFeatures;
    }

    public int getOutFeatures() {
        return outFeatures;
    }

    public void setOutFeatures(int outFeatures) {
        this.outFeatures = outFeatures;
    }

    public long[] getQweightShape() {
        return qweightShape;
    }

    public void setQweightShape(long[] qweightShape) {
        this.qweightShape = qweightShape;
    }

    public long[] getScalesShape() {
        return scalesShape;
    }

    public void setScalesShape(long[] scalesShape) {
        this.scalesShape = scalesShape;
    }

    // ── Derived Dimension Helpers ─────────────────────────────────────────────

    /** Number of quantization groups for this layer */
    public int numGroups() {
        return (inFeatures + config.groupSize() - 1) / config.groupSize();
    }

    /** Elements packed into each INT32 (e.g., 8 for 4-bit) */
    public int packFactor() {
        return config.elementsPerInt32();
    }

    /** Whether this layer has act-order group indices */
    public boolean hasGIdx() {
        return gIdx != null;
    }

    /** Whether this layer has a bias */
    public boolean hasBias() {
        return bias != null;
    }

    /** Validates that required tensors are loaded */
    public boolean isComplete() {
        return qweight != null && qzeros != null && scales != null;
    }

    /** Estimated memory usage in bytes (off-heap) */
    public long estimatedBytes() {
        long total = 0;
        if (qweight != null)
            total += qweight.byteSize();
        if (qzeros != null)
            total += qzeros.byteSize();
        if (scales != null)
            total += scales.byteSize();
        if (gIdx != null)
            total += gIdx.byteSize();
        if (bias != null)
            total += bias.byteSize();
        return total;
    }

    @Override
    public String toString() {
        return "QuantizedLayer{name='%s', in=%d, out=%d, bits=%d, groups=%d, complete=%b}"
                .formatted(name, inFeatures, outFeatures, config.bits(), numGroups(), isComplete());
    }
}
