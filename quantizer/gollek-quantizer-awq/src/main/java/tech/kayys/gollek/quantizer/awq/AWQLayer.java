package tech.kayys.gollek.quantizer.awq;

import java.lang.foreign.MemorySegment;
import java.util.Optional;

/**
 * Represents a single AWQ-quantized linear layer, holding all
 * constituent tensors in off-heap FFM memory.
 *
 * AWQ weight layout (AutoAWQ GEMM format):
 * ──────────────────────────────────────────
 * qweight [inFeatures / packFactor, outFeatures] INT32
 * qzeros [inFeatures / groupSize, outFeatures / packFactor] INT32 (optional)
 * scales [inFeatures / groupSize, outFeatures] FP16
 *
 * Key difference from GPTQ:
 * AWQ groups along the INPUT dimension (rows of W^T, columns of W).
 * GPTQ groups along the OUTPUT dimension.
 *
 * This means in AWQ:
 * - scales[g, j] applies to input channel g*groupSize...(g+1)*groupSize − 1
 * for output feature j
 * - Dequant: w[i,j] = (q[i,j] − zero[g_i, j]) × scale[g_i, j]
 * where g_i = i / groupSize
 *
 * INT4 Packing (GEMM format, AutoAWQ):
 * ──────────────────────────────────────
 * Each INT32 holds 8 consecutive output-feature values for the same
 * packed input row. Bit layout (LSB first):
 *
 * bits [0-3] → output feature j+0
 * bits [4-7] → output feature j+1
 * ...
 * bits [28-31] → output feature j+7
 *
 * GEMV Packing:
 * ──────────────
 * Optimized for single-token generation. The 8 values packed per
 * INT32 are consecutive INPUT features (rows), not output features.
 */
public class AWQLayer {

    private final String name;
    private final AWQConfig config;

    /** Packed INT32 quantized weights — off-heap via FFM */
    private MemorySegment qweight;

    /** Packed INT32 zero-points — off-heap, optional */
    private MemorySegment qzeros;

    /** FP16 per-group scales — off-heap */
    private MemorySegment scales;

    /** Optional bias */
    private MemorySegment bias;

    // ── Dimensions ────────────────────────────────────────────────────────────

    private int inFeatures;
    private int outFeatures;

    // Raw shapes from safetensor metadata
    private long[] qweightShape;
    private long[] scalesShape;
    private long[] qzerosShape;

    public AWQLayer(String name, AWQConfig config) {
        this.name = name;
        this.config = config;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public String getName() {
        return name;
    }

    public AWQConfig getConfig() {
        return config;
    }

    public MemorySegment getQweight() {
        return qweight;
    }

    public void setQweight(MemorySegment s) {
        this.qweight = s;
    }

    public MemorySegment getQzeros() {
        return qzeros;
    }

    public void setQzeros(MemorySegment s) {
        this.qzeros = s;
    }

    public MemorySegment getScales() {
        return scales;
    }

    public void setScales(MemorySegment s) {
        this.scales = s;
    }

    public Optional<MemorySegment> getBias() {
        return Optional.ofNullable(bias);
    }

    public void setBias(MemorySegment s) {
        this.bias = s;
    }

    public int getInFeatures() {
        return inFeatures;
    }

    public void setInFeatures(int v) {
        this.inFeatures = v;
    }

    public int getOutFeatures() {
        return outFeatures;
    }

    public void setOutFeatures(int v) {
        this.outFeatures = v;
    }

    public long[] getQweightShape() {
        return qweightShape;
    }

    public void setQweightShape(long[] s) {
        this.qweightShape = s;
    }

    public long[] getScalesShape() {
        return scalesShape;
    }

    public void setScalesShape(long[] s) {
        this.scalesShape = s;
    }

    public long[] getQzerosShape() {
        return qzerosShape;
    }

    public void setQzerosShape(long[] s) {
        this.qzerosShape = s;
    }

    // ── Derived ───────────────────────────────────────────────────────────────

    /** Number of quantization groups (over the input dimension) */
    public int numGroups() {
        return (inFeatures + config.groupSize() - 1) / config.groupSize();
    }

    /** INT4 elements packed per INT32 word */
    public int packFactor() {
        return config.packFactor();
    }

    public boolean hasBias() {
        return bias != null;
    }

    public boolean hasZeros() {
        return qzeros != null;
    }

    /** All required tensors are loaded */
    public boolean isComplete() {
        return qweight != null && scales != null
                && (!config.hasZeros() || qzeros != null);
    }

    /** Total off-heap bytes used by this layer */
    public long estimatedBytes() {
        long t = 0;
        if (qweight != null)
            t += qweight.byteSize();
        if (qzeros != null)
            t += qzeros.byteSize();
        if (scales != null)
            t += scales.byteSize();
        if (bias != null)
            t += bias.byteSize();
        return t;
    }

    @Override
    public String toString() {
        return ("AWQLayer{name='%s', in=%d, out=%d, groups=%d, zeros=%b, complete=%b}")
                .formatted(name, inFeatures, outFeatures, numGroups(), hasZeros(), isComplete());
    }
}
