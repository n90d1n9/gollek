package tech.kayys.gollek.quantizer.autoround;

import java.lang.foreign.MemorySegment;
import java.util.Optional;

/**
 * Represents a single AutoRound-quantized linear layer with all constituent
 * tensors stored in off-heap FFM {@link MemorySegment}s.
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ AutoRound Tensor Layout (native format) │
 * ├──────────────────┬──────────────┬──────────────────────────────────┤
 * │ Tensor │ DType │ Shape │
 * ├──────────────────┼──────────────┼──────────────────────────────────┤
 * │ .weight / qweight│ INT32 │ [outF/pack, inF] │
 * │ .scale / scales │ FP32 or FP16 │ [outF, inF/group] │
 * │ .zp / qzeros │ INT32 unpacked│ [outF, inF/group] (optional) │
 * │ .bias │ FP32 or FP16 │ [outF] (optional) │
 * └──────────────────┴──────────────┴──────────────────────────────────┘
 *
 * Key differences from GPTQ and AWQ:
 *
 * vs GPTQ:
 * - AutoRound scale/zp are shaped [outF, inF/group] — groups over INPUT
 * - AutoRound zero-points are plain INT32, NOT packed into INT32 words
 * - AutoRound scales may be FP32 (not FP16)
 *
 * vs AWQ:
 * - qweight packing is identical to GPTQ ([outF/pack, inF])
 * - scales shaped [outF, inF/group] — same as GPTQ/AWQ in effect
 * - No per-channel activation scaling (AWQ's diag(s) trick)
 *
 * Dequantization formula:
 * w_fp32[i,j] = (q[i,j] − zp[j, g_i]) × scale[j, g_i]
 * where g_i = i / groupSize (group over input dimension)
 *
 * For GPTQ-compat format:
 * Zero-points ARE packed (same INT4 packing as qweight/qzeros in GPTQ).
 * The loader normalises these to unpacked INT32 before storing here.
 */
public class AutoRoundLayer {

    private final String name;
    private final AutoRoundConfig config;

    // ── Off-heap tensor segments (FFM MemorySegment) ──────────────────────────

    /** Packed INT32 quantized weights [outF/pack, inF] */
    private MemorySegment weight;

    /**
     * Per-group scale factors.
     * Native: FP32 [outF, inF/group]
     * GPTQ-compat: FP16 [outF, inF/group]
     * Both are normalised to FP32 during loading.
     */
    private MemorySegment scale;

    /**
     * Per-group zero-points.
     * Native: INT32 unpacked [outF, inF/group] — one integer per group
     * GPTQ-compat: originally packed INT32, unpacked to INT32 during loading
     * Absent for symmetric quantization.
     */
    private MemorySegment zp;

    /** Optional bias [outF] */
    private MemorySegment bias;

    // ── Dimension metadata ─────────────────────────────────────────────────────

    private int inFeatures;
    private int outFeatures;

    private long[] weightShape;
    private long[] scaleShape;
    private long[] zpShape;

    /** Whether this layer's scale was originally FP16 (GPTQ-compat export) */
    private boolean scaleWasFp16;

    public AutoRoundLayer(String name, AutoRoundConfig config) {
        this.name = name;
        this.config = config;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public String getName() {
        return name;
    }

    public AutoRoundConfig getConfig() {
        return config;
    }

    public MemorySegment getWeight() {
        return weight;
    }

    public void setWeight(MemorySegment s) {
        this.weight = s;
    }

    public MemorySegment getScale() {
        return scale;
    }

    public void setScale(MemorySegment s) {
        this.scale = s;
    }

    public Optional<MemorySegment> getZp() {
        return Optional.ofNullable(zp);
    }

    public void setZp(MemorySegment s) {
        this.zp = s;
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

    public long[] getWeightShape() {
        return weightShape;
    }

    public void setWeightShape(long[] s) {
        this.weightShape = s;
    }

    public long[] getScaleShape() {
        return scaleShape;
    }

    public void setScaleShape(long[] s) {
        this.scaleShape = s;
    }

    public long[] getZpShape() {
        return zpShape;
    }

    public void setZpShape(long[] s) {
        this.zpShape = s;
    }

    public boolean isScaleWasFp16() {
        return scaleWasFp16;
    }

    public void setScaleWasFp16(boolean b) {
        this.scaleWasFp16 = b;
    }

    // ── Derived Properties ────────────────────────────────────────────────────

    /** Number of quantization groups (over input dimension) */
    public int numGroups() {
        return (inFeatures + config.groupSize() - 1) / config.groupSize();
    }

    /** INT(bits) elements packed per INT32 word */
    public int packFactor() {
        return config.packFactor();
    }

    /** Whether zero-point tensor is present */
    public boolean hasZp() {
        return zp != null;
    }

    /** Whether bias tensor is present */
    public boolean hasBias() {
        return bias != null;
    }

    /**
     * Layer is complete if weight and scale are loaded, and zero-point
     * is present when required by config.
     */
    public boolean isComplete() {
        return weight != null && scale != null
                && (!config.hasZeroPoint() || zp != null);
    }

    /** Total off-heap bytes for this layer */
    public long estimatedBytes() {
        long t = 0;
        if (weight != null)
            t += weight.byteSize();
        if (scale != null)
            t += scale.byteSize();
        if (zp != null)
            t += zp.byteSize();
        if (bias != null)
            t += bias.byteSize();
        return t;
    }

    @Override
    public String toString() {
        return ("AutoRoundLayer{name='%s', in=%d, out=%d, groups=%d, " +
                "hasZP=%b, bias=%b, complete=%b}")
                .formatted(name, inFeatures, outFeatures, numGroups(),
                        hasZp(), hasBias(), isComplete());
    }
}
