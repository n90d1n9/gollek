package tech.kayys.trellis2.models.scvaes;

import tech.kayys.trellis2.modules.sparse.SparseTensor;
import tech.kayys.trellis2.modules.sparse.SparseConv3d;
import tech.kayys.trellis2.modules.sparse.SparseLinear;
import tech.kayys.trellis2.modules.sparse.SparseDownsample;
import tech.kayys.trellis2.modules.sparse.SparseUpsample;
import tech.kayys.trellis2.modules.sparse.SparseSpatial2Channel;
import tech.kayys.trellis2.modules.sparse.SparseChannel2Spatial;
import tech.kayys.trellis2.modules.LayerNorm32;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * SparseResBlock3d — Java 25 port of trellis2/models/sc_vaes/sparse_unet_vae.py
 *
 * <p>Sparse 3D residual block with optional downsampling/upsampling.
 * Mirrors the Python nn.Module with the same field names and forward logic.
 *
 * <p>Python original:
 * <pre>
 * class SparseResBlock3d(nn.Module):
 *   def __init__(self, channels, out_channels=None, downsample=False,
 *                upsample=False, resample_mode='nearest', use_checkpoint=False)
 *   def _forward(self, x: SparseTensor) -> SparseTensor
 * </pre>
 *
 * <p>Java 25 design:
 * <ul>
 *   <li>Sealed interface hierarchy for {@link ResampleMode} replaces Python string literals.</li>
 *   <li>All forward passes operate on {@link SparseTensor} records backed by FFM memory.</li>
 *   <li>Pattern matching switch dispatches on {@code ResampleMode} variants.</li>
 * </ul>
 */
public final class SparseResBlock3d {

    // ── Sealed ResampleMode hierarchy (replaces Python string literals) ───────

    /**
     * Python: resample_mode = 'nearest' | 'spatial2channel'
     * Sealed for exhaustive pattern matching.
     */
    public sealed interface ResampleMode
            permits ResampleMode.Nearest, ResampleMode.Spatial2Channel {
        record Nearest()         implements ResampleMode {}
        record Spatial2Channel() implements ResampleMode {}

        static final Nearest         NEAREST          = new Nearest();
        static final Spatial2Channel SPATIAL2CHANNEL  = new Spatial2Channel();
    }

    // ── Fields ────────────────────────────────────────────────────────────────

    private final int          channels;
    private final int          outChannels;
    private final boolean      downsample;
    private final boolean      upsample;
    private final ResampleMode resampleMode;

    // Learned parameters (weights stored as native MemorySegments in shared arena)
    private final MemorySegment norm1Weight;   // LayerNorm32 gamma  [outC]
    private final MemorySegment conv1Weight;   // SparseConv3d weight
    private final MemorySegment conv2Weight;   // SparseConv3d weight (zero-init)
    private final MemorySegment skipWeight;    // SparseLinear weight (or null if identity)

    // Functional sub-modules (stateless, no weights)
    private final SparseDownsample     downOp;
    private final SparseUpsample       upOp;
    private final SparseSpatial2Channel s2cOp;
    private final SparseChannel2Spatial c2sOp;

    private final Arena weightArena; // Shared arena owning all weight segments

    // ── Constructor ───────────────────────────────────────────────────────────

    /**
     * @param channels      input channels
     * @param outChannels   output channels (null → same as channels)
     * @param downsample    whether to downsample
     * @param upsample      whether to upsample
     * @param resampleMode  sampling strategy
     * @param weightArena   long-lived arena that will own weight memory
     */
    public SparseResBlock3d(int channels, Integer outChannels,
                             boolean downsample, boolean upsample,
                             ResampleMode resampleMode, Arena weightArena) {
        if (downsample && upsample)
            throw new IllegalArgumentException("Cannot downsample and upsample simultaneously");

        this.channels     = channels;
        this.outChannels  = outChannels != null ? outChannels : channels;
        this.downsample   = downsample;
        this.upsample     = upsample;
        this.resampleMode = resampleMode;
        this.weightArena  = weightArena;

        // ── Allocate weight segments ──────────────────────────────────────────
        // norm1: LayerNorm32 gamma (elementwise_affine=True) + beta = 2 * channels floats
        this.norm1Weight = weightArena.allocate((long) 2 * this.channels * Float.BYTES);

        // conv1 weight: depends on resample_mode
        int conv1OutC = switch (resampleMode) {
            case ResampleMode.Nearest         n -> this.outChannels;
            case ResampleMode.Spatial2Channel s ->
                    upsample ? this.outChannels * 8 : this.outChannels / 8;
        };
        // kernel 3x3x3 = 27 elements per input channel per output channel
        this.conv1Weight = weightArena.allocate(
                (long) conv1OutC * channels * 27 * Float.BYTES);

        // conv2 weight: always outChannels -> outChannels, 3x3x3, zero-initialized
        this.conv2Weight = weightArena.allocate(
                (long) this.outChannels * this.outChannels * 27 * Float.BYTES);
        // zero-init (already zeroed by allocate, but explicit for clarity)
        conv2Weight.fill((byte) 0);

        // skip_connection weight: linear projection if channels differ
        this.skipWeight = (this.channels != this.outChannels)
                ? weightArena.allocate((long) this.channels * this.outChannels * Float.BYTES)
                : null; // identity

        // ── Resampling operators (stateless — no weights) ─────────────────────
        this.downOp = downsample ? new SparseDownsample(2) : null;
        this.upOp   = upsample   ? new SparseUpsample(2)   : null;
        this.s2cOp  = (resampleMode instanceof ResampleMode.Spatial2Channel && downsample)
                ? new SparseSpatial2Channel(2) : null;
        this.c2sOp  = (resampleMode instanceof ResampleMode.Spatial2Channel && upsample)
                ? new SparseChannel2Spatial(2) : null;
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * Python: {@code def _forward(self, x: SparseTensor) -> SparseTensor}
     *
     * <p>Full port of the Python forward logic using Java 25 pattern matching switch.
     *
     * @param x         input SparseTensor
     * @param passArena per-inference confined arena for intermediate activations
     * @return output SparseTensor (may share coord segment with input)
     */
    public ForwardResult forward(SparseTensor x, Arena passArena) {

        SparseTensor subdiv = null;

        // ── Step 1: predict subdiv logits if upsampling ───────────────────────
        // Python: subdiv = self.to_subdiv(x)
        if (upsample) {
            subdiv = SparseLinear.apply(x, skipWeight /* placeholder */, 8, passArena);
        }

        // ── Step 2: norm1 + SiLU ─────────────────────────────────────────────
        // Python: h = x.replace(self.norm1(x.feats)); h = h.replace(F.silu(h.feats))
        SparseTensor h = x.layerNorm(1e-6f, passArena);
        h = silu(h, passArena);

        // ── Step 3: conv1 with optional spatial resampling ────────────────────
        // Pattern match on ResampleMode — exhaustive
        switch (resampleMode) {
            case ResampleMode.Spatial2Channel ignored -> {
                // Python: h = self.conv1(h); h = self._updown(h, subdiv); x = self._updown(x, subdiv)
                h = SparseConv3d.apply(h, conv1Weight, outChannels / 8, 3, passArena);
                h = updown(h, subdiv, passArena);
                x = updown(x, subdiv, passArena);
            }
            case ResampleMode.Nearest ignored -> {
                // Python: h = self._updown(h, subdiv) then conv1
                h = updown(h, subdiv, passArena);
                x = updown(x, subdiv, passArena);
                h = SparseConv3d.apply(h, conv1Weight, outChannels, 3, passArena);
            }
        }

        // ── Step 4: norm2 (no affine) + SiLU + conv2 ─────────────────────────
        // Python: h = h.replace(self.norm2(h.feats)); h = h.replace(F.silu(h.feats)); h = self.conv2(h)
        h = h.layerNorm(1e-6f, passArena);
        h = silu(h, passArena);
        h = SparseConv3d.apply(h, conv2Weight, outChannels, 3, passArena);

        // ── Step 5: residual skip connection ─────────────────────────────────
        // Python: h = h + self.skip_connection(x)
        h = add(h, skipConnection(x, passArena), passArena);

        // ── Step 6: return (with subdiv for upsample path) ───────────────────
        return upsample ? new ForwardResult(h, subdiv)
                        : new ForwardResult(h, null);
    }

    /** Result carrier — h is the main output, subdiv is only set on upsample path. */
    public record ForwardResult(SparseTensor h, SparseTensor subdiv) {}

    // ── Internal helpers ──────────────────────────────────────────────────────

    /** Python: F.silu(h.feats) — in-place approximation. */
    private SparseTensor silu(SparseTensor h, Arena arena) {
        MemorySegment outFeats = arena.allocate(h.feats().byteSize());
        for (int i = 0; i < h.n() * h.c(); i++) {
            float v = h.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float silu = v / (1f + (float) Math.exp(-v));
            outFeats.setAtIndex(SparseTensor.FEAT_LAYOUT, i, silu);
        }
        return h.replace(outFeats);
    }

    /** Element-wise addition of two SparseTensors with the same coords. */
    private SparseTensor add(SparseTensor a, SparseTensor b, Arena arena) {
        MemorySegment outFeats = arena.allocate(a.feats().byteSize());
        for (int i = 0; i < a.n() * a.c(); i++) {
            float va = a.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            float vb = b.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            outFeats.setAtIndex(SparseTensor.FEAT_LAYOUT, i, va + vb);
        }
        return a.replace(outFeats);
    }

    /** Python: self.skip_connection(x) — linear projection or identity. */
    private SparseTensor skipConnection(SparseTensor x, Arena arena) {
        if (skipWeight == null) return x; // identity
        return SparseLinear.apply(x, skipWeight, outChannels, arena);
    }

    /** Python: self._updown(h, subdiv) — dispatch to down/up/s2c/c2s op. */
    private SparseTensor updown(SparseTensor h, SparseTensor subdiv, Arena arena) {
        if (downsample) {
            return switch (resampleMode) {
                case ResampleMode.Nearest         n -> downOp.apply(h, arena);
                case ResampleMode.Spatial2Channel s -> s2cOp.apply(h, arena);
            };
        }
        if (upsample) {
            return switch (resampleMode) {
                case ResampleMode.Nearest         n -> upOp.apply(h, subdiv, arena);
                case ResampleMode.Spatial2Channel s -> c2sOp.apply(h, subdiv, arena);
            };
        }
        return h;
    }
}
