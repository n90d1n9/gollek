package tech.kayys.trellis2.models;

import tech.kayys.trellis2.modules.sparse.SparseTensor;
import tech.kayys.trellis2.modules.attention.MultiHeadAttention;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;

/**
 * FlowMatchingSampler — Java 25 port of the rectified flow sampling logic
 * from trellis2/models/sparse_structure_flow.py and structured_latent_flow.py
 *
 * <p>TRELLIS.2 uses <b>Rectified Flow Matching</b> (flow matching with ODE
 * trajectories from noise → data) to generate both:
 * <ul>
 *   <li>Sparse structure (stage 1): 3D occupancy grid in latent space</li>
 *   <li>Structured latents (stage 2/3): per-voxel shape and texture features</li>
 * </ul>
 *
 * <p>The sampler implements the DDIM-like ODE solver:
 * <pre>
 *   x_{t-1} = x_t + (v_θ(x_t, t, cond) + w * (v_θ(x_t, t, cond) - v_θ(x_t, t, ∅))) * Δt
 * </pre>
 * where {@code w} = guidance_strength and {@code v_θ} is the DiT velocity network.
 *
 * <p>Java 25 design:
 * <ul>
 *   <li>{@link SamplerConfig} is a record — immutable, validated in compact constructor.</li>
 *   <li>The velocity function {@code v_theta} is a {@link BiFunction} accepting
 *       (noisy_latent, timestep) → velocity, allowing any model implementation to plug in.</li>
 *   <li>All intermediate tensors use confined arenas freed at the end of each step.</li>
 *   <li>The {@code rescale_t} trick from the TRELLIS.2 paper is a direct port.</li>
 * </ul>
 *
 * Python original:
 * <pre>
 * def sample(self, noise, cond, neg_cond, sampler_params) -> SparseTensor
 *   # DDIM loop over timesteps with CFG
 * </pre>
 */
public final class FlowMatchingSampler {

    // ── Sampler configuration (immutable record) ──────────────────────────────

    /**
     * Configuration mirroring the Python {@code sampler_params} dict.
     * Compact constructor validates all parameters.
     */
    public record SamplerConfig(
            int   steps,
            float guidanceStrength,  // cfg_strength
            float guidanceRescale,   // guidance_rescale
            float rescaleT           // rescale_t: time-axis rescaling trick
    ) {
        public SamplerConfig {
            if (steps < 1)                             throw new IllegalArgumentException("steps >= 1");
            if (guidanceStrength < 0)                  throw new IllegalArgumentException("guidanceStrength >= 0");
            if (guidanceRescale < 0 || guidanceRescale > 1)
                throw new IllegalArgumentException("guidanceRescale in [0,1]");
            if (rescaleT < 1.0f)                       throw new IllegalArgumentException("rescaleT >= 1.0");
        }

        /** Defaults matching TRELLIS.2 paper for sparse-structure stage. */
        public static SamplerConfig sparseStructureDefaults() {
            return new SamplerConfig(12, 7.5f, 0.7f, 5.0f);
        }

        /** Defaults for shape SLAT stage. */
        public static SamplerConfig shapeSlatDefaults() {
            return new SamplerConfig(12, 7.5f, 0.5f, 3.0f);
        }

        /** Defaults for texture SLAT stage. */
        public static SamplerConfig textureSlatDefaults() {
            return new SamplerConfig(12, 1.0f, 0.0f, 3.0f);
        }
    }

    // ── VelocityModel functional interface ────────────────────────────────────

    /**
     * A velocity model v_θ(x_t, t, conditioning).
     *
     * <p>Takes noisy latent {@code x} and timestep {@code t ∈ [0,1]}
     * and returns the predicted velocity field.
     */
    @FunctionalInterface
    public interface VelocityModel {
        /**
         * @param x     noisy latent SparseTensor at time t
         * @param t     normalized timestep in [0, 1]
         * @param cond  conditioning MemorySegment (image embeddings)
         * @param arena arena for output allocation
         * @return velocity prediction (same shape as x.feats)
         */
        SparseTensor predict(SparseTensor x, float t,
                             MemorySegment cond, Arena arena);
    }

    // ── Core sampling algorithm ───────────────────────────────────────────────

    /**
     * Run the rectified flow ODE sampler.
     *
     * <p>Python equivalent:
     * <pre>
     * def sample(noise, cond, neg_cond, steps, cfg_strength, cfg_rescale, rescale_t)
     *   x = noise
     *   ts = linspace(1/steps, 1, steps).flip(0)
     *   for t in ts:
     *       cond_v  = model(x, t, cond)
     *       uncond_v = model(x, t, neg_cond)
     *       v = uncond_v + cfg_strength * (cond_v - uncond_v)  # CFG
     *       v = apply_rescale(v, cond_v, cfg_rescale)
     *       x = x - v * (1/steps)
     *   return x
     * </pre>
     *
     * @param noise     initial noise SparseTensor (Gaussian, same structure as output)
     * @param model     velocity network
     * @param cond      positive conditioning embeddings
     * @param negCond   null-conditioning embeddings (for CFG)
     * @param config    sampler hyperparameters
     * @param sharedArena arena owning noise and output (long-lived for this call)
     * @return denoised SparseTensor
     */
    public static SparseTensor sample(
            SparseTensor    noise,
            VelocityModel   model,
            MemorySegment   cond,
            MemorySegment   negCond,
            SamplerConfig   config,
            Arena           sharedArena
    ) {
        SparseTensor x = noise;
        float dt = 1.0f / config.steps();

        // Build timestep schedule: linspace(1.0, dt, steps) → ODE from t=1→0
        float[] ts = buildTimesteps(config.steps(), config.rescaleT());

        for (float t : ts) {
            // Each step gets its own confined arena — freed automatically after step
            try (Arena stepArena = Arena.ofConfined()) {

                // ── Conditional velocity prediction ───────────────────────────
                SparseTensor condV = model.predict(x, t, cond, stepArena);

                // ── Unconditional velocity (for CFG) ──────────────────────────
                SparseTensor uncondV = model.predict(x, t, negCond, stepArena);

                // ── Classifier-free guidance ─────────────────────────────────
                // v = uncond + cfg_strength * (cond - uncond)
                SparseTensor v = cfgCombine(condV, uncondV,
                        config.guidanceStrength(), config.guidanceRescale(),
                        stepArena);

                // ── Euler ODE step ────────────────────────────────────────────
                // x = x - v * dt   (rectified flow: d/dt x = v, so x(t-dt) = x(t) - v*dt)
                x = eulerStep(x, v, -dt, sharedArena);
            }
        }

        return x;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Python: ts = torch.linspace(1/steps, 1, steps).flip(0)
     * with the rescale_t trick: ts = ts * rescale_t - (rescale_t - 1)
     */
    private static float[] buildTimesteps(int steps, float rescaleT) {
        float[] ts = new float[steps];
        for (int i = 0; i < steps; i++) {
            // t from 1.0 down to 1/steps (reversed linspace)
            float t = 1.0f - (float) i / steps;
            // Apply rescale_t: maps [0,1] → [-(rescaleT-1), 1] then clamp to [0,1]
            t = t * rescaleT - (rescaleT - 1.0f);
            t = Math.max(0f, Math.min(1f, t));
            ts[i] = t;
        }
        return ts;
    }

    /**
     * Classifier-Free Guidance combination:
     * <pre>
     *   v_cfg = uncond + w * (cond - uncond)
     * </pre>
     * followed by optional rescaling (prevent std collapse).
     *
     * Python:
     * <pre>
     *   v = uncond_v + cfg_strength * (cond_v - uncond_v)
     *   if cfg_rescale > 0:
     *       std_cond = cond_v.feats.std()
     *       std_cfg  = v.feats.std()
     *       v = v * (std_cond / std_cfg) * cfg_rescale + v * (1 - cfg_rescale)
     * </pre>
     */
    private static SparseTensor cfgCombine(SparseTensor condV, SparseTensor uncondV,
                                            float w, float rescale, Arena arena) {
        int n = condV.n(), c = condV.c();
        MemorySegment out = arena.allocate((long) n * c * Float.BYTES);

        // v = uncond + w * (cond - uncond)
        for (int i = 0; i < n * c; i++) {
            float cv = condV.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            float uv = uncondV.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            out.setAtIndex(SparseTensor.FEAT_LAYOUT, i, uv + w * (cv - uv));
        }

        SparseTensor v = condV.replace(out);

        // Optional rescaling
        if (rescale > 0f) {
            v = applyRescale(v, condV, rescale, arena);
        }
        return v;
    }

    private static SparseTensor applyRescale(SparseTensor v, SparseTensor condV,
                                              float rescale, Arena arena) {
        float stdCond = std(condV);
        float stdV    = std(v);
        if (stdV < 1e-8f) return v;

        float scale = (stdCond / stdV) * rescale + (1.0f - rescale);
        MemorySegment out = arena.allocate(v.feats().byteSize());
        for (int i = 0; i < v.n() * v.c(); i++) {
            float val = v.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            out.setAtIndex(SparseTensor.FEAT_LAYOUT, i, val * scale);
        }
        return v.replace(out);
    }

    /** Euler integration step: x_new = x + velocity * dt */
    private static SparseTensor eulerStep(SparseTensor x, SparseTensor v,
                                           float dt, Arena arena) {
        MemorySegment out = arena.allocate(x.feats().byteSize());
        for (int i = 0; i < x.n() * x.c(); i++) {
            float xv    = x.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            float velV  = v.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
            out.setAtIndex(SparseTensor.FEAT_LAYOUT, i, xv + velV * dt);
        }
        // Euler step preserves sparsity structure (same coords)
        return x.replace(out);
    }

    /** Standard deviation of all feat values (for rescaling). */
    private static float std(SparseTensor t) {
        long total = (long) t.n() * t.c();
        double mean = 0;
        for (long i = 0; i < total; i++)
            mean += t.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i);
        mean /= total;
        double var = 0;
        for (long i = 0; i < total; i++) {
            double d = t.feats().getAtIndex(SparseTensor.FEAT_LAYOUT, i) - mean;
            var += d * d;
        }
        return (float) Math.sqrt(var / total);
    }

    /**
     * Generate Gaussian noise SparseTensor for initial condition.
     * Python: torch.randn_like(template_sparse_tensor)
     */
    public static SparseTensor gaussianNoise(SparseTensor template, long seed,
                                              Arena arena) {
        var rng = new java.util.Random(seed);
        MemorySegment noiseFeats = arena.allocate(template.feats().byteSize());
        for (long i = 0; i < (long) template.n() * template.c(); i++) {
            noiseFeats.setAtIndex(SparseTensor.FEAT_LAYOUT, i,
                    (float) rng.nextGaussian());
        }
        return template.replace(noiseFeats);
    }
}
