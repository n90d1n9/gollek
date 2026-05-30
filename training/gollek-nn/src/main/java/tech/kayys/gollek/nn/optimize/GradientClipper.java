package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.List;

/**
 * Gradient clipping utilities to prevent exploding gradients during training.
 *
 * <p>
 * Two strategies are provided:
 * <ul>
 * <li>{@link #clipByNorm} — rescales all gradients so the global L2 norm ≤
 * {@code maxNorm}</li>
 * <li>{@link #clipByValue} — clamps each gradient element to
 * {@code [minVal, maxVal]}</li>
 * </ul>
 *
 * <p>
 * Clip-by-norm is the standard approach for RNNs and Transformers.
 * Call it <em>before</em> {@code optimizer.step()}.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * loss.backward();
 * float norm = GradientClipper.clipByNorm(model.parameters(), 1.0f);
 * optimizer.step();
 * }</pre>
 */
public final class GradientClipper {

    private GradientClipper() {
    }

    /**
     * Detailed clipping telemetry for trainer logs, callbacks, and checkpoint
     * diagnostics.
     *
     * @param preClipNorm  global L2 gradient norm before clipping
     * @param postClipNorm global L2 gradient norm after clipping
     * @param maxNorm      configured clipping threshold
     * @param scale        multiplier applied to gradients
     * @param clipped      whether gradients were actually rescaled
     */
    public record ClipResult(float preClipNorm, float postClipNorm, float maxNorm, float scale, boolean clipped) {
    }

    /**
     * Computes the current global L2 gradient norm without mutating gradients.
     *
     * @param params list of parameters whose gradients to inspect
     * @return global L2 norm across all non-null gradient buffers
     */
    public static float totalNorm(List<Parameter> params) {
        OptimizerValidation.requireStepInputs(params, "GradientClipper");
        return (float) Math.sqrt(totalSquaredNorm(params));
    }

    /**
     * Clips gradients by global L2 norm.
     *
     * <p>
     * Computes the total gradient norm across all parameters:
     * {@code totalNorm = sqrt(Σ ||g||²)}. If {@code totalNorm > maxNorm},
     * scales every gradient by {@code maxNorm / totalNorm}.
     *
     * <p>
     * Uses {@link VectorOps} for SIMD-accelerated inner-product computation.
     *
     * @param params  list of parameters whose gradients to clip
     * @param maxNorm maximum allowed gradient norm (must be positive)
     * @return the pre-clipping total gradient norm
     */
    public static float clipByNorm(List<Parameter> params, float maxNorm) {
        return clipByNormDetailed(params, maxNorm).preClipNorm();
    }

    /**
     * Clips gradients by global L2 norm and returns detailed telemetry.
     *
     * <p>
     * The existing {@link #clipByNorm(List, float)} API remains intentionally
     * small. This richer variant is meant for trainers that need to log or react
     * to exploding gradients.
     *
     * @param params  list of parameters whose gradients to clip
     * @param maxNorm maximum allowed gradient norm (must be positive)
     * @return clipping telemetry, including pre/post norm and applied scale
     */
    public static ClipResult clipByNormDetailed(List<Parameter> params, float maxNorm) {
        OptimizerValidation.finitePositive(maxNorm, "maxNorm");
        OptimizerValidation.requireStepInputs(params, "GradientClipper");
        float preClipNorm = (float) Math.sqrt(totalSquaredNorm(params));
        float scale = 1.0f;
        boolean clipped = preClipNorm > maxNorm;

        if (clipped) {
            scale = maxNorm / (preClipNorm + 1e-6f);
            for (Parameter p : params) {
                if (p.data().grad() == null)
                    continue;
                float[] g = p.data().grad().data();
                VectorOps.mulScalar(g, scale, g);
            }
        }

        float postClipNorm = clipped ? (float) Math.sqrt(totalSquaredNorm(params)) : preClipNorm;
        return new ClipResult(preClipNorm, postClipNorm, maxNorm, scale, clipped);
    }

    /**
     * Clips each gradient element to the range {@code [minVal, maxVal]}.
     *
     * <p>
     * This is a simpler but less principled approach than norm clipping.
     * Useful when gradient distributions are known to be bounded.
     *
     * @param params list of parameters whose gradients to clip
     * @param minVal minimum allowed gradient value
     * @param maxVal maximum allowed gradient value
     */
    public static void clipByValue(List<Parameter> params, float minVal, float maxVal) {
        OptimizerValidation.requireParameters(params);
        if (!Float.isFinite(minVal) || !Float.isFinite(maxVal) || minVal > maxVal) {
            throw new IllegalArgumentException(
                    "clip range must be finite and ordered, got [" + minVal + ", " + maxVal + "]");
        }
        OptimizerValidation.requireStepInputs(params, "GradientClipper");
        for (Parameter p : params) {
            if (p.data().grad() == null)
                continue;
            float[] g = p.data().grad().data();
            for (int i = 0; i < g.length; i++) {
                if (g[i] < minVal)
                    g[i] = minVal;
                else if (g[i] > maxVal)
                    g[i] = maxVal;
            }
        }
    }

    private static float totalSquaredNorm(List<Parameter> params) {
        float totalSq = 0f;
        for (Parameter p : params) {
            if (p.data().grad() == null)
                continue;
            float[] g = p.data().grad().data();
            for (float v : g) {
                totalSq += v * v;
            }
        }
        return totalSq;
    }
}
