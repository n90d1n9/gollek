package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;

/**
 * Minimal loss helpers for transition-mean supervision in DiffusionOPD.
 *
 * <p>This utility sits directly under {@link TransitionMeanAdapter}: adapters
 * convert model outputs into scheduler-aligned transition means, and this
 * helper evaluates the corresponding mean-matching objective.
 */
final class DiffusionOpdLosses {

    private DiffusionOpdLosses() {
    }

    static Tensor meanMatchingLoss(
            Tensor studentMean,
            Tensor teacherMean,
            DiffusionSamplerType samplerType,
            float stepVariance) {
        // DiffusionOPD reduces per-step supervision to mean matching; in the
        // stochastic case this corresponds to a closed-form KL scaled by the
        // scheduler variance, while the deterministic case becomes a pure L2
        // transition objective. See arXiv:2605.15055.
        Tensor squaredDifference = studentMean.sub(teacherMean).pow(2.0f).mean();
        if (samplerType == DiffusionSamplerType.SDE) {
            float safeVariance = Math.max(stepVariance, 1.0e-6f);
            return squaredDifference.div(2.0f * safeVariance);
        }
        return squaredDifference.mul(0.5f);
    }
}
