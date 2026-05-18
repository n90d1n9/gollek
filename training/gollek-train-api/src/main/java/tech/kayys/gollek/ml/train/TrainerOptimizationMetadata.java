package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Publishes optimizer, gradient, and parameter diagnostics for trainer summaries.
 */
final class TrainerOptimizationMetadata {
    private TrainerOptimizationMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            int gradientAccumulationSteps,
            int pendingGradientAccumulationBatches,
            int optimizerStepCount,
            double gradientClip,
            GradientDiagnostics gradients,
            ParameterDiagnostics parameters) {
        metadata.put("gradientAccumulationSteps", gradientAccumulationSteps);
        metadata.put("pendingGradientAccumulationBatches", pendingGradientAccumulationBatches);
        metadata.put("optimizerStepCount", optimizerStepCount);
        metadata.put("gradientClipEnabled", gradientClip > 0.0);
        metadata.put("gradientClipThreshold", Math.max(0.0, gradientClip));
        metadata.put("latestGradientL2NormBeforeClip", gradients.l2NormBeforeClip());
        metadata.put("latestGradientL2Norm", gradients.l2Norm());
        metadata.put("latestGradientMaxAbsBeforeClip", gradients.maxAbsBeforeClip());
        metadata.put("latestGradientMaxAbs", gradients.maxAbs());
        metadata.put("latestGradientParameterCount", gradients.parameterCount());
        metadata.put("latestGradientValueCount", gradients.valueCount());
        metadata.put("latestGradientClipped", gradients.clipped());
        metadata.put("latestParameterL2Norm", parameters.l2Norm());
        metadata.put("latestParameterMaxAbs", parameters.maxAbs());
        metadata.put("latestParameterCount", parameters.count());
        metadata.put("latestParameterValueCount", parameters.valueCount());
    }

    record GradientDiagnostics(
            double l2NormBeforeClip,
            double l2Norm,
            double maxAbsBeforeClip,
            double maxAbs,
            int parameterCount,
            long valueCount,
            boolean clipped) {
    }

    record ParameterDiagnostics(
            double l2Norm,
            double maxAbs,
            int count,
            long valueCount) {
    }
}
