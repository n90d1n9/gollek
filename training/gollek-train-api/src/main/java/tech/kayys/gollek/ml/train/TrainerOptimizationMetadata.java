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
            ParameterDiagnostics parameters,
            UpdateDiagnostics updates) {
        put(
                metadata,
                gradientAccumulationSteps,
                pendingGradientAccumulationBatches,
                optimizerStepCount,
                TrainerGradientClipConfig.norm(gradientClip),
                gradients,
                parameters,
                updates);
    }

    static void put(
            Map<String, Object> metadata,
            int gradientAccumulationSteps,
            int pendingGradientAccumulationBatches,
            int optimizerStepCount,
            TrainerGradientClipConfig gradientClip,
            GradientDiagnostics gradients,
            ParameterDiagnostics parameters,
            UpdateDiagnostics updates) {
        metadata.put("gradientAccumulationSteps", gradientAccumulationSteps);
        metadata.put("pendingGradientAccumulationBatches", pendingGradientAccumulationBatches);
        metadata.put("optimizerStepCount", optimizerStepCount);
        metadata.put("gradientClipEnabled", gradientClip.enabled());
        metadata.put("gradientClipThreshold", gradientClip.normThreshold());
        metadata.put("gradientClipNormEnabled", gradientClip.normEnabled());
        metadata.put("gradientClipNormThreshold", gradientClip.normThreshold());
        metadata.put("gradientClipValueEnabled", gradientClip.valueEnabled());
        metadata.put("gradientClipValueThreshold", gradientClip.valueThreshold());
        metadata.put("latestGradientL2NormBeforeClip", gradients.l2NormBeforeClip());
        metadata.put("latestGradientL2Norm", gradients.l2Norm());
        metadata.put("latestGradientMaxAbsBeforeClip", gradients.maxAbsBeforeClip());
        metadata.put("latestGradientMaxAbs", gradients.maxAbs());
        metadata.put("latestGradientMeanAbsBeforeClip", gradients.meanAbsBeforeClip());
        metadata.put("latestGradientMeanAbs", gradients.meanAbs());
        metadata.put("latestGradientRmsBeforeClip", gradients.rmsBeforeClip());
        metadata.put("latestGradientRms", gradients.rms());
        metadata.put("latestGradientParameterCount", gradients.parameterCount());
        metadata.put("latestGradientValueCount", gradients.valueCount());
        metadata.put("latestGradientZeroCount", gradients.zeroCount());
        metadata.put("latestGradientZeroFraction", gradients.zeroFraction());
        metadata.put("latestGradientFiniteCount", gradients.finiteCount());
        metadata.put("latestGradientNonFiniteCount", gradients.nonFiniteCount());
        metadata.put("latestGradientNanCount", gradients.nanCount());
        metadata.put("latestGradientPositiveInfinityCount", gradients.positiveInfinityCount());
        metadata.put("latestGradientNegativeInfinityCount", gradients.negativeInfinityCount());
        metadata.put("latestGradientNonFiniteFraction", gradients.nonFiniteFraction());
        metadata.put("latestGradientClipScale", gradients.clipScale());
        metadata.put("latestGradientClipped", gradients.clipped());
        metadata.put("latestParameterL2Norm", parameters.l2Norm());
        metadata.put("latestParameterMaxAbs", parameters.maxAbs());
        metadata.put("latestParameterMeanAbs", parameters.meanAbs());
        metadata.put("latestParameterRms", parameters.rms());
        metadata.put("latestParameterCount", parameters.count());
        metadata.put("latestParameterValueCount", parameters.valueCount());
        metadata.put("latestParameterZeroCount", parameters.zeroCount());
        metadata.put("latestParameterZeroFraction", parameters.zeroFraction());
        metadata.put("latestParameterFiniteCount", parameters.finiteCount());
        metadata.put("latestParameterNonFiniteCount", parameters.nonFiniteCount());
        metadata.put("latestParameterNanCount", parameters.nanCount());
        metadata.put("latestParameterPositiveInfinityCount", parameters.positiveInfinityCount());
        metadata.put("latestParameterNegativeInfinityCount", parameters.negativeInfinityCount());
        metadata.put("latestParameterNonFiniteFraction", parameters.nonFiniteFraction());
        metadata.put("latestGradientToParameterL2Ratio", ratio(gradients.l2Norm(), parameters.l2Norm()));
        metadata.put("latestGradientToParameterMaxAbsRatio", ratio(gradients.maxAbs(), parameters.maxAbs()));
        metadata.put("latestGradientToParameterMeanAbsRatio", ratio(gradients.meanAbs(), parameters.meanAbs()));
        metadata.put("latestGradientToParameterRmsRatio", ratio(gradients.rms(), parameters.rms()));
        metadata.put("parameterUpdateDiagnosticsEnabled", updates.enabled());
        metadata.put("latestParameterUpdateL2Norm", updates.l2Norm());
        metadata.put("latestParameterUpdateMaxAbs", updates.maxAbs());
        metadata.put("latestParameterUpdateMeanAbs", updates.meanAbs());
        metadata.put("latestParameterUpdateRms", updates.rms());
        metadata.put("latestParameterUpdateCount", updates.count());
        metadata.put("latestParameterUpdateValueCount", updates.valueCount());
        metadata.put("latestParameterUpdateZeroCount", updates.zeroCount());
        metadata.put("latestParameterUpdateZeroFraction", updates.zeroFraction());
        metadata.put("latestParameterUpdateFiniteCount", updates.finiteCount());
        metadata.put("latestParameterUpdateNonFiniteCount", updates.nonFiniteCount());
        metadata.put("latestParameterUpdateNanCount", updates.nanCount());
        metadata.put("latestParameterUpdatePositiveInfinityCount", updates.positiveInfinityCount());
        metadata.put("latestParameterUpdateNegativeInfinityCount", updates.negativeInfinityCount());
        metadata.put("latestParameterUpdateNonFiniteFraction", updates.nonFiniteFraction());
        metadata.put("latestParameterUpdateToParameterL2Ratio", ratio(updates.l2Norm(), parameters.l2Norm()));
        metadata.put("latestParameterUpdateToParameterMaxAbsRatio", ratio(updates.maxAbs(), parameters.maxAbs()));
        metadata.put("latestParameterUpdateToParameterMeanAbsRatio", ratio(updates.meanAbs(), parameters.meanAbs()));
        metadata.put("latestParameterUpdateToParameterRmsRatio", ratio(updates.rms(), parameters.rms()));
        metadata.put("latestParameterUpdateToGradientL2Ratio", ratio(updates.l2Norm(), gradients.l2Norm()));
        metadata.put("latestParameterUpdateToGradientMaxAbsRatio", ratio(updates.maxAbs(), gradients.maxAbs()));
        metadata.put("latestParameterUpdateToGradientMeanAbsRatio", ratio(updates.meanAbs(), gradients.meanAbs()));
        metadata.put("latestParameterUpdateToGradientRmsRatio", ratio(updates.rms(), gradients.rms()));
    }

    static double ratio(double numerator, double denominator) {
        if (!Double.isFinite(numerator) || !Double.isFinite(denominator)) {
            return Double.NaN;
        }
        if (denominator == 0.0) {
            return numerator == 0.0 ? 0.0 : Double.POSITIVE_INFINITY;
        }
        return numerator / denominator;
    }

    record GradientDiagnostics(
            double l2NormBeforeClip,
            double l2Norm,
            double maxAbsBeforeClip,
            double maxAbs,
            double meanAbsBeforeClip,
            double meanAbs,
            double rmsBeforeClip,
            double rms,
            int parameterCount,
            long valueCount,
            long zeroCount,
            double zeroFraction,
            long finiteCount,
            long nonFiniteCount,
            long nanCount,
            long positiveInfinityCount,
            long negativeInfinityCount,
            double nonFiniteFraction,
            double clipScale,
            boolean clipped) {
        GradientDiagnostics(
                double l2NormBeforeClip,
                double l2Norm,
                double maxAbsBeforeClip,
                double maxAbs,
                double meanAbsBeforeClip,
                double meanAbs,
                double rmsBeforeClip,
                double rms,
                int parameterCount,
                long valueCount,
                long zeroCount,
                double zeroFraction,
                double clipScale,
                boolean clipped) {
            this(
                    l2NormBeforeClip,
                    l2Norm,
                    maxAbsBeforeClip,
                    maxAbs,
                    meanAbsBeforeClip,
                    meanAbs,
                    rmsBeforeClip,
                    rms,
                    parameterCount,
                    valueCount,
                    zeroCount,
                    zeroFraction,
                    Math.max(0L, valueCount),
                    0L,
                    0L,
                    0L,
                    0L,
                    0.0,
                    clipScale,
                    clipped);
        }
    }

    record ParameterDiagnostics(
            double l2Norm,
            double maxAbs,
            double meanAbs,
            double rms,
            int count,
            long valueCount,
            long zeroCount,
            double zeroFraction,
            long finiteCount,
            long nonFiniteCount,
            long nanCount,
            long positiveInfinityCount,
            long negativeInfinityCount,
            double nonFiniteFraction) {
        ParameterDiagnostics(
                double l2Norm,
                double maxAbs,
                double meanAbs,
                double rms,
                int count,
                long valueCount,
                long zeroCount,
                double zeroFraction) {
            this(
                    l2Norm,
                    maxAbs,
                    meanAbs,
                    rms,
                    count,
                    valueCount,
                    zeroCount,
                    zeroFraction,
                    Math.max(0L, valueCount),
                    0L,
                    0L,
                    0L,
                    0L,
                    0.0);
        }
    }

    record UpdateDiagnostics(
            boolean enabled,
            double l2Norm,
            double maxAbs,
            double meanAbs,
            double rms,
            int count,
            long valueCount,
            long zeroCount,
            double zeroFraction,
            long finiteCount,
            long nonFiniteCount,
            long nanCount,
            long positiveInfinityCount,
            long negativeInfinityCount,
            double nonFiniteFraction) {
        UpdateDiagnostics(
                boolean enabled,
                double l2Norm,
                double maxAbs,
                double meanAbs,
                double rms,
                int count,
                long valueCount,
                long zeroCount,
                double zeroFraction) {
            this(
                    enabled,
                    l2Norm,
                    maxAbs,
                    meanAbs,
                    rms,
                    count,
                    valueCount,
                    zeroCount,
                    zeroFraction,
                    Math.max(0L, valueCount),
                    0L,
                    0L,
                    0L,
                    0L,
                    0.0);
        }

        static UpdateDiagnostics disabled() {
            return new UpdateDiagnostics(false, 0.0, 0.0, 0.0, 0.0, 0, 0L, 0L, 0.0, 0L, 0L, 0L, 0L, 0L, 0.0);
        }

        static UpdateDiagnostics enabled(TensorDiagnostics diagnostics) {
            return new UpdateDiagnostics(
                    true,
                    diagnostics.l2Norm(),
                    diagnostics.maxAbs(),
                    diagnostics.meanAbs(),
                    diagnostics.rms(),
                    diagnostics.tensorCount(),
                    diagnostics.valueCount(),
                    diagnostics.zeroCount(),
                    diagnostics.zeroFraction(),
                    diagnostics.finiteCount(),
                    diagnostics.nonFiniteCount(),
                    diagnostics.nanCount(),
                    diagnostics.positiveInfinityCount(),
                    diagnostics.negativeInfinityCount(),
                    diagnostics.nonFiniteFraction());
        }
    }
}
