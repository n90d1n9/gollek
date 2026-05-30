package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerOptimizationMetadataTest {

    @Test
    void publishesOptimizerGradientAndParameterDiagnostics() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerOptimizationMetadata.put(
                metadata,
                4,
                2,
                11,
                0.75,
                new TrainerOptimizationMetadata.GradientDiagnostics(
                        8.0,
                        0.7,
                        4.0,
                        0.5,
                        2.0,
                        0.25,
                        3.0,
                        0.2,
                        3,
                        15,
                        5,
                        1.0 / 3.0,
                        0.125,
                        true),
                new TrainerOptimizationMetadata.ParameterDiagnostics(
                        12.0,
                        5.0,
                        1.5,
                        2.0,
                        4,
                        20,
                        2,
                        0.1),
                new TrainerOptimizationMetadata.UpdateDiagnostics(
                        true,
                        0.5,
                        0.25,
                        0.05,
                        0.1,
                        4,
                        20L,
                        18L,
                        0.9));

        assertEquals(4, metadata.get("gradientAccumulationSteps"));
        assertEquals(2, metadata.get("pendingGradientAccumulationBatches"));
        assertEquals(11, metadata.get("optimizerStepCount"));
        assertEquals(Boolean.TRUE, metadata.get("gradientClipEnabled"));
        assertEquals(0.75, metadata.get("gradientClipThreshold"));
        assertEquals(Boolean.TRUE, metadata.get("gradientClipNormEnabled"));
        assertEquals(0.75, metadata.get("gradientClipNormThreshold"));
        assertEquals(Boolean.FALSE, metadata.get("gradientClipValueEnabled"));
        assertEquals(0.0, metadata.get("gradientClipValueThreshold"));
        assertEquals(8.0, metadata.get("latestGradientL2NormBeforeClip"));
        assertEquals(0.7, metadata.get("latestGradientL2Norm"));
        assertEquals(4.0, metadata.get("latestGradientMaxAbsBeforeClip"));
        assertEquals(0.5, metadata.get("latestGradientMaxAbs"));
        assertEquals(2.0, metadata.get("latestGradientMeanAbsBeforeClip"));
        assertEquals(0.25, metadata.get("latestGradientMeanAbs"));
        assertEquals(3.0, metadata.get("latestGradientRmsBeforeClip"));
        assertEquals(0.2, metadata.get("latestGradientRms"));
        assertEquals(3, metadata.get("latestGradientParameterCount"));
        assertEquals(15L, metadata.get("latestGradientValueCount"));
        assertEquals(5L, metadata.get("latestGradientZeroCount"));
        assertEquals(1.0 / 3.0, (double) metadata.get("latestGradientZeroFraction"), 1e-12);
        assertEquals(15L, metadata.get("latestGradientFiniteCount"));
        assertEquals(0L, metadata.get("latestGradientNonFiniteCount"));
        assertEquals(0L, metadata.get("latestGradientNanCount"));
        assertEquals(0L, metadata.get("latestGradientPositiveInfinityCount"));
        assertEquals(0L, metadata.get("latestGradientNegativeInfinityCount"));
        assertEquals(0.0, metadata.get("latestGradientNonFiniteFraction"));
        assertEquals(0.125, metadata.get("latestGradientClipScale"));
        assertEquals(Boolean.TRUE, metadata.get("latestGradientClipped"));
        assertEquals(12.0, metadata.get("latestParameterL2Norm"));
        assertEquals(5.0, metadata.get("latestParameterMaxAbs"));
        assertEquals(1.5, metadata.get("latestParameterMeanAbs"));
        assertEquals(2.0, metadata.get("latestParameterRms"));
        assertEquals(4, metadata.get("latestParameterCount"));
        assertEquals(20L, metadata.get("latestParameterValueCount"));
        assertEquals(2L, metadata.get("latestParameterZeroCount"));
        assertEquals(0.1, metadata.get("latestParameterZeroFraction"));
        assertEquals(20L, metadata.get("latestParameterFiniteCount"));
        assertEquals(0L, metadata.get("latestParameterNonFiniteCount"));
        assertEquals(0L, metadata.get("latestParameterNanCount"));
        assertEquals(0L, metadata.get("latestParameterPositiveInfinityCount"));
        assertEquals(0L, metadata.get("latestParameterNegativeInfinityCount"));
        assertEquals(0.0, metadata.get("latestParameterNonFiniteFraction"));
        assertEquals(0.7 / 12.0, (double) metadata.get("latestGradientToParameterL2Ratio"), 1e-12);
        assertEquals(0.1, metadata.get("latestGradientToParameterMaxAbsRatio"));
        assertEquals(1.0 / 6.0, (double) metadata.get("latestGradientToParameterMeanAbsRatio"), 1e-12);
        assertEquals(0.1, metadata.get("latestGradientToParameterRmsRatio"));
        assertEquals(Boolean.TRUE, metadata.get("parameterUpdateDiagnosticsEnabled"));
        assertEquals(0.5, metadata.get("latestParameterUpdateL2Norm"));
        assertEquals(0.25, metadata.get("latestParameterUpdateMaxAbs"));
        assertEquals(0.05, metadata.get("latestParameterUpdateMeanAbs"));
        assertEquals(0.1, metadata.get("latestParameterUpdateRms"));
        assertEquals(4, metadata.get("latestParameterUpdateCount"));
        assertEquals(20L, metadata.get("latestParameterUpdateValueCount"));
        assertEquals(18L, metadata.get("latestParameterUpdateZeroCount"));
        assertEquals(0.9, metadata.get("latestParameterUpdateZeroFraction"));
        assertEquals(20L, metadata.get("latestParameterUpdateFiniteCount"));
        assertEquals(0L, metadata.get("latestParameterUpdateNonFiniteCount"));
        assertEquals(0L, metadata.get("latestParameterUpdateNanCount"));
        assertEquals(0L, metadata.get("latestParameterUpdatePositiveInfinityCount"));
        assertEquals(0L, metadata.get("latestParameterUpdateNegativeInfinityCount"));
        assertEquals(0.0, metadata.get("latestParameterUpdateNonFiniteFraction"));
        assertEquals(0.5 / 12.0, (double) metadata.get("latestParameterUpdateToParameterL2Ratio"), 1e-12);
        assertEquals(0.05, metadata.get("latestParameterUpdateToParameterMaxAbsRatio"));
        assertEquals(1.0 / 30.0, (double) metadata.get("latestParameterUpdateToParameterMeanAbsRatio"), 1e-12);
        assertEquals(0.05, metadata.get("latestParameterUpdateToParameterRmsRatio"));
        assertEquals(0.5 / 0.7, (double) metadata.get("latestParameterUpdateToGradientL2Ratio"), 1e-12);
        assertEquals(0.5, metadata.get("latestParameterUpdateToGradientMaxAbsRatio"));
        assertEquals(0.2, metadata.get("latestParameterUpdateToGradientMeanAbsRatio"));
        assertEquals(0.5, metadata.get("latestParameterUpdateToGradientRmsRatio"));
    }

    @Test
    void reportsDisabledClipForZeroOrNegativeThreshold() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerOptimizationMetadata.put(
                metadata,
                1,
                0,
                0,
                -2.0,
                new TrainerOptimizationMetadata.GradientDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0, false),
                new TrainerOptimizationMetadata.ParameterDiagnostics(0, 0, 0, 0, 0, 0, 0, 0.0),
                TrainerOptimizationMetadata.UpdateDiagnostics.disabled());

        assertEquals(Boolean.FALSE, metadata.get("gradientClipEnabled"));
        assertEquals(0.0, metadata.get("gradientClipThreshold"));
        assertEquals(Boolean.FALSE, metadata.get("gradientClipNormEnabled"));
        assertEquals(0.0, metadata.get("gradientClipNormThreshold"));
        assertEquals(Boolean.FALSE, metadata.get("gradientClipValueEnabled"));
        assertEquals(0.0, metadata.get("gradientClipValueThreshold"));
        assertEquals(Boolean.FALSE, metadata.get("parameterUpdateDiagnosticsEnabled"));
    }

    @Test
    void publishesValueClipConfiguration() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerOptimizationMetadata.put(
                metadata,
                1,
                0,
                0,
                TrainerGradientClipConfig.value(0.25),
                new TrainerOptimizationMetadata.GradientDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0, false),
                new TrainerOptimizationMetadata.ParameterDiagnostics(0, 0, 0, 0, 0, 0, 0, 0.0),
                TrainerOptimizationMetadata.UpdateDiagnostics.disabled());

        assertEquals(Boolean.TRUE, metadata.get("gradientClipEnabled"));
        assertEquals(0.0, metadata.get("gradientClipThreshold"));
        assertEquals(Boolean.FALSE, metadata.get("gradientClipNormEnabled"));
        assertEquals(0.0, metadata.get("gradientClipNormThreshold"));
        assertEquals(Boolean.TRUE, metadata.get("gradientClipValueEnabled"));
        assertEquals(0.25, metadata.get("gradientClipValueThreshold"));
    }

    @Test
    void ratioHandlesZeroAndNonFiniteInputs() {
        assertEquals(0.0, TrainerOptimizationMetadata.ratio(0.0, 0.0));
        assertTrue(Double.isInfinite(TrainerOptimizationMetadata.ratio(1.0, 0.0)));
        assertTrue(Double.isNaN(TrainerOptimizationMetadata.ratio(Double.NaN, 1.0)));
    }
}
