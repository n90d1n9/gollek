package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
                        3,
                        15,
                        true),
                new TrainerOptimizationMetadata.ParameterDiagnostics(
                        12.0,
                        5.0,
                        4,
                        20));

        assertEquals(4, metadata.get("gradientAccumulationSteps"));
        assertEquals(2, metadata.get("pendingGradientAccumulationBatches"));
        assertEquals(11, metadata.get("optimizerStepCount"));
        assertEquals(Boolean.TRUE, metadata.get("gradientClipEnabled"));
        assertEquals(0.75, metadata.get("gradientClipThreshold"));
        assertEquals(8.0, metadata.get("latestGradientL2NormBeforeClip"));
        assertEquals(0.7, metadata.get("latestGradientL2Norm"));
        assertEquals(4.0, metadata.get("latestGradientMaxAbsBeforeClip"));
        assertEquals(0.5, metadata.get("latestGradientMaxAbs"));
        assertEquals(3, metadata.get("latestGradientParameterCount"));
        assertEquals(15L, metadata.get("latestGradientValueCount"));
        assertEquals(Boolean.TRUE, metadata.get("latestGradientClipped"));
        assertEquals(12.0, metadata.get("latestParameterL2Norm"));
        assertEquals(5.0, metadata.get("latestParameterMaxAbs"));
        assertEquals(4, metadata.get("latestParameterCount"));
        assertEquals(20L, metadata.get("latestParameterValueCount"));
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
                new TrainerOptimizationMetadata.GradientDiagnostics(0, 0, 0, 0, 0, 0, false),
                new TrainerOptimizationMetadata.ParameterDiagnostics(0, 0, 0, 0));

        assertEquals(Boolean.FALSE, metadata.get("gradientClipEnabled"));
        assertEquals(0.0, metadata.get("gradientClipThreshold"));
    }
}
