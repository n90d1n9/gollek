package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportRuntimeProfileBalanceAssessmentTest {
    @Test
    void classifiesRuntimeBalanceWithDefaultThresholds() {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(balance(
                        "input",
                        80.0,
                        50.0,
                        62.5,
                        30.0,
                        37.5,
                        25.0,
                        31.25,
                        5.0,
                        6.25,
                        0.0,
                        0.0));

        assertTrue(assessment.available());
        assertTrue(assessment.requiresAttention());
        assertTrue(assessment.inputBound());
        assertFalse(assessment.optimizerBound());
        assertFalse(assessment.validationBound());
        assertFalse(assessment.trainComputeBound());
        assertEquals(TrainingReportRuntimeProfile.BalanceBucket.INPUT, assessment.dominantBucket());
        assertEquals(62.5, assessment.dominantPercent().orElseThrow(), 1e-12);
        assertEquals(50.0,
                assessment.totalMillis(TrainingReportRuntimeProfile.BalanceBucket.INPUT).orElseThrow(), 1e-12);
        assertEquals(62.5,
                assessment.percentTotal(TrainingReportRuntimeProfile.BalanceBucket.INPUT).orElseThrow(), 1e-12);
    }

    @Test
    void separatesDiagnosticThresholdsFromStrictBudgetExceedance() {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(balance(
                        "input",
                        100.0,
                        50.0,
                        50.0,
                        50.0,
                        50.0,
                        40.0,
                        40.0,
                        10.0,
                        10.0,
                        0.0,
                        0.0));

        assertTrue(assessment.inputBound());
        assertFalse(assessment.exceeds(TrainingReportRuntimeProfile.BalanceBucket.INPUT, 50.0));
        assertTrue(assessment.exceeds(TrainingReportRuntimeProfile.BalanceBucket.INPUT, 49.999));
    }

    @Test
    void supportsCustomThresholdsAndStableExportMaps() {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(
                        balance(
                                "optimizer",
                                100.0,
                                30.0,
                                30.0,
                                70.0,
                                70.0,
                                20.0,
                                20.0,
                                5.0,
                                5.0,
                                65.0,
                                65.0),
                        new TrainingReportRuntimeProfileBalanceAssessment.Thresholds(80.0, 60.0, 40.0, 80.0));

        Map<String, Object> map = assessment.toMap();
        Map<String, Object> evidence = assessment.recommendationEvidence();

        assertFalse(assessment.inputBound());
        assertTrue(assessment.optimizerBound());
        assertFalse(assessment.validationBound());
        assertFalse(assessment.trainComputeBound());
        assertEquals(TrainingReportRuntimeProfile.BalanceBucket.OPTIMIZER, assessment.dominantBucket());
        assertEquals(Boolean.TRUE, map.get("optimizerBound"));
        assertEquals("OPTIMIZER", map.get("dominantBucket"));
        assertEquals("optimizer", evidence.get("bottleneckGroup"));
        assertEquals("OPTIMIZER", evidence.get("dominantBucket"));
        assertEquals(65.0, (double) evidence.get("optimizerPercent"), 1e-12);
        assertEquals(65.0, (double) evidence.get("optimizerMillis"), 1e-12);
    }

    @Test
    void handlesMissingBalanceAsUnavailable() {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(null);

        assertFalse(assessment.available());
        assertFalse(assessment.requiresAttention());
        assertEquals(TrainingReportRuntimeProfile.BalanceBucket.NONE, assessment.dominantBucket());
        assertTrue(assessment.toMap().containsKey("thresholds"));
    }

    private static TrainingReportRuntimeProfile.Balance balance(
            String bottleneckGroup,
            double totalMillis,
            double inputMillis,
            double inputPercent,
            double computeMillis,
            double computePercent,
            double trainMillis,
            double trainPercent,
            double validationMillis,
            double validationPercent,
            double optimizerMillis,
            double optimizerPercent) {
        return TrainingReportRuntimeProfile.Balance.fromMetadata(Map.ofEntries(
                Map.entry("runtimeProfile.totalMillis", totalMillis),
                Map.entry("runtimeProfile.balance.bottleneckGroup", bottleneckGroup),
                Map.entry("runtimeProfile.balance.bottleneck.totalMillis", inputMillis),
                Map.entry("runtimeProfile.balance.bottleneck.percentTotal", inputPercent),
                Map.entry("runtimeProfile.balance.input.totalMillis", inputMillis),
                Map.entry("runtimeProfile.balance.input.percentTotal", inputPercent),
                Map.entry("runtimeProfile.balance.compute.totalMillis", computeMillis),
                Map.entry("runtimeProfile.balance.compute.percentTotal", computePercent),
                Map.entry("runtimeProfile.balance.train.totalMillis", trainMillis),
                Map.entry("runtimeProfile.balance.train.percentTotal", trainPercent),
                Map.entry("runtimeProfile.balance.validation.totalMillis", validationMillis),
                Map.entry("runtimeProfile.balance.validation.percentTotal", validationPercent),
                Map.entry("runtimeProfile.balance.optimizer.totalMillis", optimizerMillis),
                Map.entry("runtimeProfile.balance.optimizer.percentTotal", optimizerPercent)));
    }
}
