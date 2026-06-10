package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

class TrainingReportRuntimeWallClockAssessmentTest {
    @Test
    void reportsUnavailableWhenWallClockProfileIsMissing() {
        TrainingReportRuntimeWallClockAssessment assessment =
                TrainingReportRuntimeWallClockAssessment.assess(TrainingReportRuntimeProfile.WallClock.empty());

        assertFalse(assessment.available());
        assertFalse(assessment.overheadDetected());
        assertEquals(TrainingReportRuntimeWallClockAssessment.Scope.NONE, assessment.scope());
        assertEquals("none", assessment.scopeKey());
    }

    @Test
    void classifiesMeaningfulTrainBatchOverhead() {
        TrainingReportRuntimeWallClockAssessment assessment =
                TrainingReportRuntimeWallClockAssessment.assess(wallClock("trainBatch", 40.0, 8.0));

        assertTrue(assessment.available());
        assertTrue(assessment.overheadDetected());
        assertEquals(TrainingReportRuntimeWallClockAssessment.Scope.TRAIN_BATCH, assessment.scope());
        assertEquals(TrainingReportRecommendation.Priority.HIGH, assessment.priority());
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, assessment.severity());
        assertEquals(TrainingReportRecommendation.Category.TRAINING_DYNAMICS, assessment.category());

        Map<String, Object> evidence = assessment.recommendationEvidence(OptionalDouble.of(20.0), 1);
        assertEquals("trainBatch", evidence.get("scope"));
        assertEquals(20.0, (double) evidence.get("totalMillis"), 1e-12);
        assertEquals(12.0, (double) evidence.get("profiledMillis"), 1e-12);
        assertEquals(8.0, (double) evidence.get("overheadMillis"), 1e-12);
        assertEquals(40.0, (double) evidence.get("overheadPercent"), 1e-12);
        assertEquals(20.0, (double) evidence.get("wallTotalMillis"), 1e-12);
        assertEquals(1, evidence.get("scopeCount"));
    }

    @Test
    void detectsBudgetExcessAndExportsBudgetEvidence() {
        TrainingReportRuntimeWallClockAssessment assessment =
                TrainingReportRuntimeWallClockAssessment.assess(wallClock("optimizerStep", 22.0, 3.0));

        assertTrue(assessment.exceedsBudget(20.0, Double.POSITIVE_INFINITY));
        assertTrue(assessment.exceedsBudget(80.0, 2.0));
        assertFalse(assessment.exceedsBudget(80.0, 10.0));
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, assessment.category());

        Map<String, Object> evidence =
                assessment.budgetEvidence(OptionalDouble.of(13.0), 20.0, Double.POSITIVE_INFINITY);
        assertEquals("optimizerStep", evidence.get("scope"));
        assertEquals(22.0, (double) evidence.get("overheadPercent"), 1e-12);
        assertEquals(20.0, (double) evidence.get("thresholdPercent"), 1e-12);
        assertFalse(evidence.containsKey("thresholdMillis"));
    }

    private static TrainingReportRuntimeProfile.WallClock wallClock(
            String scope,
            double overheadPercent,
            double overheadMillis) {
        TrainingReportRuntimeProfile.WallScope overhead = new TrainingReportRuntimeProfile.WallScope(
                OptionalLong.of(2L),
                OptionalDouble.of(20.0),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.of(12.0),
                OptionalDouble.of(overheadMillis),
                OptionalDouble.of(overheadPercent));
        return new TrainingReportRuntimeProfile.WallClock(
                true,
                OptionalDouble.of(20.0),
                1,
                scope,
                overhead,
                scope.equals("trainBatch") ? overhead : TrainingReportRuntimeProfile.WallScope.empty(),
                scope.equals("validationBatch") ? overhead : TrainingReportRuntimeProfile.WallScope.empty(),
                scope.equals("optimizerStep") ? overhead : TrainingReportRuntimeProfile.WallScope.empty());
    }
}
