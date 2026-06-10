package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportRuntimeEfficiencyAdvisorTest {
    @Test
    void addsActionableRecommendationsForInefficientRuntimeProfile() {
        TrainingReport report = reportWithRuntimeProfile(42.0, 100.0, 48.0, 14.0, "input", 88.0);

        List<TrainingReportRecommendation> recommendations = report.actionPlan().recommendations();

        TrainingReportRecommendation lowAccounted = recommendation(
                recommendations,
                "runtime_profile.efficiency.low_accounted_wall_time");
        assertEquals(TrainingReportRecommendation.Priority.HIGH, lowAccounted.priority());
        assertEquals(TrainingReportRecommendation.Category.REPORTING, lowAccounted.category());
        assertEquals(42.0, (double) lowAccounted.evidence().get("accountedPercent"), 1e-12);
        assertTrue(lowAccounted.actions().stream().anyMatch(action -> action.contains("unmeasured trainer loop")));

        TrainingReportRecommendation overhead = recommendation(
                recommendations,
                "runtime_profile.efficiency.wall_clock_overhead");
        assertEquals(TrainingReportRecommendation.Priority.MEDIUM, overhead.priority());
        assertEquals(TrainingReportRecommendation.Category.TRAINING_DYNAMICS, overhead.category());
        assertEquals("trainBatch", overhead.evidence().get("overheadScope"));
        assertTrue(overhead.actions().stream().anyMatch(action -> action.contains("batch orchestration")));

        TrainingReportRecommendation bottleneck = recommendation(
                recommendations,
                "runtime_profile.efficiency.dominant_bottleneck");
        assertEquals(TrainingReportRecommendation.Priority.HIGH, bottleneck.priority());
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH, bottleneck.category());
        assertEquals("input", bottleneck.evidence().get("bottleneck"));
        assertTrue(bottleneck.actions().stream().anyMatch(action -> action.contains("DataLoader prefetching")));
    }

    @Test
    void includesEfficiencyRecommendationsInRuntimeProfileActionPlan() {
        TrainingReport report = reportWithRuntimeProfile(42.0, 100.0, 48.0, 14.0, "input", 88.0);

        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();
        String markdown = report.runtimeProfileActionPlanMarkdown();

        assertEquals(TrainingReportRuntimeProfileActionPlan.Status.NEEDS_OPTIMIZATION, plan.status());
        assertTrue(plan.targets().stream()
                .anyMatch(target -> target.diagnosticCode()
                        .equals("runtime_profile.efficiency.wall_clock_overhead")));
        assertTrue(plan.targets().stream()
                .anyMatch(target -> target.diagnosticCode()
                        .equals("runtime_profile.efficiency.dominant_bottleneck")));
        assertTrue(plan.nextActions().stream()
                .anyMatch(action -> action.contains("DataLoader prefetching")));
        assertTrue(markdown.contains("runtime_profile.efficiency.dominant_bottleneck"));
    }

    @Test
    void staysQuietForEfficientRuntimeProfile() {
        TrainingReport report = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "compute", 45.0);

        assertTrue(report.actionPlan().recommendations().stream()
                .noneMatch(recommendation -> recommendation.diagnosticCode().startsWith("runtime_profile.efficiency.")));
    }

    private static TrainingReportRecommendation recommendation(
            List<TrainingReportRecommendation> recommendations,
            String code) {
        return recommendations.stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals(code))
                .findFirst()
                .orElseThrow();
    }

    private static TrainingReport reportWithRuntimeProfile(
            double measuredMillis,
            double wallMillis,
            double overheadPercent,
            double overheadMillis,
            String bottleneck,
            double bottleneckPercent) {
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                Map.ofEntries(
                        Map.entry("epochHistory", List.of(
                                Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8, "learningRate", 0.01))),
                        Map.entry("runtimeProfile.totalMillis", measuredMillis),
                        Map.entry("runtimeProfile.balance.bottleneckGroup", bottleneck),
                        Map.entry("runtimeProfile.balance." + bottleneck + ".totalMillis", bottleneckPercent),
                        Map.entry("runtimeProfile.balance." + bottleneck + ".percentTotal", bottleneckPercent),
                        Map.entry("runtimeProfile.groupCount", 1),
                        Map.entry("runtimeProfile.primaryGroup.name", bottleneck),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", measuredMillis),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", bottleneckPercent),
                        Map.entry("runtimeProfile.groups", List.of(Map.of(
                                "name", bottleneck,
                                "count", 3L,
                                "totalMillis", measuredMillis,
                                "percentTotal", bottleneckPercent,
                                "averageMillis", measuredMillis / 3.0))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "train.forward"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", measuredMillis / 2.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 30.0),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "train.forward",
                                "count", 2L,
                                "totalMillis", measuredMillis / 2.0,
                                "percentTotal", 30.0,
                                "averageMillis", measuredMillis / 4.0))),
                        Map.entry("runtimeProfile.wall.totalMillis", wallMillis),
                        Map.entry("runtimeProfile.wall.scopeCount", 1),
                        Map.entry("runtimeProfile.wall.primaryOverhead.scope", "trainBatch"),
                        Map.entry("runtimeProfile.wall.primaryOverhead.totalMillis", wallMillis),
                        Map.entry("runtimeProfile.wall.primaryOverhead.profiledMillis", measuredMillis),
                        Map.entry("runtimeProfile.wall.primaryOverhead.overheadMillis", overheadMillis),
                        Map.entry("runtimeProfile.wall.primaryOverhead.overheadPercent", overheadPercent),
                        Map.entry("runtimeProfile.wall.trainBatch.count", 2L),
                        Map.entry("runtimeProfile.wall.trainBatch.totalMillis", wallMillis),
                        Map.entry("runtimeProfile.wall.trainBatch.profiledMillis", measuredMillis),
                        Map.entry("runtimeProfile.wall.trainBatch.overheadMillis", overheadMillis),
                        Map.entry("runtimeProfile.wall.trainBatch.overheadPercent", overheadPercent)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }
}
