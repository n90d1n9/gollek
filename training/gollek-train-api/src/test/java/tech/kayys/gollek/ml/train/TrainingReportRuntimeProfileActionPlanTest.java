package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportRuntimeProfileActionPlanTest {
    @Test
    void ranksRuntimeProfileTargetsAndRendersMarkdown() {
        TrainingReport report = reportWithRuntimeProfile();

        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();
        Map<String, Object> map = report.runtimeProfileActionPlanMap();
        String markdown = report.runtimeProfileActionPlanMarkdown();

        assertEquals(TrainingReportRuntimeProfileActionPlan.Status.NEEDS_OPTIMIZATION, plan.status());
        assertTrue(plan.available());
        assertTrue(plan.requiresOptimization());
        assertEquals(2, plan.targets().size());
        assertEquals("input", plan.targets().get(0).name());
        assertEquals(TrainingReportRuntimeProfileActionPlan.TargetKind.GROUP, plan.targets().get(0).kind());
        assertEquals(24.0, plan.targets().get(0).totalMillis().orElseThrow(), 1e-12);
        assertEquals("input.train.next", plan.targets().get(1).name());
        assertEquals(TrainingReportRuntimeProfileActionPlan.TargetKind.HOTSPOT, plan.targets().get(1).kind());
        assertEquals(20.0, plan.targets().get(1).totalMillis().orElseThrow(), 1e-12);
        assertTrue(plan.nextActions().stream()
                .anyMatch(action -> action.contains("Prioritize the `train` input loader `next()` path")));
        assertEquals("NEEDS_OPTIMIZATION", map.get("status"));
        assertEquals(2, map.get("targetCount"));
        assertEquals(plan.toMap(), Gollek.DL.trainingReportRuntimeProfileActionPlanMap(report));
        assertEquals(plan, Gollek.DL.trainingReportRuntimeProfileActionPlan(report));
        assertTrue(markdown.startsWith("# Gollek Runtime Profile Action Plan\n"));
        assertTrue(markdown.contains("**Status:** `NEEDS_OPTIMIZATION`"));
        assertTrue(markdown.contains("| 1 | `GROUP` | `input` | `MEDIUM` | `DATA_HEALTH` | 24.000"));
        assertTrue(markdown.contains("| 2 | `HOTSPOT` | `input.train.next` | `MEDIUM` | `DATA_HEALTH` | 20.000"));
        assertTrue(markdown.contains("## Next Actions"));
        assertTrue(markdown.contains("DataLoader.prefetch"));
        assertEquals(markdown, Gollek.DL.trainingReportRuntimeProfileActionPlanMarkdown(report));
        assertFalse(markdown.contains("null"));
    }

    @Test
    void reportsNoProfileWhenRuntimeMetadataIsAbsent() {
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                Map.of("epochHistory", List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8, "learningRate", 0.01))));
        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));

        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();
        String markdown = report.runtimeProfileActionPlanMarkdown();

        assertEquals(TrainingReportRuntimeProfileActionPlan.Status.NO_PROFILE, plan.status());
        assertFalse(plan.available());
        assertFalse(plan.requiresOptimization());
        assertTrue(plan.targets().isEmpty());
        assertTrue(markdown.contains("No runtime profile metadata is available"));
    }

    @Test
    void addsRuntimeBalanceRecommendationWhenInputPipelineDominates() {
        TrainingReport report = reportWithInputBoundRuntimeBalance();

        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();

        assertEquals(TrainingReportRuntimeProfileActionPlan.Status.NEEDS_OPTIMIZATION, plan.status());
        assertTrue(plan.targets().stream()
                .anyMatch(target -> target.diagnosticCode().equals("runtime_profile.balance.input_bound")
                        && target.evidence().get("bottleneckGroup").equals("input")
                        && ((Number) target.evidence().get("inputPercent")).doubleValue() == 62.5));
        assertTrue(plan.nextActions().stream()
                .anyMatch(action -> action.contains("DataLoader.prefetch")));
    }

    @Test
    void addsWallClockOverheadTargetWhenScopeTimeExceedsProfiledPhases() {
        TrainingReport report = reportWithTrainBatchWallClockOverhead();

        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();
        String markdown = report.runtimeProfileActionPlanMarkdown();

        assertEquals(TrainingReportRuntimeProfileActionPlan.Status.NEEDS_OPTIMIZATION, plan.status());
        assertTrue(plan.targets().stream()
                .anyMatch(target -> target.kind() == TrainingReportRuntimeProfileActionPlan.TargetKind.OVERHEAD
                        && target.name().equals("trainBatch")
                        && target.diagnosticCode().equals("runtime_profile.wall_clock.overhead")
                        && target.totalMillis().orElseThrow() == 8.0
                        && target.percentTotal().orElseThrow() == 40.0
                        && target.evidence().get("profiledMillis").equals(12.0)));
        assertTrue(plan.nextActions().stream()
                .anyMatch(action -> action.contains("trainer batch orchestration")));
        assertTrue(markdown.contains("| 1 | `OVERHEAD` | `trainBatch` | `HIGH`"));
        assertTrue(markdown.contains("runtime_profile.wall_clock.overhead"));
    }

    private static TrainingReport reportWithRuntimeProfile() {
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
                        Map.entry("runtimeProfile.groupCount", 1),
                        Map.entry("runtimeProfile.primaryGroup.name", "input"),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 24.0),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", 85.714),
                        Map.entry("runtimeProfile.groups", List.of(Map.of(
                                "name", "input",
                                "count", 12L,
                                "totalMillis", 24.0,
                                "percentTotal", 85.714,
                                "averageMillis", 2.0))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "input.train.next"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", 20.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 71.429),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "input.train.next",
                                "count", 4L,
                                "totalMillis", 20.0,
                                "percentTotal", 71.429,
                                "averageMillis", 5.0))),
                        Map.entry("runtimeProfile.input.train.iterator.count", 1L),
                        Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                        Map.entry("runtimeProfile.input.train.hasNext.count", 6L),
                        Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 3.0),
                        Map.entry("runtimeProfile.input.train.next.count", 4L),
                        Map.entry("runtimeProfile.input.train.next.totalMillis", 20.0),
                        Map.entry("runtimeProfile.input.validation.next.count", 2L),
                        Map.entry("runtimeProfile.input.validation.next.totalMillis", 4.0)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }

    private static TrainingReport reportWithInputBoundRuntimeBalance() {
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
                        Map.entry("runtimeProfile.totalMillis", 80.0),
                        Map.entry("runtimeProfile.groupCount", 2),
                        Map.entry("runtimeProfile.primaryGroup.name", "input"),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 50.0),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", 62.5),
                        Map.entry("runtimeProfile.groups", List.of(
                                Map.of(
                                        "name", "input",
                                        "count", 5L,
                                        "totalMillis", 50.0,
                                        "percentTotal", 62.5,
                                        "averageMillis", 10.0),
                                Map.of(
                                        "name", "train",
                                        "count", 4L,
                                        "totalMillis", 30.0,
                                        "percentTotal", 37.5,
                                        "averageMillis", 7.5))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "input.train.next"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", 40.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 50.0),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "input.train.next",
                                "count", 4L,
                                "totalMillis", 40.0,
                                "percentTotal", 50.0,
                                "averageMillis", 10.0))),
                        Map.entry("runtimeProfile.balance.bottleneckGroup", "input"),
                        Map.entry("runtimeProfile.balance.bottleneck.totalMillis", 50.0),
                        Map.entry("runtimeProfile.balance.bottleneck.percentTotal", 62.5),
                        Map.entry("runtimeProfile.balance.input.totalMillis", 50.0),
                        Map.entry("runtimeProfile.balance.input.percentTotal", 62.5),
                        Map.entry("runtimeProfile.balance.compute.totalMillis", 30.0),
                        Map.entry("runtimeProfile.balance.compute.percentTotal", 37.5),
                        Map.entry("runtimeProfile.balance.train.totalMillis", 30.0),
                        Map.entry("runtimeProfile.balance.train.percentTotal", 37.5),
                        Map.entry("runtimeProfile.balance.validation.totalMillis", 0.0),
                        Map.entry("runtimeProfile.balance.validation.percentTotal", 0.0),
                        Map.entry("runtimeProfile.balance.optimizer.totalMillis", 0.0),
                        Map.entry("runtimeProfile.balance.optimizer.percentTotal", 0.0),
                        Map.entry("runtimeProfile.input.train.next.count", 4L),
                        Map.entry("runtimeProfile.input.train.next.totalMillis", 40.0)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }

    private static TrainingReport reportWithTrainBatchWallClockOverhead() {
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
                        Map.entry("runtimeProfile.totalMillis", 12.0),
                        Map.entry("runtimeProfile.groupCount", 1),
                        Map.entry("runtimeProfile.primaryGroup.name", "train"),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 12.0),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", 100.0),
                        Map.entry("runtimeProfile.groups", List.of(Map.of(
                                "name", "train",
                                "count", 3L,
                                "totalMillis", 12.0,
                                "percentTotal", 100.0,
                                "averageMillis", 4.0))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "train.forward"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", 9.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 75.0),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "train.forward",
                                "count", 2L,
                                "totalMillis", 9.0,
                                "percentTotal", 75.0,
                                "averageMillis", 4.5))),
                        Map.entry("runtimeProfile.wall.totalMillis", 20.0),
                        Map.entry("runtimeProfile.wall.scopeCount", 1),
                        Map.entry("runtimeProfile.wall.primaryOverhead.scope", "trainBatch"),
                        Map.entry("runtimeProfile.wall.primaryOverhead.totalMillis", 20.0),
                        Map.entry("runtimeProfile.wall.primaryOverhead.profiledMillis", 12.0),
                        Map.entry("runtimeProfile.wall.primaryOverhead.overheadMillis", 8.0),
                        Map.entry("runtimeProfile.wall.primaryOverhead.overheadPercent", 40.0),
                        Map.entry("runtimeProfile.wall.trainBatch.count", 2L),
                        Map.entry("runtimeProfile.wall.trainBatch.totalMillis", 20.0),
                        Map.entry("runtimeProfile.wall.trainBatch.averageMillis", 10.0),
                        Map.entry("runtimeProfile.wall.trainBatch.profiledMillis", 12.0),
                        Map.entry("runtimeProfile.wall.trainBatch.overheadMillis", 8.0),
                        Map.entry("runtimeProfile.wall.trainBatch.overheadPercent", 40.0)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }
}
