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

class TrainingReportRuntimeEfficiencyTest {
    @Test
    void reportsNoProfileWhenRuntimeMetadataIsMissing() {
        TrainingReportRuntimeEfficiency efficiency =
                TrainingReport.of(Map.of("metadata", Map.of())).runtimeEfficiency();

        assertFalse(efficiency.available());
        assertEquals(TrainingReportRuntimeEfficiency.Status.NO_PROFILE, efficiency.status());
        assertTrue(efficiency.toMap().containsKey("status"));
        assertEquals("", TrainingReportRuntimeEfficiencyMarkdown.render(efficiency));
    }

    @Test
    void summarizesEfficientRuntimeProfile() {
        TrainingReport report = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "compute", 45.0);

        TrainingReportRuntimeEfficiency efficiency = report.runtimeEfficiency();

        assertTrue(efficiency.available());
        assertEquals(TrainingReportRuntimeEfficiency.Status.EFFICIENT, efficiency.status());
        assertEquals(90.0, efficiency.measuredMillis().orElseThrow(), 1e-12);
        assertEquals(100.0, efficiency.wallMillis().orElseThrow(), 1e-12);
        assertEquals(90.0, efficiency.accountedPercent().orElseThrow(), 1e-12);
        assertEquals("compute", efficiency.bottleneck());
        assertEquals("train.forward", efficiency.primaryHotspot());
        assertEquals(efficiency.toMap(), report.runtimeEfficiencyMap());
        assertEquals(efficiency, Gollek.DL.trainingReportRuntimeEfficiency(report));
    }

    @Test
    void flagsRuntimeProfileThatNeedsOptimization() {
        TrainingReport report = reportWithRuntimeProfile(90.0, 100.0, 42.0, 9.0, "input", 82.0);

        TrainingReportRuntimeEfficiency efficiency = report.runtimeEfficiency();
        String markdown = report.runtimeEfficiencyMarkdown();
        String profileMarkdown = report.runtimeProfileMarkdown();

        assertTrue(efficiency.needsOptimization());
        assertEquals(TrainingReportRuntimeEfficiency.Status.NEEDS_OPTIMIZATION, efficiency.status());
        assertEquals("trainBatch", efficiency.overheadScope());
        assertEquals(42.0, efficiency.overheadPercent().orElseThrow(), 1e-12);
        assertEquals(82.0, efficiency.bottleneckPercent().orElseThrow(), 1e-12);
        assertTrue(markdown.contains("**Status:** `NEEDS_OPTIMIZATION`"));
        assertTrue(markdown.contains("| Largest wall overhead | `trainBatch` `9.000 ms` / `42.000%` |"));
        assertTrue(markdown.contains("| Dominant bottleneck | `input` `82.000%` |"));
        assertTrue(profileMarkdown.contains("### Runtime Efficiency"));
        assertEquals(markdown, Gollek.DL.trainingReportRuntimeEfficiencyMarkdown(report));
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
