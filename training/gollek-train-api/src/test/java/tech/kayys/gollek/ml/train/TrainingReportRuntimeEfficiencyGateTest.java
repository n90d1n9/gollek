package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportRuntimeEfficiencyGateTest {
    @Test
    void passesWhenRuntimeEfficiencyIsMissing() {
        TrainingReportRuntimeEfficiencyGate.Result result =
                TrainingReport.of(Map.of("metadata", Map.of())).runtimeEfficiencyGate();

        assertFalse(result.available());
        assertTrue(result.passed());
        assertEquals("Runtime efficiency metadata is not available.", result.message());
        assertTrue(result.markdown().contains("Runtime efficiency metadata is not available."));
    }

    @Test
    void passesWhenEfficiencyStaysWithinPolicy() {
        TrainingReport report = reportWithRuntimeProfile(90.0, 100.0, 10.0, 2.0, "compute", 45.0);

        TrainingReportRuntimeEfficiencyGate.Result result =
                report.runtimeEfficiencyGate(TrainingReportRuntimeEfficiencyGate.Policy.strict());

        assertTrue(result.available());
        assertTrue(result.passed());
        assertTrue(result.findings().isEmpty());
        assertEquals("Runtime efficiency gate passed.", result.message());
        result.requirePassed();
    }

    @Test
    void reportsAllEfficiencyBudgetFindings() {
        TrainingReport report = reportWithRuntimeProfile(55.0, 100.0, 52.0, 12.0, "input", 88.0);
        TrainingReportRuntimeEfficiencyGate.Policy policy =
                new TrainingReportRuntimeEfficiencyGate.Policy(80.0, 35.0, 75.0);

        TrainingReportRuntimeEfficiencyGate.Result result = report.runtimeEfficiencyGate(policy);

        assertTrue(result.available());
        assertFalse(result.passed());
        assertEquals(List.of(
                        "runtime-efficiency-low-accounted-wall-time",
                        "runtime-efficiency-wall-clock-overhead",
                        "runtime-efficiency-dominant-bottleneck"),
                result.findings().stream()
                        .map(TrainingReportRuntimeEfficiencyGate.Finding::code)
                        .toList());
        assertEquals("failure", result.findings().get(1).severity());
        assertEquals("failure", result.findings().get(2).severity());
        assertEquals("NEEDS_OPTIMIZATION", result.efficiency().get("status"));
        assertEquals(55.0, (double) result.findings().getFirst().evidence().get("accountedPercent"), 1e-12);
        assertEquals(80.0, (double) result.findings().getFirst().evidence().get("thresholdPercent"), 1e-12);
        assertTrue(result.markdown().contains("# Runtime Efficiency Gate"));
        assertTrue(result.markdown().contains("| Largest wall overhead | `trainBatch` `12.000 ms` / `52.000%` |"));
        assertTrue(result.markdown().contains("`runtime-efficiency-dominant-bottleneck`"));
        assertTrue(result.junitXml().startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assertTrue(result.junitXml().contains("name=\"gollek.training.runtime.efficiency\""));
        assertTrue(result.junitXml().contains("policy.minAccountedWallPercent"));
        assertTrue(result.junitXml().contains("gate.findingCodes"));
        assertTrue(result.junitXml().contains("type=\"RUNTIME_EFFICIENCY_LOW_ACCOUNTED_WALL_TIME\""));
        assertTrue(result.junitXml().contains("<system-out># Runtime Efficiency Gate"));
        assertTrue(result.junitXml().contains("# Runtime Efficiency Gate"));
        assertEquals(result.toMap(), Gollek.DL.trainingReportRuntimeEfficiencyGate(report, policy).toMap());
        assertEquals(result.markdown(), Gollek.DL.trainingReportRuntimeEfficiencyGateMarkdown(report, policy));
        assertEquals(result.junitXml(), Gollek.DL.trainingReportRuntimeEfficiencyGateJUnitXml(report, policy));
        assertThrows(IllegalStateException.class, result::requirePassed);
    }

    @Test
    void parsesPolicyFromMapAndSupportsWithers() {
        TrainingReportRuntimeEfficiencyGate.Policy policy =
                TrainingReportRuntimeEfficiencyGate.Policy.fromMap(Map.of(
                        "minAccountedWallPercent", 81.0,
                        "maxWallClockOverheadPercent", 31.0,
                        "maxBottleneckPercent", 71.0));

        assertEquals(81.0, policy.minAccountedWallPercent());
        assertEquals(31.0, policy.maxWallClockOverheadPercent());
        assertEquals(71.0, policy.maxBottleneckPercent());
        assertEquals(82.0, policy.withMinAccountedWallPercent(82.0).minAccountedWallPercent());
        assertEquals(32.0, policy.withMaxWallClockOverheadPercent(32.0).maxWallClockOverheadPercent());
        assertEquals(72.0, policy.withMaxBottleneckPercent(72.0).maxBottleneckPercent());
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
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "train.forward"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", measuredMillis / 2.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 30.0),
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
