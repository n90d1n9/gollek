package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportRuntimeProfileBudgetGateTest {
    @TempDir
    Path tempDir;

    @Test
    void passesWhenRuntimeProfileIsMissing() {
        TrainingReportRuntimeProfileBudgetGate.Result result =
                TrainingReport.of(Map.of("metadata", Map.of())).runtimeProfileBudgetGate();

        assertFalse(result.available());
        assertTrue(result.passed());
        assertEquals("Runtime profile is not available.", result.message());
        assertTrue(result.markdown().contains("Runtime profile timings are not available."));
    }

    @Test
    void passesWhenRuntimeProfileStaysWithinBudgets() {
        TrainingReport report = report(40.0, 28.0, 120.0);
        TrainingReportRuntimeProfileBudgetGate.Policy policy =
                new TrainingReportRuntimeProfileBudgetGate.Policy(70.0, 45.0, 250.0);

        TrainingReportRuntimeProfileBudgetGate.Result result = report.runtimeProfileBudgetGate(policy);

        assertTrue(result.available());
        assertTrue(result.passed());
        assertTrue(result.findings().isEmpty());
        assertEquals("Runtime profile budget gate passed.", result.message());
        result.requirePassed();
    }

    @Test
    void warnsWhenPrimaryRuntimeBudgetsAreExceeded() {
        TrainingReport report = report(86.0, 72.0, 320.0);
        TrainingReportRuntimeProfileBudgetGate.Policy policy =
                new TrainingReportRuntimeProfileBudgetGate.Policy(80.0, 60.0, 250.0);

        TrainingReportRuntimeProfileBudgetGate.Result result =
                Gollek.DL.trainingReportRuntimeProfileBudgetGate(report, policy);

        assertTrue(result.available());
        assertFalse(result.passed());
        assertEquals(3, result.findings().size());
        assertEquals(List.of(
                        "runtime-profile-primary-group-budget",
                        "runtime-profile-primary-hotspot-percent-budget",
                        "runtime-profile-primary-hotspot-millis-budget"),
                result.findings().stream()
                        .map(TrainingReportRuntimeProfileBudgetGate.Finding::code)
                        .toList());
        assertEquals("input", result.findings().get(0).evidence().get("group"));
        assertEquals(86.0, (double) result.findings().get(0).evidence().get("percentTotal"), 1e-12);
        assertEquals("input.train.next", result.findings().get(1).evidence().get("phase"));
        assertTrue(result.findings().get(1).action().contains("Prioritize the `train` input loader `next()` path"));
        assertEquals(result.toMap(), report.runtimeProfileBudgetGate(policy).toMap());
        assertThrows(IllegalStateException.class, result::requirePassed);

        String markdown = Gollek.DL.trainingReportRuntimeProfileBudgetGateMarkdown(report, policy);
        assertTrue(markdown.startsWith("# Runtime Profile Budget Gate\n"));
        assertTrue(markdown.contains("- Passed: `false`"));
        assertTrue(markdown.contains("| Primary group | `80.000%` |"));
        assertTrue(markdown.contains("| Primary hotspot total | `250.000 ms` |"));
        assertTrue(markdown.contains("`runtime-profile-primary-hotspot-millis-budget`"));
        assertEquals(markdown, result.markdown());

        String junitXml = Gollek.DL.trainingReportRuntimeProfileBudgetGateJUnitXml(report, policy);
        assertTrue(junitXml.startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assertTrue(junitXml.contains("<testsuite name=\"gollek.training.runtime.profile\" tests=\"1\" failures=\"1\""));
        assertTrue(junitXml.contains("name=\"gate.findingCodes\" value=\"runtime-profile-primary-group-budget,"));
        assertTrue(junitXml.contains("type=\"RUNTIME_PROFILE_PRIMARY_GROUP_BUDGET\""));
        assertTrue(junitXml.contains("<system-out># Runtime Profile Budget Gate"));
        assertEquals(junitXml, result.junitXml());
    }

    @Test
    void buildsPolicyFromMapAndPresets() {
        TrainingReportRuntimeProfileBudgetGate.Policy policy =
                TrainingReportRuntimeProfileBudgetGate.Policy.fromMap(Map.of(
                        "maxPrimaryGroupPercent", "75.5",
                        "maxPrimaryHotspotPercent", 44,
                        "maxPrimaryHotspotTotalMillis", 123.0,
                        "maxInputBalancePercent", 55.0,
                        "maxOptimizerBalancePercent", 35.0,
                        "maxValidationBalancePercent", 50.0));

        assertEquals(75.5, policy.maxPrimaryGroupPercent());
        assertEquals(44.0, policy.maxPrimaryHotspotPercent());
        assertEquals(123.0, policy.maxPrimaryHotspotTotalMillis());
        assertEquals(55.0, policy.maxInputBalancePercent());
        assertEquals(35.0, policy.maxOptimizerBalancePercent());
        assertEquals(50.0, policy.maxValidationBalancePercent());
        assertEquals(70.0, TrainingReportRuntimeProfileBudgetGate.Policy.strict().maxPrimaryGroupPercent());
        assertEquals(45.0, TrainingReportRuntimeProfileBudgetGate.Policy.strict().maxInputBalancePercent());
        assertEquals(90.0, TrainingReportRuntimeProfileBudgetGate.Policy.permissive()
                .maxPrimaryHotspotPercent());
        assertEquals(85.0, TrainingReportRuntimeProfileBudgetGate.Policy.permissive()
                .maxInputBalancePercent());
        assertEquals(33.0, policy.withMaxPrimaryHotspotPercent(33.0).maxPrimaryHotspotPercent());
        assertEquals(40.0, policy.withMaxInputBalancePercent(40.0).maxInputBalancePercent());
        assertEquals(30.0, policy.withMaxOptimizerBalancePercent(30.0).maxOptimizerBalancePercent());
        assertEquals(25.0, policy.withMaxValidationBalancePercent(25.0).maxValidationBalancePercent());
    }

    @Test
    void warnsWhenRuntimeBalanceBudgetsAreExceeded() {
        TrainingReport report = reportWithRuntimeBalance(62.5, 51.0, 63.0);
        TrainingReportRuntimeProfileBudgetGate.Policy policy =
                new TrainingReportRuntimeProfileBudgetGate.Policy(95.0, 95.0, 500.0, 60.0, 50.0, 60.0);

        TrainingReportRuntimeProfileBudgetGate.Result result = report.runtimeProfileBudgetGate(policy);

        assertTrue(result.available());
        assertFalse(result.passed());
        assertEquals(List.of(
                        "runtime-profile-input-balance-budget",
                        "runtime-profile-optimizer-balance-budget",
                        "runtime-profile-validation-balance-budget"),
                result.findings().stream()
                        .map(TrainingReportRuntimeProfileBudgetGate.Finding::code)
                        .toList());
        assertEquals("input", result.findings().get(0).evidence().get("bucket"));
        assertEquals(62.5, (double) result.findings().get(0).evidence().get("percentTotal"), 1e-12);
        assertEquals(60.0, (double) result.findings().get(0).evidence().get("thresholdPercent"), 1e-12);
        assertTrue(result.markdown().contains("| Input balance | `60.000%` |"));
        assertTrue(result.markdown().contains("| Balance bottleneck | `validation` |"));
        assertTrue(result.markdown().contains("`runtime-profile-input-balance-budget`"));
        assertTrue(result.junitXml().contains("runtime-profile-input-balance-budget"));
    }

    @Test
    void writesVerifiesAndRefreshesRuntimeProfileBudgetGateArtifacts() throws IOException {
        TrainingReportRuntimeProfileBudgetGate.Result result = report(86.0, 72.0, 320.0)
                .runtimeProfileBudgetGate(new TrainingReportRuntimeProfileBudgetGate.Policy(80.0, 60.0, 250.0));

        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportRuntimeProfileBudgetGateArtifacts(tempDir, result);
        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportRuntimeProfileBudgetGateArtifacts(tempDir);
        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportRuntimeProfileBudgetGateArtifacts(bundle);

        assertFalse(bundle.passed());
        assertFalse(inspection.passed());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertTrue(inspection.markdown().contains("# Runtime Profile Budget Gate"));
        assertTrue(inspection.junitXml().contains("gollek.training.runtime.profile"));
        assertTrue(verification.passed(), verification.message());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        verification.requirePassed();

        Files.writeString(bundle.markdownFile(), inspection.markdown() + "\n<!-- tampered -->\n");
        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportRuntimeProfileBudgetGateArtifacts(tempDir);
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportRuntimeProfileBudgetGateArtifacts(tempDir);
        assertTrue(Gollek.DL.verifyTrainingReportRuntimeProfileBudgetGateArtifacts(refreshed).passed());
    }

    private static TrainingReport report(
            double primaryGroupPercent,
            double primaryHotspotPercent,
            double primaryHotspotMillis) {
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
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 500.0),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", primaryGroupPercent),
                        Map.entry("runtimeProfile.totalMillis", 600.0),
                        Map.entry("runtimeProfile.groups", List.of(Map.of(
                                "name", "input",
                                "count", 12L,
                                "totalMillis", 500.0,
                                "percentTotal", primaryGroupPercent,
                                "averageMillis", 41.667))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "input.train.next"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", primaryHotspotMillis),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", primaryHotspotPercent),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "input.train.next",
                                "count", 4L,
                                "totalMillis", primaryHotspotMillis,
                                "percentTotal", primaryHotspotPercent,
                                "averageMillis", primaryHotspotMillis / 4.0))),
                        Map.entry("runtimeProfile.input.train.iterator.totalMillis", 10.0),
                        Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 20.0),
                        Map.entry("runtimeProfile.input.train.next.count", 4L),
                        Map.entry("runtimeProfile.input.train.next.totalMillis", primaryHotspotMillis),
                        Map.entry("runtimeProfile.input.validation.next.totalMillis", 15.0)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }

    private static TrainingReport reportWithRuntimeBalance(
            double inputPercent,
            double optimizerPercent,
            double validationPercent) {
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
                        Map.entry("runtimeProfile.totalMillis", 100.0),
                        Map.entry("runtimeProfile.groupCount", 1),
                        Map.entry("runtimeProfile.primaryGroup.name", "train"),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 20.0),
                        Map.entry("runtimeProfile.primaryGroup.percentTotal", 20.0),
                        Map.entry("runtimeProfile.groups", List.of(Map.of(
                                "name", "train",
                                "count", 2L,
                                "totalMillis", 20.0,
                                "percentTotal", 20.0))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "train.forward"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", 15.0),
                        Map.entry("runtimeProfile.primaryHotspot.percentTotal", 15.0),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "train.forward",
                                "count", 2L,
                                "totalMillis", 15.0,
                                "percentTotal", 15.0))),
                        Map.entry("runtimeProfile.balance.bottleneckGroup", "validation"),
                        Map.entry("runtimeProfile.balance.bottleneck.totalMillis", validationPercent),
                        Map.entry("runtimeProfile.balance.bottleneck.percentTotal", validationPercent),
                        Map.entry("runtimeProfile.balance.input.totalMillis", inputPercent),
                        Map.entry("runtimeProfile.balance.input.percentTotal", inputPercent),
                        Map.entry("runtimeProfile.balance.compute.totalMillis", 100.0 - inputPercent),
                        Map.entry("runtimeProfile.balance.compute.percentTotal", 100.0 - inputPercent),
                        Map.entry("runtimeProfile.balance.train.totalMillis", 20.0),
                        Map.entry("runtimeProfile.balance.train.percentTotal", 20.0),
                        Map.entry("runtimeProfile.balance.validation.totalMillis", validationPercent),
                        Map.entry("runtimeProfile.balance.validation.percentTotal", validationPercent),
                        Map.entry("runtimeProfile.balance.optimizer.totalMillis", optimizerPercent),
                        Map.entry("runtimeProfile.balance.optimizer.percentTotal", optimizerPercent)));
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
    }
}
