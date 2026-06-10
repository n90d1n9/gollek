package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportRuntimeEfficiencyRegressionTest {
    @TempDir
    Path tempDir;

    @Test
    void summarizesRuntimeEfficiencyRegressionsBetweenReports() {
        TrainingReport baseline = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "input", 60.0);
        TrainingReport candidate = reportWithRuntimeProfile(50.0, 100.0, 25.0, 8.0, "input", 80.0);

        TrainingReportRuntimeRegressionSummary summary =
                TrainingReportAdvisor.runtimeRegressionSummary(baseline.payload(), candidate.payload());

        assertTrue(summary.available());
        assertTrue(summary.regressed());
        assertTrue(summary.accountedWallTime().orElseThrow().regressed());
        assertEquals(-40.0, summary.accountedWallTime().orElseThrow().delta(), 1e-12);
        assertTrue(summary.wallClockOverhead().orElseThrow().regressed());
        assertEquals(17.0, summary.wallClockOverhead().orElseThrow().delta(), 1e-12);
        assertTrue(summary.dominantBottleneck().orElseThrow().regressed());
        assertEquals("input", summary.dominantBottleneck().orElseThrow().key());
        assertEquals(summary.toMap(), TrainingReportRuntimeRegressionSummary.fromMap(summary.toMap()).toMap());

        String markdown = TrainingReportAdvisor.runtimeRegressionMarkdown(baseline.payload(), candidate.payload());
        assertTrue(markdown.contains("| Efficiency Signal | Key | Baseline | Candidate | Delta | Threshold | Regressed |"));
        assertTrue(markdown.contains("| accounted wall time | `accountedWallTime` | 90.000% | 50.000% | -40.000% | 15.000% | `yes` |"));
        assertTrue(markdown.contains("| wall-clock overhead | `trainBatch` | 8.000% | 25.000% | 17.000% | 10.000% | `yes` |"));
        assertTrue(markdown.contains("| dominant bottleneck | `input` | 60.000% | 80.000% | 20.000% | 15.000% | `yes` |"));
    }

    @Test
    void addsRuntimeEfficiencyRegressionRecommendationsToComparisonPlan() {
        TrainingReport baseline = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "input", 60.0);
        TrainingReport candidate = reportWithRuntimeProfile(50.0, 100.0, 25.0, 8.0, "input", 80.0);

        TrainingReportActionPlan plan = TrainingReportAdvisor.actionPlan(baseline.payload(), candidate.payload());

        TrainingReportRecommendation accounted = recommendation(
                plan.recommendations(),
                "runtime_profile.efficiency.accounted_wall_time_regressed");
        assertEquals(TrainingReportRecommendation.Category.REPORTING, accounted.category());
        assertEquals(-40.0, (double) accounted.evidence().get("delta"), 1e-12);

        TrainingReportRecommendation overhead = recommendation(
                plan.recommendations(),
                "runtime_profile.efficiency.wall_clock_overhead_regressed");
        assertEquals(TrainingReportRecommendation.Category.TRAINING_DYNAMICS, overhead.category());
        assertEquals("trainBatch", overhead.evidence().get("key"));

        TrainingReportRecommendation bottleneck = recommendation(
                plan.recommendations(),
                "runtime_profile.efficiency.dominant_bottleneck_regressed");
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH, bottleneck.category());
        assertTrue(bottleneck.actions().stream().anyMatch(action -> action.contains("DataLoader")));
    }

    @Test
    void exposesRuntimeEfficiencyRegressionAsCiGate() {
        TrainingReport baseline = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "input", 60.0);
        TrainingReport candidate = reportWithRuntimeProfile(50.0, 100.0, 25.0, 8.0, "input", 80.0);

        TrainingReportRuntimeRegressionGate.Result result =
                TrainingReportAdvisor.runtimeRegressionGate(baseline.payload(), candidate.payload());

        assertTrue(result.available());
        assertTrue(result.runtimeRegression().regressed());
        assertEquals(false, result.passed());
        assertEquals(List.of(
                        "runtime-regression-accounted-wall-time",
                        "runtime-regression-wall-clock-overhead",
                        "runtime-regression-dominant-bottleneck"),
                result.findings().stream()
                        .map(TrainingReportRuntimeRegressionGate.Finding::code)
                        .toList());
        assertTrue(result.markdown().contains("# Runtime Regression Gate"));
        assertTrue(result.markdown().contains("runtime-regression-dominant-bottleneck"));
        assertTrue(result.junitXml().contains("name=\"gollek.training.runtime.regression\""));
        assertTrue(result.junitXml().contains("runtimeRegression.regressed"));
        assertTrue(result.junitXml().contains("type=\"RUNTIME_REGRESSION_ACCOUNTED_WALL_TIME\""));
        assertEquals(result.toMap(), Gollek.DL.trainingReportRuntimeRegressionGate(baseline, candidate).toMap());
        assertEquals(result.markdown(), Gollek.DL.trainingReportRuntimeRegressionGateMarkdown(baseline, candidate));
        assertEquals(result.junitXml(), Gollek.DL.trainingReportRuntimeRegressionGateJUnitXml(baseline, candidate));
    }

    @Test
    void writesRuntimeRegressionGateArtifacts() throws Exception {
        TrainingReport baseline = reportWithRuntimeProfile(90.0, 100.0, 8.0, 1.0, "input", 60.0);
        TrainingReport candidate = reportWithRuntimeProfile(50.0, 100.0, 25.0, 8.0, "input", 80.0);

        TrainingReportRuntimeRegressionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportRuntimeRegressionGateArtifacts(tempDir, baseline, candidate);

        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertTrue(Files.exists(bundle.manifestFile()));
        assertEquals(false, bundle.passed());
        assertEquals(64, bundle.jsonSha256().length());
        assertEquals(64, bundle.markdownSha256().length());
        assertEquals(64, bundle.junitXmlSha256().length());
        assertEquals(64, bundle.manifestSha256().length());
        assertTrue(bundle.artifact().hasManifest());
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertTrue(Files.readString(bundle.jsonFile()).contains("\"runtime-regression-accounted-wall-time\""));
        assertTrue(Files.readString(bundle.markdownFile()).contains("# Runtime Regression Gate"));
        assertTrue(Files.readString(bundle.junitXmlFile()).contains("gollek.training.runtime.regression"));
        String manifest = Files.readString(bundle.manifestFile());
        assertTrue(manifest.contains("\"schema\":\"gollek.training.runtime.regression.gate.artifacts.v1\""));
        assertTrue(manifest.contains("\"jsonSha256\":\"" + bundle.jsonSha256() + "\""));
        assertTrue(manifest.contains("\"junitXmlSha256\":\"" + bundle.junitXmlSha256() + "\""));
        assertTrue(manifest.contains("\"manifestFile\":\"" + bundle.manifestFile() + "\""));
        assertEquals("gollek.training.runtime.regression.gate.artifacts.v1",
                bundle.artifactManifest().get("schema"));
        assertEquals(true, bundle.artifactManifest().get("regressed"));

        TrainingReportRuntimeRegressionGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportRuntimeRegressionGateArtifacts(tempDir);
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertEquals(bundle.manifestSha256(), inspection.manifestSha256());
        assertTrue(inspection.hasManifest());
        assertEquals(bundle.markdownSha256(), inspection.manifest().get("markdownSha256"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertEquals(false, inspection.passed());

        TrainingReportRuntimeRegressionGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportRuntimeRegressionGateArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertTrue(verification.manifestMatchesFiles());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertTrue(verification.artifact().hasManifest());
        assertEquals(bundle.manifestFile().toString(), verification.artifactMap().get("manifestFile"));
        assertEquals(bundle.manifestSha256(), verification.artifactMap().get("manifestSha256"));
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        verification.requirePassed();

        Files.writeString(bundle.markdownFile(), bundle.result().markdown() + "\n# tampered\n");
        TrainingReportRuntimeRegressionGateArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportRuntimeRegressionGateArtifacts(
                        TrainingReportRuntimeRegressionGateArtifacts.read(tempDir),
                        bundle.jsonSha256(),
                        null,
                        bundle.junitXmlSha256());
        assertEquals(false, tampered.passed());
        assertTrue(tampered.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown report does not match JSON")));
        assertTrue(tampered.failures().stream()
                .anyMatch(failure -> failure.contains("Manifest does not match artifact files")));

        TrainingReportRuntimeRegressionGateArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportRuntimeRegressionGateArtifacts(tempDir);
        assertTrue(Gollek.DL.verifyTrainingReportRuntimeRegressionGateArtifacts(refreshed).passed());
        assertEquals(bundle.result().toMap(), refreshed.result().toMap());
        assertEquals(bundle.jsonSha256(), refreshed.jsonSha256());
        assertEquals(refreshed.markdownSha256(),
                Gollek.DL.readTrainingReportRuntimeRegressionGateArtifacts(tempDir)
                        .manifest()
                        .get("markdownSha256"));
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
