package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportQualityProfileTest {
    @TempDir
    Path tempDir;

    @Test
    void exposesNamedProfilesWithSdkFacadeLookup() {
        assertEquals(
                List.of("local-experiment", "research", "strict-ci", "production-promotion"),
                TrainingReportQualityProfile.ids());
        assertEquals(TrainingReportQualityProfile.STRICT_CI, TrainingReportQualityProfile.require("STRICT_CI").id());
        assertEquals(TrainingReportQualityProfile.RESEARCH, Gollek.DL.trainingReportQualityProfile("research").id());
        assertEquals(
                TrainingReportQualityProfile.ids(),
                Gollek.DL.trainingReportQualityProfiles().stream()
                        .map(TrainingReportQualityProfile::id)
                        .toList());
        assertThrows(IllegalArgumentException.class, () -> TrainingReportQualityProfile.require("unknown"));
    }

    @Test
    void strictCiRequiresDataHealthEvidenceWhileLocalExperimentDoesNot() {
        TrainingReport report = validatedReport(Map.of());

        TrainingReportValidationPolicy.Result strict =
                TrainingReportQualityProfile.strictCi().validate(report);
        TrainingReportValidationPolicy.Result local =
                Gollek.DL.validateTrainingReport(report, TrainingReportQualityProfile.localExperiment());

        assertTrue(strict.failed());
        assertTrue(strict.failureCodes().contains("data_health.missing"));
        assertTrue(strict.policy().requireDataHealthAvailable());
        assertTrue(local.passed());
        assertFalse(local.policy().requireDataHealthAvailable());
        assertFalse(local.policy().requireDataHealthGate());
        assertTrue(Gollek.DL.trainingReportValidationMarkdown(report, TrainingReportQualityProfile.strictCi())
                .contains("data_health.missing"));
    }

    @Test
    void productionPromotionRequiresCleanCandidateDataHealth() {
        TrainingReport baseline = validatedReport(Map.of());
        TrainingReport candidate = betterValidatedReport(warningDataHealthMetadata());
        TrainingReportPortfolio portfolio = TrainingReportPortfolio.fromReports(Map.of(
                "baseline", baseline,
                "candidate", candidate));

        TrainingReportPortfolio.PromotionDecision research =
                TrainingReportQualityProfile.research().promotionDecision(portfolio, "baseline");
        TrainingReportPortfolio.PromotionDecision production =
                Gollek.DL.trainingReportQualityProfile("production_promotion")
                        .promotionDecision(portfolio, "baseline");

        assertEquals(TrainingReportPortfolio.PromotionStatus.PROMOTE, research.status());
        assertEquals(TrainingReportPortfolio.PromotionStatus.HOLD, production.status());
        assertTrue(production.policy().requireCandidateDataHealthClean());
        assertTrue(production.reasons().stream()
                .anyMatch(reason -> reason.contains("data health is not clean: warning")));
        assertEquals(
                Boolean.TRUE,
                ((Map<?, ?>) production.toMap().get("policy")).get("requireCandidateDataHealthClean"));
    }

    @Test
    void profileMapDocumentsValidationAndPromotionPolicies() {
        Map<String, Object> map = TrainingReportQualityProfile.productionPromotion().toMap();

        assertEquals("production-promotion", map.get("id"));
        assertTrue(map.get("description").toString().contains("Production promotion"));
        assertEquals(
                Boolean.TRUE,
                ((Map<?, ?>) map.get("validationPolicy")).get("requireDataHealthAvailable"));
        assertEquals(
                Boolean.TRUE,
                ((Map<?, ?>) map.get("promotionPolicy")).get("requireCandidateDataHealthClean"));
    }

    @Test
    void catalogExportsProfileMarkdownAndJsonForCiDocs() {
        TrainingReportQualityProfileCatalog catalog = Gollek.DL.trainingReportQualityProfileCatalog();

        assertEquals(TrainingReportQualityProfile.ids(), catalog.ids());
        assertEquals(catalog.toMarkdown(), Gollek.DL.trainingReportQualityProfilesMarkdown());

        String markdown = Gollek.DL.trainingReportQualityProfilesMarkdown();
        assertTrue(markdown.contains("# Gollek Training Report Quality Profiles"));
        assertTrue(markdown.contains("`production-promotion`"));
        assertTrue(markdown.contains("clean data health `true`"));

        String customMarkdown = Gollek.DL.trainingReportQualityProfilesMarkdown(
                List.of(TrainingReportQualityProfile.localExperiment()));
        assertTrue(customMarkdown.contains("`local-experiment`"));
        assertFalse(customMarkdown.contains("`production-promotion`"));

        String json = Gollek.DL.trainingReportQualityProfilesJson();
        assertTrue(json.contains("\"format\":\"gollek.training-report.quality-profiles.v1\""));
        assertTrue(json.contains("\"profileCount\":4"));
        assertTrue(json.contains("\"requireCandidateDataHealthClean\":true"));
    }

    @Test
    void writesReadsVerifiesAndRefreshesQualityProfileArtifacts() throws IOException {
        Path artifacts = tempDir.resolve("quality-profiles");
        TrainingReportQualityProfileArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(artifacts);

        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertEquals(TrainingReportQualityProfile.ids(), bundle.catalog().ids());

        TrainingReportQualityProfileArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileArtifacts(artifacts);
        assertEquals(TrainingReportQualityProfile.ids(), inspection.profileIds());
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertTrue(inspection.markdown().contains("# Gollek Training Report Quality Profiles"));

        TrainingReportQualityProfileArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.jsonMatchesCatalog());
        assertTrue(verification.markdownMatchesJson());
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered quality profile docs.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportQualityProfileArtifacts(bundle);
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertTrue(tampered.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown checksum mismatch")));
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportQualityProfileArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileArtifacts(artifacts);
        TrainingReportQualityProfileArtifacts.ArtifactVerification refreshedVerification =
                Gollek.DL.verifyTrainingReportQualityProfileArtifacts(refreshed);
        assertTrue(refreshedVerification.passed());

        TrainingReportQualityProfileArtifacts.ArtifactBundle custom =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(
                        tempDir.resolve("local-quality-profile"),
                        List.of(TrainingReportQualityProfile.localExperiment()));
        assertEquals(List.of(TrainingReportQualityProfile.LOCAL_EXPERIMENT), custom.catalog().ids());
        assertTrue(Files.readString(custom.markdownFile()).contains("`local-experiment`"));
        assertFalse(Files.readString(custom.markdownFile()).contains("`production-promotion`"));
    }

    private static TrainingReport validatedReport(Map<String, Object> extraMetadata) {
        return report(
                extraMetadata,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)),
                0.7,
                1,
                0.6,
                0.7,
                120L);
    }

    private static TrainingReport betterValidatedReport(Map<String, Object> extraMetadata) {
        return report(
                extraMetadata,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45)),
                0.45,
                1,
                0.4,
                0.45,
                90L);
    }

    private static TrainingReport report(
            Map<String, Object> extraMetadata,
            List<Map<String, Object>> epochHistory,
            double bestValidationLoss,
            int bestEpoch,
            double latestTrainLoss,
            double latestValidationLoss,
            long durationMs) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("epochHistory", epochHistory);
        metadata.putAll(extraMetadata);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestValidationLoss,
                bestEpoch,
                latestTrainLoss,
                latestValidationLoss,
                durationMs,
                metadata);
        return TrainingReport.of(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-30T01:02:03Z")));
    }

    private static Map<String, Object> warningDataHealthMetadata() {
        Map<String, Object> issue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "blocking", false,
                "message", "train loader dropLast discarded samples",
                "action", "adjust batch size or disable dropLast for small datasets");
        return Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "warning"),
                Map.entry("dataLoaderPlanHealthHealthy", false),
                Map.entry("dataLoaderPlanHealthGatePassed", true),
                Map.entry("dataLoaderPlanHealthIssueDetected", true),
                Map.entry("dataLoaderPlanHealthIssueCount", 1),
                Map.entry("dataLoaderPlanHealthWarningCount", 1),
                Map.entry("dataLoaderPlanHealthErrorCount", 0),
                Map.entry("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-drop-last-discarded-samples")),
                Map.entry("dataLoaderPlanHealthIssueSeverities", List.of("warning")),
                Map.entry("dataLoaderPlanHealthRecommendedActions", List.of(issue.get("action"))),
                Map.entry("dataLoaderPlanHealthIssues", List.of(issue)),
                Map.entry("dataDistributionHealth.available", true),
                Map.entry("dataDistributionHealthStatus", "healthy"),
                Map.entry("dataDistributionHealthHealthy", true),
                Map.entry("dataDistributionHealthGatePassed", true),
                Map.entry("dataDistributionHealthIssueDetected", false),
                Map.entry("dataDistributionHealthIssueCount", 0),
                Map.entry("dataDistributionHealthWarningCount", 0),
                Map.entry("dataDistributionHealthErrorCount", 0),
                Map.entry("dataDistributionHealthIssues", List.of()));
    }
}
