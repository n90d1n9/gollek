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

class TrainingReportQualityProfilePromotionGateTest {
    @TempDir
    Path tempDir;

    @Test
    void runsProfileAwarePromotionGateWithoutManualPolicyWiring() throws IOException {
        Map<String, Path> reports = reportFiles(warningDataHealthMetadata());

        TrainingReportQualityProfilePromotionGate.Result research =
                Gollek.DL.runTrainingReportQualityProfilePromotionGate(
                        reports,
                        "baseline",
                        "research",
                        tempDir.resolve("research-gate"));
        TrainingReportQualityProfilePromotionGate.Result production =
                Gollek.DL.runTrainingReportQualityProfilePromotionGate(
                        reports,
                        "baseline",
                        TrainingReportQualityProfile.productionPromotion(),
                        tempDir.resolve("production-gate"));

        assertEquals(TrainingReportQualityProfile.RESEARCH, research.profile().id());
        assertTrue(research.passed());
        assertTrue(research.promotable());
        assertEquals("candidate", research.decision().candidate().orElseThrow().name());
        assertTrue(research.sourceVerification().passed());
        assertTrue(Files.isRegularFile(research.artifacts().jsonFile()));

        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, production.profile().id());
        assertFalse(production.passed());
        assertFalse(production.promotable());
        assertTrue(production.message().contains("Profile `production-promotion`"));
        assertTrue(production.decision().reasons().stream()
                .anyMatch(reason -> reason.contains("data health is not clean: warning")));
        assertTrue(production.verification().passed());
        assertTrue(production.sourceVerification().passed());
        assertThrows(IllegalStateException.class, production::requirePassed);

        Map<?, ?> profileMap = (Map<?, ?>) production.toMap().get("profile");
        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, profileMap.get("id"));
        assertTrue(Gollek.DL.trainingReportQualityProfilePromotionGateMarkdown(production)
                .contains("# Training Report Quality Profile Gate"));
    }

    @Test
    void runsCustomCatalogProfilePromotionGateFromJsonFile() throws IOException {
        Map<String, Path> reports = reportFiles(warningDataHealthMetadata());
        TrainingReportQualityProfile custom = new TrainingReportQualityProfile(
                "Catalog Research Promotion",
                "Catalog Research Promotion",
                "Custom catalog promotion policy that tolerates warning data health.",
                TrainingReportValidationPolicy.permissive(),
                TrainingReportPerformanceGate.Policy.permissive(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withMaxComparisonFindingSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withRequireTrackedMetricImprovement(false)
                        .withRequireCandidateDataHealthGate(false)
                        .withRequireCandidateDataHealthClean(false));
        TrainingReportQualityProfileArtifacts.ArtifactBundle catalog =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(
                        tempDir.resolve("promotion-catalog"),
                        List.of(custom));

        TrainingReportQualityProfilePromotionGate.Result result =
                Gollek.DL.runTrainingReportQualityProfilePromotionGate(
                        reports,
                        "baseline",
                        catalog.jsonFile(),
                        "CATALOG_RESEARCH_PROMOTION",
                        tempDir.resolve("catalog-promotion-gate"));

        assertEquals("catalog-research-promotion", result.profile().id());
        assertTrue(result.passed());
        assertTrue(result.promotable());
        assertEquals("candidate", result.decision().candidate().orElseThrow().name());
        assertTrue(result.verification().passed());
        assertTrue(result.sourceVerification().passed());
    }

    @Test
    void requestDefaultsToProductionPromotionProfile() throws IOException {
        Map<String, Path> reports = reportFiles(warningDataHealthMetadata());

        TrainingReportQualityProfilePromotionGate.Result result =
                Gollek.DL.runTrainingReportQualityProfilePromotionGate(
                        new TrainingReportQualityProfilePromotionGate.Request(
                                reports,
                                "baseline",
                                null,
                                tempDir.resolve("default-profile-gate"),
                                null));

        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, result.profile().id());
        assertFalse(result.passed());
        assertTrue(result.artifacts().jsonFile().endsWith("promotion-review.json"));
        assertTrue(result.toMap().containsKey("gate"));
    }

    @Test
    void writesVerifiesAndRefreshesProfileGateArtifacts() throws IOException {
        TrainingReportQualityProfilePromotionGate.Result result =
                Gollek.DL.runTrainingReportQualityProfilePromotionGate(
                        reportFiles(warningDataHealthMetadata()),
                        "baseline",
                        "production-promotion",
                        tempDir.resolve("gate"));
        Path artifacts = tempDir.resolve("profile-gate-artifacts");

        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfilePromotionGateArtifacts(artifacts, result);

        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, bundle.profileId());
        assertFalse(bundle.passed());
        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));

        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfilePromotionGateArtifacts(artifacts);
        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, inspection.profileId().orElseThrow());
        assertEquals(TrainingReportPortfolio.PromotionStatus.HOLD.name(), inspection.decisionStatus());
        assertTrue(inspection.markdown().contains("# Gollek Training Quality Profile Gate"));

        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfilePromotionGateArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.profileKnown());
        assertTrue(verification.gatePayloadConsistent());
        assertTrue(verification.markdownMatchesJson());
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered profile gate summary.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportQualityProfilePromotionGateArtifacts(bundle);
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactInspection refreshed =
                Gollek.DL.refreshTrainingReportQualityProfilePromotionGateArtifacts(artifacts);
        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification refreshedVerification =
                TrainingReportQualityProfilePromotionGateArtifacts.verify(
                        refreshed,
                        bundle.jsonSha256(),
                        refreshed.markdownSha256());
        assertTrue(refreshedVerification.passed());
    }

    private Map<String, Path> reportFiles(Map<String, Object> candidateMetadata) throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                Map.of(),
                120L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                candidateMetadata,
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        return reports;
    }

    private Path writeReport(
            String fileName,
            Map<String, Object> metadata,
            long durationMs,
            List<Map<String, Object>> epochHistory) throws IOException {
        Map<String, Object> reportMetadata = new LinkedHashMap<>();
        reportMetadata.put("epochHistory", epochHistory);
        reportMetadata.putAll(metadata);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestLoss(epochHistory, "validationLoss"),
                bestEpoch(epochHistory, "validationLoss"),
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                durationMs,
                reportMetadata);
        Path reportFile = tempDir.resolve(fileName);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-30T01:02:03Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
    }

    private static double latestLoss(List<Map<String, Object>> epochHistory, String key) {
        for (int index = epochHistory.size() - 1; index >= 0; index--) {
            Object value = epochHistory.get(index).get(key);
            if (value instanceof Number number) {
                return number.doubleValue();
            }
        }
        return Double.NaN;
    }

    private static double bestLoss(List<Map<String, Object>> epochHistory, String key) {
        double best = Double.POSITIVE_INFINITY;
        for (Map<String, Object> row : epochHistory) {
            Object value = row.get(key);
            if (value instanceof Number number && number.doubleValue() < best) {
                best = number.doubleValue();
            }
        }
        return Double.isFinite(best) ? best : Double.NaN;
    }

    private static int bestEpoch(List<Map<String, Object>> epochHistory, String key) {
        double best = Double.POSITIVE_INFINITY;
        int bestEpoch = -1;
        for (Map<String, Object> row : epochHistory) {
            Object value = row.get(key);
            if (value instanceof Number number && number.doubleValue() < best) {
                best = number.doubleValue();
                Object epoch = row.get("epoch");
                bestEpoch = epoch instanceof Number epochNumber ? epochNumber.intValue() : bestEpoch + 1;
            }
        }
        return bestEpoch;
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
