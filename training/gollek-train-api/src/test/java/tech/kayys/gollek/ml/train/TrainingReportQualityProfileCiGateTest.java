package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;

class TrainingReportQualityProfileCiGateTest {
    @TempDir
    Path tempDir;

    @Test
    void runsValidationAndPromotionAsOneProfileCiGate() throws IOException {
        Map<String, Path> reports = reportFiles(Map.of(), Map.of());

        TrainingReportQualityProfileCiGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileCiGate(
                        reports,
                        "baseline",
                        "local-experiment",
                        tempDir.resolve("local-ci"));

        assertEquals(TrainingReportQualityProfile.LOCAL_EXPERIMENT, result.profile().id());
        assertTrue(result.passed());
        assertTrue(result.validationPassed());
        assertTrue(result.promotionPassed());
        assertTrue(result.artifactsVerified());
        assertTrue(result.failedValidationNames().isEmpty());
        assertEquals("candidate", result.promotion().decision().candidate().orElseThrow().name());
        assertEquals(2, result.validations().size());
        assertTrue(Files.isRegularFile(result.validationArtifacts().get("baseline").jsonFile()));
        assertTrue(Files.isRegularFile(result.validationArtifacts().get("candidate").markdownFile()));
        assertTrue(Files.isRegularFile(result.promotionArtifacts().jsonFile()));
        assertTrue(Gollek.DL.trainingReportQualityProfileCiGateMarkdown(result)
                .contains("# Gollek Training Quality Profile CI Gate"));
        assertDoesNotThrow(result::requirePassed);
        assertEquals(Boolean.TRUE, result.toMap().get("passed"));
    }

    @Test
    void productionCiGateHoldsPromotionWhenCandidateDataHealthIsWarning() throws IOException {
        Map<String, Path> reports = reportFiles(
                TrainingReportQualityProfileTestFixtures.cleanDataHealthMetadata(),
                TrainingReportQualityProfileTestFixtures.warningDataHealthMetadata());

        TrainingReportQualityProfileCiGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileCiGate(
                        reports,
                        "baseline",
                        TrainingReportQualityProfile.productionPromotion(),
                        tempDir.resolve("production-ci"));

        assertEquals(TrainingReportQualityProfile.PRODUCTION_PROMOTION, result.profile().id());
        assertFalse(result.passed());
        assertFalse(result.validationPassed());
        assertFalse(result.promotionPassed());
        assertTrue(result.artifactsVerified());
        assertEquals(List.of("candidate"), result.failedValidationNames());
        assertTrue(result.promotion().decision().reasons().stream()
                .anyMatch(reason -> reason.contains("data health is not clean: warning")));
        assertTrue(result.message().contains("Failed validations"));
        assertTrue(result.message().contains("Promotion failed"));
        assertThrows(IllegalStateException.class, result::requirePassed);
    }

    @Test
    void runsCustomCatalogProfileCiGateFromJsonFile() throws IOException {
        Map<String, Path> reports = reportFiles(
                TrainingReportQualityProfileTestFixtures.cleanDataHealthMetadata(),
                TrainingReportQualityProfileTestFixtures.warningDataHealthMetadata());
        TrainingReportQualityProfile custom = new TrainingReportQualityProfile(
                "Catalog CI Research",
                "Catalog CI Research",
                "Custom CI profile that validates and promotes warning data-health candidates.",
                TrainingReportValidationPolicy.permissive()
                        .withRequireDataHealthGate(false),
                TrainingReportPerformanceGate.Policy.permissive(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withMaxComparisonFindingSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withRequireTrackedMetricImprovement(false)
                        .withRequireCandidateDataHealthGate(false)
                        .withRequireCandidateDataHealthClean(false));
        TrainingReportQualityProfileArtifacts.ArtifactBundle catalog =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(
                        tempDir.resolve("ci-catalog"),
                        List.of(custom));

        TrainingReportQualityProfileCiGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileCiGate(
                        reports,
                        "baseline",
                        catalog.jsonFile(),
                        "CATALOG_CI_RESEARCH",
                        tempDir.resolve("catalog-ci"));

        assertEquals("catalog-ci-research", result.profile().id());
        assertTrue(result.passed());
        assertTrue(result.validationPassed());
        assertTrue(result.promotionPassed());
        assertTrue(result.artifactsVerified());
        assertEquals("candidate", result.promotion().decision().candidate().orElseThrow().name());
        assertDoesNotThrow(result::requirePassed);
    }

    @Test
    void rejectsUnknownBaselineBeforeWritingArtifacts() throws IOException {
        Map<String, Path> reports = reportFiles(Map.of(), Map.of());

        assertThrows(
                IllegalArgumentException.class,
                () -> Gollek.DL.runTrainingReportQualityProfileCiGate(
                        reports,
                        "missing-baseline",
                        "local-experiment",
                        tempDir.resolve("bad-ci")));
    }

    private Map<String, Path> reportFiles(
            Map<String, Object> baselineMetadata,
            Map<String, Object> candidateMetadata) throws IOException {
        return TrainingReportQualityProfileTestFixtures.reportFiles(
                tempDir,
                baselineMetadata,
                candidateMetadata);
    }
}
