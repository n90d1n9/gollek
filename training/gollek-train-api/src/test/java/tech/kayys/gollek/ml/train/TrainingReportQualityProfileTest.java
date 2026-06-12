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
                ((Map<?, ?>) map.get("performancePolicy")).get("failOnAcceleratorFallback"));
        assertEquals(
                Boolean.TRUE,
                ((Map<?, ?>) map.get("promotionPolicy")).get("requireCandidateDataHealthClean"));
    }

    @Test
    void profilesCarryPerformancePolicies() {
        TrainingReport fallbackReport = validatedReport(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", false),
                Map.entry("executionFallback", true),
                Map.entry("trainBatchCount", 2L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainAverageBatchMillis", 500.0)));

        assertTrue(TrainingReportQualityProfile.localExperiment().performanceGate(fallbackReport).passed());
        assertFalse(TrainingReportQualityProfile.strictCi().performanceGate(fallbackReport).passed());
        assertFalse(TrainingReportQualityProfile.productionPromotion().performanceGate(fallbackReport).passed());
        assertFalse(TrainingReportQualityProfile.localExperiment()
                .performancePolicy()
                .failOnAcceleratorFallback());
        assertTrue(TrainingReportQualityProfile.strictCi()
                .performancePolicy()
                .failOnAcceleratorFallback());
    }

    @Test
    void catalogExportsProfileMarkdownAndJsonForCiDocs() {
        TrainingReportQualityProfileCatalog catalog = Gollek.DL.trainingReportQualityProfileCatalog();

        assertEquals(TrainingReportQualityProfile.ids(), catalog.ids());
        assertEquals(catalog.toMarkdown(), Gollek.DL.trainingReportQualityProfilesMarkdown());

        String markdown = Gollek.DL.trainingReportQualityProfilesMarkdown();
        assertTrue(markdown.contains("# Gollek Training Report Quality Profiles"));
        assertTrue(markdown.contains("`production-promotion`"));
        assertTrue(markdown.contains("accelerator fallback fails `true`"));
        assertTrue(markdown.contains("clean data health `true`"));

        String customMarkdown = Gollek.DL.trainingReportQualityProfilesMarkdown(
                List.of(TrainingReportQualityProfile.localExperiment()));
        assertTrue(customMarkdown.contains("`local-experiment`"));
        assertFalse(customMarkdown.contains("`production-promotion`"));

        String json = Gollek.DL.trainingReportQualityProfilesJson();
        assertTrue(json.contains("\"format\":\"gollek.training-report.quality-profiles.v1\""));
        assertTrue(json.contains("\"profileCount\":4"));
        assertTrue(json.contains("\"performancePolicy\""));
        assertTrue(json.contains("\"requireCandidateDataHealthClean\":true"));
    }

    @Test
    void catalogRoundTripsCustomProfilePolicies() {
        TrainingReportQualityProfile custom = new TrainingReportQualityProfile(
                "Gpu Perf Lab",
                "GPU Perf Lab",
                "Strict accelerator placement with relaxed promotion for local benchmark sweeps.",
                TrainingReportValidationPolicy.permissive()
                        .withRequireValidation(false),
                new TrainingReportPerformanceGate.Policy(true, 42.0, 1.25),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withMinimumValidationImprovement(0.05)
                        .withRequireTrackedMetricImprovement(false));
        TrainingReportQualityProfileCatalog roundTripped =
                TrainingReportQualityProfileCatalog.fromMap(
                        new TrainingReportQualityProfileCatalog(List.of(custom)).toMap());
        TrainingReportQualityProfile restored = roundTripped.profiles().getFirst();

        assertEquals("gpu-perf-lab", restored.id());
        assertFalse(restored.validationPolicy().requireValidation());
        assertEquals(42.0, restored.performancePolicy().minTrainSamplesPerSecond());
        assertEquals(1.25, restored.performancePolicy().maxValidationToTrainAverageBatchMillisRatio());
        assertEquals(
                TrainingReportDiagnostics.Severity.WARNING,
                restored.promotionPolicy().maxCandidateDiagnosticSeverity());
        assertEquals(0.05, restored.promotionPolicy().minimumValidationImprovement());
        assertFalse(restored.promotionPolicy().requireTrackedMetricImprovement());
    }

    @Test
    void sdkFacadeLoadsCustomProfileCatalogMaps() {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("id", "Sdk GPU Lab");
        profile.put("displayName", "SDK GPU Lab");
        profile.put("description", "Custom profile loaded through the public DL facade.");
        profile.put("validationPolicy", Map.of(
                "maxDiagnosticSeverity", "WARNING",
                "requireRunHealthGate", false,
                "requireDataHealthGate", false,
                "requireDataHealthAvailable", false,
                "requireFreshDiagnostics", false,
                "requireValidation", false,
                "requireCheckpointIntegrity", false));
        profile.put("performancePolicy", Map.of(
                "failOnAcceleratorFallback", true,
                "minTrainSamplesPerSecond", 64.0,
                "maxValidationToTrainAverageBatchMillisRatio", 1.1));
        profile.put("promotionPolicy", Map.of(
                "maxCandidateDiagnosticSeverity", "WARNING",
                "maxComparisonFindingSeverity", "INFO",
                "minimumValidationImprovement", 0.02,
                "requireTrackedMetricImprovement", false,
                "requireCandidateDataHealthAvailable", false,
                "requireCandidateDataHealthGate", false,
                "requireCandidateDataHealthClean", false));

        TrainingReportQualityProfile loaded = Gollek.DL.trainingReportQualityProfile(profile);
        TrainingReportQualityProfileCatalog catalog = Gollek.DL.trainingReportQualityProfileCatalog(Map.of(
                "format", TrainingReportQualityProfileCatalog.FORMAT,
                "profileCount", 1,
                "profiles", List.of(profile)));

        assertEquals("sdk-gpu-lab", loaded.id());
        assertEquals(64.0, loaded.performancePolicy().minTrainSamplesPerSecond());
        assertEquals(List.of("sdk-gpu-lab"), catalog.ids());
        assertTrue(Gollek.DL.trainingReportQualityProfilesJson(catalog).contains("\"minTrainSamplesPerSecond\":64.0"));
        assertTrue(Gollek.DL.trainingReportQualityProfilesMarkdown(catalog)
                .contains("min train samples/s `64`"));
    }

    @Test
    void sdkFacadeLoadsAndSelectsCustomProfileCatalogFiles() throws IOException {
        TrainingReportQualityProfile custom = new TrainingReportQualityProfile(
                "File GPU Lab",
                "File GPU Lab",
                "Custom profile loaded from a persisted catalog artifact.",
                TrainingReportValidationPolicy.permissive()
                        .withRequireValidation(false),
                new TrainingReportPerformanceGate.Policy(true, 128.0, 1.2),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withRequireTrackedMetricImprovement(false));
        TrainingReportQualityProfileArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(
                        tempDir.resolve("file-catalog"),
                        List.of(custom));

        TrainingReportQualityProfileCatalog loaded =
                Gollek.DL.trainingReportQualityProfileCatalog(bundle.jsonFile());
        TrainingReportQualityProfile selected =
                Gollek.DL.trainingReportQualityProfile(bundle.jsonFile(), "FILE_GPU_LAB");

        assertEquals(List.of("file-gpu-lab"), loaded.ids());
        assertEquals("file-gpu-lab", loaded.require("file gpu lab").id());
        assertEquals("file-gpu-lab", selected.id());
        assertEquals(128.0, selected.performancePolicy().minTrainSamplesPerSecond());
        assertFalse(selected.validationPolicy().requireValidation());
        assertFalse(selected.promotionPolicy().requireTrackedMetricImprovement());
    }

    @Test
    void catalogValidatorReportsDuplicateIdsAndUnknownKeysBeforeGateExecution() {
        Map<String, Object> duplicateOne = customProfileMap("GPU Lab");
        Map<String, Object> duplicateTwo = customProfileMap("gpu_lab");
        duplicateTwo.put("futurePolicy", Map.of("enabled", true));
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 2);
        catalog.put("profiles", List.of(duplicateOne, duplicateTwo));
        catalog.put("unexpectedCatalogKey", true);

        TrainingReportQualityProfileCatalogValidator.Result validation =
                Gollek.DL.validateTrainingReportQualityProfileCatalog(catalog);

        assertFalse(validation.passed());
        assertEquals(List.of("gpu-lab", "gpu-lab"), validation.profileIds());
        assertTrue(validation.errors().stream()
                .anyMatch(issue -> issue.code().equals("profile.id_duplicate")));
        assertTrue(validation.warnings().stream()
                .anyMatch(issue -> issue.path().contains("unexpectedCatalogKey")));
        assertTrue(validation.markdown().contains("profile.id_duplicate"));
        assertTrue(Gollek.DL.trainingReportQualityProfileCatalogValidationMarkdown(validation)
                .contains("FAIL"));
        assertThrows(IllegalArgumentException.class, () -> TrainingReportQualityProfileCatalog.fromMap(catalog));
    }

    @Test
    void catalogValidatorWarnsWhenPoliciesAreImplicitButStillPasses() {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 1);
        catalog.put("profiles", List.of(Map.of(
                "id", "Minimal Lab",
                "displayName", "Minimal Lab",
                "description", "Uses Gollek defaults while still documenting a custom workflow.")));

        TrainingReportQualityProfileCatalogValidator.Result validation =
                Gollek.DL.validateTrainingReportQualityProfileCatalog(catalog);

        assertTrue(validation.passed());
        assertEquals(List.of("minimal-lab"), validation.profileIds());
        assertTrue(validation.warnings().stream()
                .anyMatch(issue -> issue.code().equals("profile.policy_missing")));
        assertTrue(validation.markdown().contains("PASS"));
        assertEquals(List.of("minimal-lab"), Gollek.DL.trainingReportQualityProfileCatalog(catalog).ids());
    }

    @Test
    void catalogValidatorHandlesInvalidJsonFilesWithoutThrowing(@TempDir Path tempDir) throws IOException {
        Path invalid = tempDir.resolve("quality-profiles.json");
        Files.writeString(invalid, "{ invalid json", StandardCharsets.UTF_8);

        TrainingReportQualityProfileCatalogValidator.Result validation =
                Gollek.DL.validateTrainingReportQualityProfileCatalog(invalid);

        assertFalse(validation.passed());
        assertFalse(validation.validJson());
        assertTrue(validation.errors().stream()
                .anyMatch(issue -> issue.code().equals("catalog.json_invalid")));
        assertTrue(validation.markdown().contains("catalog.json_invalid"));
    }

    @Test
    void writesVerifiesAndRefreshesCatalogValidationArtifacts() throws IOException {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 1);
        catalog.put("profiles", List.of(customProfileMap("Artifact GPU Lab")));
        TrainingReportQualityProfileCatalogValidator.Result validation =
                Gollek.DL.validateTrainingReportQualityProfileCatalog(catalog);

        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileCatalogValidationArtifacts(
                        tempDir.resolve("catalog-validation"),
                        validation);
        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileCatalogValidationArtifacts(bundle.directory());
        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogValidationArtifacts(bundle);

        assertTrue(bundle.passed());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(validation.toMap(), inspection.result());
        assertEquals(List.of("artifact-gpu-lab"), inspection.parsedResult().profileIds());
        assertTrue(inspection.junitXml().contains("gollek.training.quality-profile-catalog.validation"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertTrue(Gollek.DL.trainingReportQualityProfileCatalogValidationJUnitXml(validation)
                .contains("failures=\"0\""));
        assertTrue(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered catalog validation report.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogValidationArtifacts(
                        bundle.directory(),
                        bundle.jsonSha256(),
                        bundle.markdownSha256());
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        Files.writeString(
                bundle.junitXmlFile(),
                "\n<!-- tampered catalog validation junit -->\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactVerification junitTampered =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogValidationArtifacts(
                        Gollek.DL.readTrainingReportQualityProfileCatalogValidationArtifacts(bundle.directory()),
                        bundle.jsonSha256(),
                        null,
                        bundle.junitXmlSha256());
        assertFalse(junitTampered.passed());
        assertFalse(junitTampered.junitXmlSha256Matches());
        assertFalse(junitTampered.junitXmlMatchesJson());

        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileCatalogValidationArtifacts(bundle.directory());
        TrainingReportQualityProfileCatalogValidationArtifacts.ArtifactVerification refreshedVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogValidationArtifacts(refreshed);
        assertTrue(refreshedVerification.passed());
    }

    @Test
    void catalogAdvisorKeepsBuiltInProfilesQuiet() {
        TrainingReportQualityProfileCatalogAdvisor.Result advice =
                Gollek.DL.adviseTrainingReportQualityProfileCatalog(
                        TrainingReportQualityProfileCatalog.defaults());

        assertTrue(advice.validation().passed());
        assertTrue(advice.readyForCi());
        assertTrue(advice.recommendations().isEmpty());
        assertTrue(Gollek.DL.trainingReportQualityProfileCatalogAdviceMarkdown(advice)
                .contains("No catalog advisory recommendations."));
    }

    @Test
    void catalogAdvisorRecommendsWorkflowCoverageForMinimalCatalog() {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 1);
        catalog.put("profiles", List.of(Map.of(
                "id", "Minimal Lab",
                "displayName", "Minimal Lab",
                "description", "Only one profile with implicit defaults.")));

        TrainingReportQualityProfileCatalogAdvisor.Result advice =
                Gollek.DL.adviseTrainingReportQualityProfileCatalog(catalog);

        assertTrue(advice.validation().passed());
        assertTrue(advice.readyForCi());
        assertTrue(advice.recommendations().stream()
                .anyMatch(recommendation -> recommendation.diagnosticCode()
                        .equals("quality_profile_catalog.implicit_policy_defaults")));
        assertTrue(advice.recommendations().stream()
                .anyMatch(recommendation -> recommendation.diagnosticCode()
                        .equals("quality_profile_catalog.single_profile")));
        assertTrue(advice.recommendations().stream()
                .anyMatch(recommendation -> recommendation.diagnosticCode()
                        .equals("quality_profile_catalog.strict_accelerator_ci_missing")));
        assertTrue(advice.markdown().contains("Add a strict accelerator-aware CI profile"));
        assertEquals(advice.recommendations().size(), advice.toMap().get("recommendationCount"));
    }

    @Test
    void catalogAdvisorBlocksInvalidCatalogBeforeWorkflowAdvice() {
        Map<String, Object> first = customProfileMap("Duplicate Lab");
        Map<String, Object> second = customProfileMap("duplicate_lab");
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 2);
        catalog.put("profiles", List.of(first, second));

        TrainingReportQualityProfileCatalogAdvisor.Result advice =
                Gollek.DL.adviseTrainingReportQualityProfileCatalog(catalog);

        assertFalse(advice.validation().passed());
        assertFalse(advice.readyForCi());
        assertEquals(1, advice.recommendations().size());
        assertEquals(TrainingReportRecommendation.Priority.BLOCKER, advice.recommendations().getFirst().priority());
        assertEquals(
                "quality_profile_catalog.validation_failed",
                advice.recommendations().getFirst().diagnosticCode());
        assertTrue(advice.markdown().contains("Fix quality-profile catalog validation errors"));
    }

    @Test
    void writesVerifiesAndRefreshesCatalogAdviceArtifacts() throws IOException {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format", TrainingReportQualityProfileCatalog.FORMAT);
        catalog.put("profileCount", 1);
        catalog.put("profiles", List.of(Map.of(
                "id", "Advice Minimal Lab",
                "displayName", "Advice Minimal Lab",
                "description", "Minimal profile that should produce advisory recommendations.")));
        TrainingReportQualityProfileCatalogAdvisor.Result advice =
                Gollek.DL.adviseTrainingReportQualityProfileCatalog(catalog);

        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileCatalogAdviceArtifacts(
                        tempDir.resolve("catalog-advice"),
                        advice);
        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileCatalogAdviceArtifacts(bundle.directory());
        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogAdviceArtifacts(bundle);

        assertTrue(bundle.readyForCi());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(advice.readyForCi(), inspection.parsedResult().readyForCi());
        assertEquals(advice.validation().toMap(), inspection.parsedResult().validation().toMap());
        assertEquals(advice.recommendations().size(), inspection.parsedResult().recommendations().size());
        assertTrue(inspection.markdown().contains("Add a strict accelerator-aware CI profile"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertTrue(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered catalog advice report.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogAdviceArtifacts(
                        bundle.directory(),
                        bundle.jsonSha256(),
                        bundle.markdownSha256());
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileCatalogAdviceArtifacts(bundle.directory());
        TrainingReportQualityProfileCatalogAdviceArtifacts.ArtifactVerification refreshedVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCatalogAdviceArtifacts(refreshed);
        assertTrue(refreshedVerification.passed());
    }

    @Test
    void writesReadsVerifiesAndRefreshesQualityProfileArtifacts() throws IOException {
        Path artifacts = tempDir.resolve("quality-profiles");
        TrainingReportQualityProfileArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileArtifacts(artifacts);

        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertEquals(TrainingReportQualityProfile.ids(), bundle.catalog().ids());
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));

        TrainingReportQualityProfileArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileArtifacts(artifacts);
        assertEquals(TrainingReportQualityProfile.ids(), inspection.profileIds());
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertTrue(inspection.markdown().contains("# Gollek Training Report Quality Profiles"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());

        TrainingReportQualityProfileArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.jsonMatchesCatalog());
        assertTrue(verification.markdownMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
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

    private static Map<String, Object> customProfileMap(String id) {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("id", id);
        profile.put("displayName", id);
        profile.put("description", "Custom quality profile for test gates.");
        profile.put("validationPolicy", Map.of(
                "maxDiagnosticSeverity", "WARNING",
                "requireRunHealthGate", false,
                "requireDataHealthGate", false,
                "requireDataHealthAvailable", false,
                "requireFreshDiagnostics", false,
                "requireValidation", false,
                "requireCheckpointIntegrity", false));
        profile.put("performancePolicy", Map.of(
                "failOnAcceleratorFallback", true,
                "minTrainSamplesPerSecond", 32.0,
                "maxValidationToTrainAverageBatchMillisRatio", 1.5));
        profile.put("promotionPolicy", Map.of(
                "maxCandidateDiagnosticSeverity", "WARNING",
                "maxComparisonFindingSeverity", "WARNING",
                "minimumValidationImprovement", 0.0,
                "requireTrackedMetricImprovement", false,
                "requireCandidateDataHealthAvailable", false,
                "requireCandidateDataHealthGate", false,
                "requireCandidateDataHealthClean", false));
        return profile;
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
