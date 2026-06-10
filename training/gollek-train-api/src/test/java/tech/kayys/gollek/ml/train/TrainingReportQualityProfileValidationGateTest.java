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
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportQualityProfileValidationGateTest {
    @TempDir
    Path tempDir;

    @Test
    void runsProfileAwareValidationGateWithoutManualPolicyWiring() throws IOException {
        Path reportFile = writeReport("report.json");

        TrainingReportQualityProfileValidationGate.Result strict =
                Gollek.DL.runTrainingReportQualityProfileValidationGate(
                        reportFile,
                        "strict-ci",
                        tempDir.resolve("strict-validation"));
        TrainingReportQualityProfileValidationGate.Result local =
                Gollek.DL.runTrainingReportQualityProfileValidationGate(
                        reportFile,
                        TrainingReportQualityProfile.localExperiment(),
                        tempDir.resolve("local-validation"));

        assertEquals(TrainingReportQualityProfile.STRICT_CI, strict.profile().id());
        assertFalse(strict.passed());
        assertFalse(strict.validationPassed());
        assertTrue(strict.validation().failureCodes().contains("data_health.missing"));
        assertTrue(strict.verification().passed());
        assertTrue(Files.isRegularFile(strict.artifacts().jsonFile()));
        assertTrue(Files.isRegularFile(strict.artifacts().markdownFile()));
        assertTrue(Files.isRegularFile(strict.artifacts().junitXmlFile()));
        assertTrue(strict.message().contains("Profile `strict-ci` validation gate failed"));
        assertThrows(IllegalStateException.class, strict::requirePassed);

        assertEquals(TrainingReportQualityProfile.LOCAL_EXPERIMENT, local.profile().id());
        assertTrue(local.passed());
        assertTrue(local.validationPassed());
        assertTrue(local.verification().passed());
        assertDoesNotThrow(local::requirePassed);
        assertTrue(Gollek.DL.trainingReportQualityProfileValidationGateMarkdown(local)
                .contains("# Training Report Quality Profile Validation Gate"));
        assertEquals(Boolean.TRUE, local.toMap().get("passed"));
    }

    @Test
    void requestDefaultsToStrictCiProfile() throws IOException {
        Path reportFile = writeReport("default-report.json");

        TrainingReportQualityProfileValidationGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileValidationGate(
                        new TrainingReportQualityProfileValidationGate.Request(
                                reportFile,
                                null,
                                tempDir.resolve("default-validation"),
                                null));

        assertEquals(TrainingReportQualityProfile.STRICT_CI, result.profile().id());
        assertFalse(result.passed());
        assertTrue(result.artifacts().jsonFile().endsWith("training-report-validation.json"));
        assertTrue(result.toMap().containsKey("verification"));
    }

    @Test
    void writesVerifiesAndRefreshesProfileValidationGateArtifacts() throws IOException {
        TrainingReportQualityProfileValidationGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileValidationGate(
                        writeReport("artifact-report.json"),
                        "local-experiment",
                        tempDir.resolve("validation-gate"));
        Path artifacts = tempDir.resolve("profile-validation-gate-artifacts");

        TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileValidationGateArtifacts(artifacts, result);

        assertEquals(TrainingReportQualityProfile.LOCAL_EXPERIMENT, bundle.profileId());
        assertTrue(bundle.passed());
        assertTrue(bundle.validationPassed());
        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));

        TrainingReportQualityProfileValidationGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileValidationGateArtifacts(artifacts);
        assertEquals(TrainingReportQualityProfile.LOCAL_EXPERIMENT, inspection.profileId().orElseThrow());
        assertTrue(inspection.passed());
        assertTrue(inspection.validationPassed());
        assertTrue(inspection.failureCodes().isEmpty());
        assertTrue(inspection.markdown().contains("# Gollek Training Quality Profile Validation Gate"));

        TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileValidationGateArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.profileKnown());
        assertTrue(verification.validationPayloadConsistent());
        assertTrue(verification.markdownMatchesJson());
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered profile validation gate summary.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportQualityProfileValidationGateArtifacts(bundle);
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportQualityProfileValidationGateArtifacts.ArtifactInspection refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileValidationGateArtifacts(artifacts);
        TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification refreshedVerification =
                TrainingReportQualityProfileValidationGateArtifacts.verify(
                        refreshed,
                        bundle.jsonSha256(),
                        refreshed.markdownSha256());
        assertTrue(refreshedVerification.passed());
    }

    private Path writeReport(String fileName) throws IOException {
        List<Map<String, Object>> epochHistory = List.of(
                Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7));
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                0.7,
                1,
                0.6,
                0.7,
                120L,
                Map.of("epochHistory", epochHistory));
        Path reportFile = tempDir.resolve(fileName);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-30T01:02:03Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
    }
}
