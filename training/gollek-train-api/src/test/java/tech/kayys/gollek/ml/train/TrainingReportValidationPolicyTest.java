package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportValidationPolicyTest {
    @TempDir
    Path tempDir;

    @Test
    void strictPolicyPassesFreshValidatedReportAndFacade() throws IOException {
        TrainingReport report = TrainingReport.of(validatedPayload(Map.of()));
        Path reportFile = tempDir.resolve("validated-report.json");
        Files.writeString(reportFile, TrainerJson.toJson(report.payload()), StandardCharsets.UTF_8);

        TrainingReportValidationPolicy.Result result = report.validate();
        TrainingReportValidationPolicy.Result facadeResult = Gollek.DL.validateTrainingReport(reportFile);

        assertTrue(result.passed());
        assertFalse(result.failed());
        assertTrue(result.validationAvailable());
        assertFalse(result.dataHealth().available());
        assertTrue(result.policy().requireDataHealthGate());
        assertFalse(result.policy().requireDataHealthAvailable());
        assertEquals("FRESH", result.diagnosticProvenance().status());
        assertEquals(List.of(), result.failureCodes());
        assertEquals(Boolean.TRUE, result.toMap().get("passed"));
        assertDoesNotThrow(result::requirePassed);
        assertTrue(facadeResult.passed());

        String markdown = result.markdown();
        String xml = result.junitXml();

        assertEquals(markdown, Gollek.DL.trainingReportValidationMarkdown(reportFile));
        assertEquals(xml, Gollek.DL.trainingReportValidationJUnitXml(reportFile));
        assertTrue(markdown.contains("# Gollek Training Report Validation"));
        assertTrue(markdown.contains("**Validation:** `PASS`"));
        assertTrue(markdown.contains("**Data health:** `not-recorded`"));
        assertTrue(markdown.contains("**Diagnostics provenance:** `FRESH`"));
        assertTrue(markdown.contains("`requireDataHealthGate` | `true`"));
        assertTrue(markdown.contains("`requireDataHealthAvailable` | `false`"));
        assertWellFormedXml(xml);
        assertTrue(xml.contains("<testsuite name=\"gollek.training.report.validation\" tests=\"1\" failures=\"0\""));
        assertTrue(xml.contains("name=\"validation.passed\" value=\"true\""));
        assertTrue(xml.contains("name=\"policy.requireDataHealthGate\" value=\"true\""));
        assertTrue(xml.contains("name=\"policy.requireDataHealthAvailable\" value=\"false\""));
        assertTrue(xml.contains("name=\"dataHealth.available\" value=\"false\""));
        assertTrue(xml.contains("<system-out># Gollek Training Report Validation"));
        assertFalse(xml.contains("<failure"));
    }

    @Test
    void strictPolicyBlocksBackfilledRunHealthAndStaleDiagnostics() {
        TrainingReport report = TrainingReport.of(staleFailedRunHealthReport());

        TrainingReportValidationPolicy.Result result = report.validate();

        assertTrue(result.failed());
        assertTrue(result.failureCodes().contains("diagnostics.exceeded"));
        assertTrue(result.failureCodes().contains("run_health.gate_failed"));
        assertTrue(result.failureCodes().contains("diagnostics.not_fresh"));
        assertTrue(result.failureCodes().contains("validation.missing"));
        assertEquals("BACKFILLED", result.diagnosticProvenance().status());
        assertFalse(result.validationAvailable());
        assertThrows(IllegalStateException.class, result::requirePassed);
        assertTrue(result.message().contains("Training report validation failed"));
        assertEquals(Boolean.FALSE, result.toMap().get("passed"));
    }

    @Test
    void strictPolicyBlocksCheckpointIntegrityMetadata() {
        TrainingReport report = TrainingReport.of(validatedPayload(Map.of(
                "checkpointManifestIntegrityMismatch",
                true)));

        TrainingReportValidationPolicy.Result result = report.validate();

        assertTrue(result.failed());
        assertTrue(result.failureCodes().contains("checkpoint_integrity.failed"));
        assertTrue(result.checkpointIntegrityAvailable());
        assertFalse(result.checkpointIntegrityPassed());
        assertEquals(List.of("checkpointManifestIntegrityMismatch"), result.checkpointIntegrityFailureKeys());

        String markdown = Gollek.DL.trainingReportValidationMarkdown(result);
        String xml = Gollek.DL.trainingReportValidationJUnitXml(result);

        assertTrue(markdown.contains("**Validation:** `FAIL`"));
        assertTrue(markdown.contains("`checkpoint_integrity.failed`"));
        assertTrue(markdown.contains("checkpointManifestIntegrityMismatch"));
        assertWellFormedXml(xml);
        assertTrue(xml.contains("<testsuite name=\"gollek.training.report.validation\" tests=\"1\" failures=\"1\""));
        assertTrue(xml.contains("<failure type=\"CHECKPOINT_INTEGRITY_FAILED\""));
        assertTrue(xml.contains("name=\"checkpointIntegrity.failureKeys\" value=\"checkpointManifestIntegrityMismatch\""));
    }

    @Test
    void explicitDataHealthGateBlocksEvenWhenDiagnosticSeverityIsPermissive() {
        Map<String, Object> dataIssue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-plan-unavailable",
                "severity", "error",
                "blocking", true,
                "message", "train loader plan was not captured",
                "action", "enable data-loader plan capture before promotion");
        TrainingReport report = TrainingReport.of(validatedPayload(Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "error"),
                Map.entry("dataLoaderPlanHealthHealthy", false),
                Map.entry("dataLoaderPlanHealthGatePassed", false),
                Map.entry("dataLoaderPlanHealthIssueDetected", true),
                Map.entry("dataLoaderPlanHealthIssueCount", 1),
                Map.entry("dataLoaderPlanHealthWarningCount", 0),
                Map.entry("dataLoaderPlanHealthErrorCount", 1),
                Map.entry("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-plan-unavailable")),
                Map.entry("dataLoaderPlanHealthIssueSeverities", List.of("error")),
                Map.entry("dataLoaderPlanHealthRecommendedActions", List.of(dataIssue.get("action"))),
                Map.entry("dataLoaderPlanHealthIssues", List.of(dataIssue)),
                Map.entry("dataDistributionHealth.available", true),
                Map.entry("dataDistributionHealthStatus", "healthy"),
                Map.entry("dataDistributionHealthHealthy", true),
                Map.entry("dataDistributionHealthGatePassed", true),
                Map.entry("dataDistributionHealthIssueDetected", false),
                Map.entry("dataDistributionHealthIssueCount", 0),
                Map.entry("dataDistributionHealthWarningCount", 0),
                Map.entry("dataDistributionHealthErrorCount", 0),
                Map.entry("dataDistributionHealthIssues", List.of()))));
        TrainingReportValidationPolicy policy = TrainingReportValidationPolicy.permissive()
                .withRequireDataHealthGate(true);

        TrainingReportValidationPolicy.Result result = report.validate(policy);

        assertTrue(result.failed());
        assertTrue(result.diagnosticGate().passed());
        assertFalse(result.dataHealth().gatePassed());
        assertEquals(1, result.dataHealth().errorCount());
        assertEquals(List.of("data_health.gate_failed"), result.failureCodes());
        assertTrue(result.reasons().get(0).contains("enable data-loader plan capture"));
        assertEquals(Boolean.TRUE, ((Map<?, ?>) result.toMap().get("policy")).get("requireDataHealthGate"));
        assertEquals(Boolean.FALSE, ((Map<?, ?>) result.toMap().get("dataHealth")).get("gatePassed"));

        TrainingReportValidationPolicy.Result roundTrip =
                TrainingReportValidationPolicy.Result.fromMap(result.toMap());
        assertEquals(result.failureCodes(), roundTrip.failureCodes());
        assertEquals(result.dataHealth().issueCodes(), roundTrip.dataHealth().issueCodes());
        assertTrue(result.markdown().contains("**Data health:** `error`"));
        assertTrue(result.junitXml().contains("name=\"dataHealth.gatePassed\" value=\"false\""));
    }

    @Test
    void explicitDataHealthAvailabilityBlocksLegacyReportsWithoutDataHealth() {
        TrainingReport report = TrainingReport.of(validatedPayload(Map.of()));
        TrainingReportValidationPolicy policy = TrainingReportValidationPolicy.permissive()
                .withRequireDataHealthAvailable(true);

        TrainingReportValidationPolicy.Result result = report.validate(policy);

        assertTrue(result.failed());
        assertTrue(result.diagnosticGate().passed());
        assertFalse(result.dataHealth().available());
        assertTrue(result.policy().requireDataHealthAvailable());
        assertEquals(List.of("data_health.missing"), result.failureCodes());
        assertTrue(result.reasons().get(0).contains("data health was not recorded"));
        assertEquals(Boolean.TRUE, ((Map<?, ?>) result.toMap().get("policy")).get("requireDataHealthAvailable"));

        TrainingReportValidationPolicy.Result roundTrip =
                TrainingReportValidationPolicy.Result.fromMap(result.toMap());
        assertTrue(roundTrip.policy().requireDataHealthAvailable());
        assertEquals(result.failureCodes(), roundTrip.failureCodes());
        assertTrue(result.markdown().contains("`requireDataHealthAvailable` | `true`"));
        assertTrue(result.junitXml().contains("name=\"policy.requireDataHealthAvailable\" value=\"true\""));
    }

    @Test
    void permissivePolicyAllowsLegacyReportsWhenCallerOptsIn() {
        TrainingReport report = TrainingReport.of(staleFailedRunHealthReport());

        TrainingReportValidationPolicy.Result result = report.validate(TrainingReportValidationPolicy.permissive());

        assertTrue(result.passed());
        assertEquals(List.of(), result.failureCodes());
        assertFalse(result.validationAvailable());
        assertEquals("BACKFILLED", result.diagnosticProvenance().status());
    }

    @Test
    void writesReadsAndVerifiesValidationArtifacts() throws IOException {
        TrainingReport report = TrainingReport.of(validatedPayload(Map.of()));
        Path reportFile = tempDir.resolve("artifact-source-report.json");
        Files.writeString(reportFile, TrainerJson.toJson(report.payload()), StandardCharsets.UTF_8);
        TrainingReportValidationPolicy.Result result = Gollek.DL.validateTrainingReport(reportFile);

        TrainingReportValidationArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportValidationArtifacts(tempDir.resolve("validation-artifacts"), result);

        assertTrue(bundle.passed());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertDoesNotThrow(bundle::requirePassed);

        TrainingReportValidationArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportValidationArtifacts(bundle.directory());
        assertTrue(inspection.passed());
        assertEquals(List.of(), inspection.failureCodes());
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertTrue(inspection.markdown().contains("# Gollek Training Report Validation"));
        assertTrue(inspection.junitXml().contains("gollek.training.report.validation"));

        TrainingReportValidationArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportValidationArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.junitXmlSha256Matches());
        assertTrue(verification.junitXmlWellFormed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertDoesNotThrow(verification::requirePassed);

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered after artifact verification.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportValidationArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportValidationArtifacts(bundle);

        assertFalse(tampered.passed());
        assertFalse(tampered.markdownSha256Matches());
        assertFalse(tampered.markdownMatchesJson());
        assertTrue(tampered.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown checksum mismatch")));
        assertThrows(IllegalStateException.class, tampered::requirePassed);
    }

    @Test
    void validationArtifactVerificationAndRefreshUseJsonAsSourceOfTruth() throws IOException {
        TrainingReportValidationPolicy.Result result =
                TrainingReport.of(validatedPayload(Map.of())).validate();
        Path artifacts = tempDir.resolve("refresh-validation-artifacts");
        TrainingReportValidationArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportValidationArtifacts(artifacts, result);

        Files.writeString(
                bundle.markdownFile(),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        Files.writeString(
                bundle.junitXmlFile(),
                "\n<!-- manual stale edit -->\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportValidationArtifacts.ArtifactVerification stale =
                Gollek.DL.verifyTrainingReportValidationArtifacts(artifacts, null, null, null);

        assertFalse(stale.passed());
        assertFalse(stale.markdownMatchesJson());
        assertFalse(stale.junitXmlMatchesJson());
        assertTrue(stale.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown report does not match validation JSON")));

        TrainingReportValidationArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportValidationArtifacts(artifacts);
        TrainingReportValidationArtifacts.ArtifactVerification refreshedVerification =
                Gollek.DL.verifyTrainingReportValidationArtifacts(refreshed);

        assertEquals(bundle.jsonSha256(), refreshed.jsonSha256());
        assertFalse(Files.readString(refreshed.markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertFalse(Files.readString(refreshed.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertTrue(refreshedVerification.passed());
        assertTrue(refreshedVerification.markdownMatchesJson());
        assertTrue(refreshedVerification.junitXmlMatchesJson());
    }

    @Test
    void validationArtifactVerificationRejectsXmlDoctype() throws IOException {
        TrainingReportValidationPolicy.Result result =
                TrainingReport.of(validatedPayload(Map.of())).validate();
        TrainingReportValidationArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportValidationArtifacts(tempDir.resolve("xxe-validation-artifacts"), result);

        Files.writeString(bundle.junitXmlFile(), xmlWithDoctype(), StandardCharsets.UTF_8);

        TrainingReportValidationArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportValidationArtifacts(bundle.directory(), null, null, null);

        assertFalse(verification.passed());
        assertFalse(verification.junitXmlWellFormed());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML is not well-formed")));
    }

    private static void assertWellFormedXml(String xml) {
        assertDoesNotThrow(() -> DocumentBuilderFactory.newInstance()
                .newDocumentBuilder()
                .parse(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8))));
    }

    private static Map<String, Object> validatedPayload(Map<String, Object> extraMetadata) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("epochHistory", List.of(
                Map.ofEntries(
                        Map.entry("epoch", 0),
                        Map.entry("trainLoss", 0.8),
                        Map.entry("validationLoss", 0.9),
                        Map.entry("learningRate", 0.01)),
                Map.ofEntries(
                        Map.entry("epoch", 1),
                        Map.entry("trainLoss", 0.5),
                        Map.entry("validationLoss", 0.55),
                        Map.entry("learningRate", 0.005))));
        metadata.putAll(extraMetadata);
        TrainingSummary summary = new TrainingSummary(
                2,
                0.55,
                1,
                0.5,
                0.55,
                123L,
                metadata);
        return TrainerTrainingReport.payload(summary, Instant.parse("2026-05-28T01:02:03Z"));
    }

    private static String xmlWithDoctype() {
        return """
                <?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE testsuite [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
                <testsuite name="unsafe">&xxe;</testsuite>
                """;
    }

    private static Map<String, Object> staleFailedRunHealthReport() {
        Map<String, Object> primaryIssue = Map.of(
                "kind", "training-failure",
                "code", "non-finite-detected",
                "severity", "error",
                "blocking", true,
                "artifact", "trainer",
                "message", "train gradient must be finite, got NaN",
                "action", "inspect data and loss scale before rerunning");
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schema", TrainingReportReader.CANONICAL_SCHEMA);
        report.put("history", List.of());
        report.put("historySummary", Map.of("available", false));
        report.put("runHealth", Map.ofEntries(
                Map.entry("status", "failed"),
                Map.entry("healthy", false),
                Map.entry("gatePassed", false),
                Map.entry("issueDetected", true),
                Map.entry("issueCount", 1),
                Map.entry("blockingIssueDetected", true),
                Map.entry("blockingIssueCount", 1),
                Map.entry("recommendedAction", primaryIssue.get("action")),
                Map.entry("primaryIssue", primaryIssue),
                Map.entry("primaryBlockingIssue", primaryIssue),
                Map.entry("issueCodes", List.of("non-finite-detected")),
                Map.entry("issueSeverities", List.of("error")),
                Map.entry("issueCountsByKind", Map.of("training-failure", 1)),
                Map.entry("issueCountsBySeverity", Map.of("error", 1)),
                Map.entry("issues", List.of(primaryIssue))));
        report.put("diagnostics", List.of(Map.of(
                "severity", "INFO",
                "code", "history.missing",
                "message", "No epoch history is available in this training report.",
                "evidence", Map.of("historyRows", 0))));
        report.put("diagnosticsSummary", Map.of(
                "available", true,
                "total", 1,
                "highestSeverity", "INFO",
                "bySeverity", Map.of("INFO", 1, "WARNING", 0, "CRITICAL", 0),
                "codes", List.of("history.missing")));
        return report;
    }
}
