package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;

class TrainingReportQualityProfileCiGateManifestTest {
    @TempDir
    Path tempDir;

    @Test
    void writesVerifiesRefreshesAndAuditsCiGateManifest() throws IOException {
        Path outputDirectory = tempDir.resolve("manifest-ci");
        TrainingReportQualityProfileCiGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileCiGate(
                        TrainingReportQualityProfileTestFixtures.reportFiles(tempDir, Map.of(), Map.of()),
                        "baseline",
                        "local-experiment",
                        outputDirectory);

        TrainingReportQualityProfileCiGateManifest.ManifestBundle bundle =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifest(outputDirectory, result);

        assertTrue(bundle.passed());
        assertEquals(14, bundle.artifacts().size());
        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));

        TrainingReportQualityProfileCiGateManifest.ManifestInspection inspection =
                Gollek.DL.readTrainingReportQualityProfileCiGateManifest(outputDirectory);
        assertEquals(TrainingReportQualityProfileCiGateManifest.FORMAT, inspection.format());
        assertEquals(TrainingReportQualityProfile.LOCAL_EXPERIMENT, inspection.profileId().orElseThrow());
        assertEquals(14, inspection.artifactMaps().size());
        assertTrue(inspection.markdown().contains("# Gollek Training Quality Profile CI Gate Manifest"));

        TrainingReportQualityProfileCiGateManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifest(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.profileKnown());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.structureValid());
        assertTrue(verification.artifactsMatch());
        assertTrue(verification.checksumFailures().isEmpty());
        assertTrue(verification.formatFailures().isEmpty());
        assertTrue(verification.profileFailures().isEmpty());
        assertTrue(verification.markdownFailures().isEmpty());
        assertTrue(verification.structureFailures().isEmpty());
        assertTrue(verification.artifactFailures().isEmpty());
        assertTrue(verification.structuredFailures().isEmpty());
        assertTrue(verification.failureCountsByCategory().isEmpty());
        TrainingReportQualityProfileCiGateManifestSummary cleanSummary = verification.summary();
        assertTrue(cleanSummary.passed());
        assertTrue(cleanSummary.readyForRelease());
        assertEquals("passed", cleanSummary.status());
        assertEquals(0, cleanSummary.failureCount());
        assertEquals(List.of(), cleanSummary.failedCategories());
        assertTrue(cleanSummary.primaryFailure().isEmpty());
        assertEquals(Boolean.TRUE, verification.toMap().get("structureValid"));
        assertEquals(
                "passed",
                ((Map<?, ?>) verification.toMap().get("summary")).get("status"));
        assertEquals(List.of(), verification.toMap().get("checksumFailures"));
        assertEquals(List.of(), verification.toMap().get("formatFailures"));
        assertEquals(List.of(), verification.toMap().get("profileFailures"));
        assertEquals(List.of(), verification.toMap().get("markdownFailures"));
        assertEquals(List.of(), verification.toMap().get("structuredFailures"));
        assertDoesNotThrow(verification::requirePassed);
        String junitXml = Gollek.DL.trainingReportQualityProfileCiGateManifestJUnitXml(verification);
        assertTrue(TrainingReportXml.isWellFormed(junitXml));
        assertTrue(junitXml.contains("tests=\"6\" failures=\"0\""));
        assertTrue(junitXml.contains("name=\"structure\""));
        assertTrue(junitXml.contains("manifest.status\" value=\"passed\""));
        assertTrue(junitXml.contains("manifest.readyForRelease\" value=\"true\""));
        assertTrue(junitXml.contains("manifest.failures.checksum\" value=\"0\""));
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection junitXmlContract =
                Gollek.DL.inspectTrainingReportQualityProfileCiGateManifestJUnitXml(junitXml, verification);
        assertTrue(junitXmlContract.contractValid());
        assertEquals(
                List.of("checksums", "format", "profile", "markdown", "structure", "artifacts"),
                junitXmlContract.testcaseNames());
        assertEquals(6, junitXmlContract.testcaseCount());
        assertEquals(17, junitXmlContract.propertyCount());
        assertEquals(6, junitXmlContract.counts().expectedTestCount());
        assertEquals(6, junitXmlContract.counts().declaredTestCount());
        assertEquals(6, junitXmlContract.counts().observedTestcaseCount());
        assertEquals(0, junitXmlContract.counts().expectedFailureCount());
        assertEquals(0, junitXmlContract.counts().declaredFailureCount());
        assertEquals("passed", junitXmlContract.manifestStatus());
        assertEquals("true", junitXmlContract.manifestReadyForRelease());
        assertEquals("true", junitXmlContract.properties().get("manifest.readyForRelease"));
        assertEquals("0", junitXmlContract.properties().get("manifest.failureCount"));
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection badJunitXmlContract =
                Gollek.DL.inspectTrainingReportQualityProfileCiGateManifestJUnitXml(
                        junitXml.replace("tests=\"6\"", "tests=\"99\""),
                        verification);
        assertFalse(badJunitXmlContract.contractValid());
        assertEquals(99, badJunitXmlContract.counts().declaredTestCount());
        assertEquals(6, badJunitXmlContract.counts().observedTestcaseCount());
        assertTrue(badJunitXmlContract.failures().stream()
                .anyMatch(failure -> failure.contains("Expected 6 JUnit XML testcases")));
        TrainingReportQualityProfileCiGateManifestJUnitXml.Report junitXmlReport =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestJUnitXml(
                        outputDirectory.resolve("manifest-verification.junit.xml"),
                        verification);
        assertTrue(junitXmlReport.passed());
        assertTrue(junitXmlReport.wellFormed());
        assertEquals(TrainerCheckpointIO.sha256Hex(junitXmlReport.junitXmlFile()), junitXmlReport.junitXmlSha256());
        String verificationJson = Gollek.DL.trainingReportQualityProfileCiGateManifestVerificationJson(verification);
        String verificationMarkdown =
                Gollek.DL.trainingReportQualityProfileCiGateManifestVerificationMarkdown(verification);
        assertTrue(verificationJson.contains(
                "\"format\":\"" + TrainingReportQualityProfileCiGateManifestVerificationReport.FORMAT + "\""));
        assertTrue(verificationJson.contains("\"readyForRelease\":true"));
        assertTrue(verificationJson.contains("\"junitXmlContract\":"));
        assertTrue(verificationJson.contains("\"counts\":"));
        assertTrue(verificationJson.contains("\"contractValid\":true"));
        assertTrue(verificationJson.contains("\"declaredTestCount\":6"));
        assertTrue(verificationJson.contains("\"expectedTestCount\":6"));
        assertTrue(verificationJson.contains("\"observedTestcaseCount\":6"));
        assertTrue(verificationJson.contains("\"propertyCount\":17"));
        assertTrue(verificationJson.contains("\"manifestStatus\":\"passed\""));
        assertTrue(verificationJson.contains("\"testcaseNames\":[\"checksums\",\"format\",\"profile\""));
        assertTrue(verificationMarkdown.contains("**Status:** `PASS`"));
        assertTrue(verificationMarkdown.contains("No manifest verification failures were found."));
        assertTrue(verificationMarkdown.contains("## JUnit XML Contract"));
        assertTrue(verificationMarkdown.contains("**Contract valid:** `true`"));
        assertTrue(verificationMarkdown.contains("**Expected tests:** `6`"));
        assertTrue(verificationMarkdown.contains("**Observed testcases:** `6`"));
        assertTrue(verificationMarkdown.contains("**Property count:** `17`"));
        assertTrue(verificationMarkdown.contains("**Manifest status:** `passed`"));
        assertTrue(verificationMarkdown.contains("**Testcases:** `checksums, format, profile"));
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle verificationReport =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                        outputDirectory.resolve("manifest-verification-report"),
                        verification);
        assertTrue(verificationReport.passed());
        assertTrue(verificationReport.readyForRelease());
        assertTrue(Files.isRegularFile(verificationReport.jsonFile()));
        assertTrue(Files.isRegularFile(verificationReport.markdownFile()));
        assertTrue(Files.isRegularFile(verificationReport.junitXmlReport().junitXmlFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(verificationReport.jsonFile()), verificationReport.jsonSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(verificationReport.markdownFile()),
                verificationReport.markdownSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(verificationReport.junitXmlReport().junitXmlFile()),
                verificationReport.junitXmlReport().junitXmlSha256());
        assertTrue(TrainingReportXml.isWellFormed(verificationReport.junitXmlReport().junitXml()));
        assertEquals(Boolean.TRUE, verificationReport.toMap().get("readyForRelease"));
        assertEquals(
                Boolean.TRUE,
                ((Map<?, ?>) verificationReport.toMap().get("junitXmlContract")).get("contractValid"));
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportInspection reportInspection =
                Gollek.DL.readTrainingReportQualityProfileCiGateManifestVerificationReport(
                        verificationReport.directory());
        assertEquals(TrainingReportQualityProfileCiGateManifestVerificationReport.FORMAT, reportInspection.format());
        assertEquals(verificationReport.jsonSha256(), reportInspection.jsonSha256());
        assertEquals(verificationReport.markdownSha256(), reportInspection.markdownSha256());
        assertEquals(verificationReport.junitXmlReport().junitXmlSha256(), reportInspection.junitXmlSha256());
        assertTrue(reportInspection.junitXmlWellFormed());
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification reportVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                        verificationReport.directory(),
                        verification);
        assertTrue(reportVerification.passed());
        assertTrue(reportVerification.formatValid());
        assertTrue(reportVerification.jsonMatchesVerification());
        assertTrue(reportVerification.markdownMatchesVerification());
        assertTrue(reportVerification.junitXmlMatchesVerification());
        assertTrue(reportVerification.junitXmlWellFormed());
        assertTrue(reportVerification.junitXmlContractValid());
        assertTrue(reportVerification.junitXmlContract().contractValid());
        assertDoesNotThrow(reportVerification::requirePassed);
        assertEquals(Boolean.TRUE, reportVerification.toMap().get("jsonMatchesVerification"));
        assertEquals(Boolean.TRUE, reportVerification.toMap().get("junitXmlContractValid"));
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.Receipt receipt =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                        outputDirectory.resolve("manifest-verification-report.receipt.json"),
                        reportVerification);
        assertTrue(receipt.passed());
        assertTrue(Files.isRegularFile(receipt.receiptFile()));
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection receiptInspection =
                Gollek.DL.readTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                        receipt.receiptFile());
        assertEquals(
                TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.FORMAT,
                receiptInspection.format());
        assertEquals(reportInspection.directory(), receiptInspection.reportDirectory());
        assertEquals(reportInspection.jsonFile(), receiptInspection.jsonFile());
        assertEquals(Files.size(reportInspection.jsonFile()), receiptInspection.jsonBytes());
        assertEquals(Files.size(reportInspection.markdownFile()), receiptInspection.markdownBytes());
        assertEquals(Files.size(reportInspection.junitXmlFile()), receiptInspection.junitXmlBytes());
        assertEquals(reportInspection.jsonFile(), receiptInspection.jsonFingerprint().file());
        assertEquals(reportInspection.markdownFile(), receiptInspection.markdownFingerprint().file());
        assertEquals(reportInspection.junitXmlFile(), receiptInspection.junitXmlFingerprint().file());
        assertEquals(reportInspection.jsonSha256(), receiptInspection.jsonFingerprint().sha256());
        assertEquals(reportInspection.markdownSha256(), receiptInspection.markdownFingerprint().sha256());
        assertEquals(reportInspection.junitXmlSha256(), receiptInspection.junitXmlFingerprint().sha256());
        assertTrue(receiptInspection.artifactFingerprints().containsKey("json"));
        assertTrue(receiptInspection.artifactFingerprints().containsKey("markdown"));
        assertTrue(receiptInspection.artifactFingerprints().containsKey("junitXml"));
        assertTrue(receiptInspection.passed());
        assertTrue(receiptInspection.reportVerified());
        assertTrue(receiptInspection.junitXmlContractValid());
        assertEquals("passed", receiptInspection.manifestStatus());
        assertTrue(receiptInspection.receipt().containsKey("junitXmlContract"));
        assertTrue(receiptInspection.receipt().containsKey("verificationSummary"));
        assertTrue(receiptInspection.receipt().containsKey("artifactFingerprints"));
        assertFalse(receiptInspection.receipt().containsKey("verification"));
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification receiptVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                        receipt.receiptFile(),
                        receipt.receiptSha256(),
                        verification);
        assertTrue(receiptVerification.passed());
        assertTrue(receiptVerification.receiptSha256Matches());
        assertTrue(receiptVerification.schemaValid());
        assertTrue(receiptVerification.reportRevalidated());
        assertDoesNotThrow(receiptVerification::requirePassed);
        assertEquals(Boolean.TRUE, receiptVerification.toMap().get("reportRevalidated"));
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification
                badReceiptChecksumVerification =
                        Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                                receipt.receiptFile(),
                                "0000",
                                verification);
        assertFalse(badReceiptChecksumVerification.passed());
        assertFalse(badReceiptChecksumVerification.receiptSha256Matches());
        assertTrue(badReceiptChecksumVerification.failures().stream()
                .anyMatch(failure -> failure.contains("receipt checksum mismatch")));
        Path staleReceiptFile = outputDirectory.resolve("stale-manifest-verification-report.receipt.json");
        Files.writeString(
                staleReceiptFile,
                Files.readString(receipt.receiptFile(), StandardCharsets.UTF_8)
                        .replace("\"jsonBytes\":" + receiptInspection.jsonBytes(), "\"jsonBytes\":1"),
                StandardCharsets.UTF_8);
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification
                staleReceiptVerification =
                        Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                                staleReceiptFile,
                                verification);
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.failures().stream()
                .anyMatch(failure -> failure.contains("jsonBytes")));
        Path staleFingerprintReceiptFile =
                outputDirectory.resolve("stale-fingerprint-manifest-verification-report.receipt.json");
        Map<String, Object> staleFingerprintReceipt = new LinkedHashMap<>(receiptInspection.receipt());
        Map<String, Object> staleArtifactFingerprints =
                stringKeyMap((Map<?, ?>) staleFingerprintReceipt.get("artifactFingerprints"));
        Map<String, Object> staleJsonFingerprint =
                stringKeyMap((Map<?, ?>) staleArtifactFingerprints.get("json"));
        staleJsonFingerprint.put("bytes", 1);
        staleArtifactFingerprints.put("json", staleJsonFingerprint);
        staleFingerprintReceipt.put("artifactFingerprints", staleArtifactFingerprints);
        Files.writeString(
                staleFingerprintReceiptFile,
                TrainerJson.toJson(staleFingerprintReceipt) + "\n",
                StandardCharsets.UTF_8);
        TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification
                staleFingerprintReceiptVerification =
                        Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                                staleFingerprintReceiptFile,
                                verification);
        assertFalse(staleFingerprintReceiptVerification.passed());
        assertTrue(staleFingerprintReceiptVerification.failures().stream()
                .anyMatch(failure -> failure.contains("artifactFingerprints.json")));

        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle badJunitXmlReport =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                        outputDirectory.resolve("bad-junit-manifest-verification-report"),
                        verification);
        Files.writeString(
                badJunitXmlReport.junitXmlReport().junitXmlFile(),
                badJunitXmlReport.junitXmlReport().junitXml().replace("tests=\"6\"", "tests=\"99\""),
                StandardCharsets.UTF_8);
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification badJunitXmlReportVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                        badJunitXmlReport.directory(),
                        verification);
        assertFalse(badJunitXmlReportVerification.passed());
        assertTrue(badJunitXmlReportVerification.junitXmlWellFormed());
        assertFalse(badJunitXmlReportVerification.junitXmlMatchesVerification());
        assertFalse(badJunitXmlReportVerification.junitXmlContractValid());
        assertTrue(badJunitXmlReportVerification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML contract mismatch")));

        Files.writeString(
                bundle.markdownFile(),
                "\nTampered CI gate manifest docs.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification tamperedMarkdown =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifest(bundle);
        assertFalse(tamperedMarkdown.passed());
        assertFalse(tamperedMarkdown.markdownSha256Matches());
        assertFalse(tamperedMarkdown.markdownMatchesJson());
        assertTrue(tamperedMarkdown.checksumFailures().stream()
                .anyMatch(failure -> failure.contains("Markdown checksum mismatch")));
        assertTrue(tamperedMarkdown.markdownFailures().stream()
                .anyMatch(failure -> failure.contains("Markdown does not match")));
        assertTrue(tamperedMarkdown.formatFailures().isEmpty());
        assertTrue(tamperedMarkdown.profileFailures().isEmpty());
        assertTrue(tamperedMarkdown.structureFailures().isEmpty());
        assertTrue(tamperedMarkdown.artifactFailures().isEmpty());
        assertTrue(((List<?>) tamperedMarkdown.toMap().get("checksumFailures")).stream()
                .anyMatch(failure -> String.valueOf(failure).contains("Markdown checksum mismatch")));
        assertTrue(((List<?>) tamperedMarkdown.toMap().get("markdownFailures")).stream()
                .anyMatch(failure -> String.valueOf(failure).contains("Markdown does not match")));
        assertEquals(
                1,
                tamperedMarkdown.failureCountsByCategory()
                        .get(TrainingReportQualityProfileCiGateManifestFailureCategory.CHECKSUM));
        assertEquals(
                1,
                tamperedMarkdown.failureCountsByCategory()
                        .get(TrainingReportQualityProfileCiGateManifestFailureCategory.MARKDOWN));
        assertTrue(tamperedMarkdown.structuredFailures().stream()
                .anyMatch(failure -> failure.category()
                        == TrainingReportQualityProfileCiGateManifestFailureCategory.CHECKSUM));
        assertTrue(tamperedMarkdown.structuredFailures().stream()
                .anyMatch(failure -> failure.category()
                        == TrainingReportQualityProfileCiGateManifestFailureCategory.MARKDOWN));
        Map<?, ?> tamperedCategoryCounts =
                (Map<?, ?>) tamperedMarkdown.toMap().get("failureCountsByCategory");
        assertEquals(1, tamperedCategoryCounts.get("checksum"));
        assertEquals(1, tamperedCategoryCounts.get("markdown"));
        assertTrue(((List<?>) tamperedMarkdown.toMap().get("structuredFailures")).stream()
                .map(item -> (Map<?, ?>) item)
                .anyMatch(item -> "markdown".equals(item.get("category"))));
        TrainingReportQualityProfileCiGateManifestSummary markdownSummary = tamperedMarkdown.summary();
        assertFalse(markdownSummary.readyForRelease());
        assertEquals("failed", markdownSummary.status());
        assertEquals(2, markdownSummary.failureCount());
        assertEquals(List.of("checksum", "markdown"), markdownSummary.failedCategories());
        assertEquals(
                "checksum",
                markdownSummary.primaryFailureCategory().orElseThrow());
        assertTrue(markdownSummary.primaryFailureMessage().orElseThrow().contains("Markdown checksum mismatch"));
        assertEquals(1, markdownSummary.count(TrainingReportQualityProfileCiGateManifestFailureCategory.CHECKSUM));
        assertTrue(markdownSummary.hasFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.MARKDOWN));
        Map<?, ?> markdownSummaryMap = markdownSummary.toMap();
        assertEquals(Boolean.FALSE, markdownSummaryMap.get("readyForRelease"));
        assertEquals("checksum", markdownSummaryMap.get("primaryFailureCategory"));
        assertEquals(2, markdownSummaryMap.get("failureCount"));
        String tamperedMarkdownJunitXml =
                Gollek.DL.trainingReportQualityProfileCiGateManifestJUnitXml(tamperedMarkdown);
        assertTrue(tamperedMarkdownJunitXml.contains("tests=\"6\" failures=\"2\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.status\" value=\"failed\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.readyForRelease\" value=\"false\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.failedCategoryCount\" value=\"2\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.primaryFailureCategory\" value=\"checksum\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.failures.checksum\" value=\"1\""));
        assertTrue(tamperedMarkdownJunitXml.contains("manifest.failures.markdown\" value=\"1\""));
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle tamperedMarkdownReport =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                        outputDirectory.resolve("tampered-manifest-verification-report"),
                        tamperedMarkdown);
        assertFalse(tamperedMarkdownReport.passed());
        assertFalse(tamperedMarkdownReport.readyForRelease());
        assertTrue(tamperedMarkdownReport.json().contains("\"readyForRelease\":false"));
        assertTrue(tamperedMarkdownReport.json().contains("\"junitXmlContract\":"));
        assertTrue(tamperedMarkdownReport.json().contains("\"contractValid\":true"));
        assertTrue(tamperedMarkdownReport.markdown().contains("**Status:** `FAIL`"));
        assertTrue(tamperedMarkdownReport.markdown().contains("## JUnit XML Contract"));
        assertTrue(tamperedMarkdownReport.markdown().contains("| `checksum` | 1 |"));
        assertTrue(tamperedMarkdownReport.markdown().contains("| `markdown` | 1 |"));
        assertTrue(tamperedMarkdownReport.markdown().contains("- `checksum` Manifest Markdown checksum mismatch"));
        assertTrue(TrainingReportXml.isWellFormed(tamperedMarkdownReport.junitXmlReport().junitXml()));
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification cleanTamperedReportVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                        tamperedMarkdownReport.directory(),
                        tamperedMarkdown);
        assertTrue(cleanTamperedReportVerification.passed());
        Files.writeString(
                tamperedMarkdownReport.markdownFile(),
                "\nTampered verification report evidence.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification staleReportVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                        tamperedMarkdownReport.directory(),
                        tamperedMarkdown);
        assertFalse(staleReportVerification.passed());
        assertTrue(staleReportVerification.jsonMatchesVerification());
        assertFalse(staleReportVerification.markdownMatchesVerification());
        assertTrue(staleReportVerification.junitXmlMatchesVerification());
        assertTrue(staleReportVerification.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown content mismatch")));
        assertEquals(Boolean.FALSE, staleReportVerification.toMap().get("markdownMatchesVerification"));

        TrainingReportQualityProfileCiGateManifest.ManifestInspection refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileCiGateManifest(outputDirectory);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification refreshedVerification =
                TrainingReportQualityProfileCiGateManifest.verify(
                        refreshed,
                        bundle.jsonSha256(),
                        refreshed.markdownSha256());
        assertTrue(refreshedVerification.passed());

        Files.writeString(
                result.validationArtifacts().get("baseline").jsonFile(),
                "\nTampered validation artifact.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification tamperedArtifact =
                TrainingReportQualityProfileCiGateManifest.verify(
                        refreshed,
                        bundle.jsonSha256(),
                        refreshed.markdownSha256());
        assertFalse(tamperedArtifact.passed());
        assertFalse(tamperedArtifact.artifactsMatch());
        assertTrue(tamperedArtifact.failures().stream()
                .anyMatch(failure -> failure.contains("profileValidation.baseline.json")));
        assertTrue(tamperedArtifact.artifactFailures().stream()
                .anyMatch(failure -> failure.contains("profileValidation.baseline.json")));
        assertTrue(((List<?>) tamperedArtifact.toMap().get("artifactFailures")).stream()
                .anyMatch(failure -> String.valueOf(failure).contains("profileValidation.baseline.json")));
        assertEquals(
                2,
                tamperedArtifact.failureCountsByCategory()
                        .get(TrainingReportQualityProfileCiGateManifestFailureCategory.ARTIFACT));
        TrainingReportQualityProfileCiGateManifestSummary artifactSummary = tamperedArtifact.summary();
        assertEquals("failed", artifactSummary.status());
        assertEquals(2, artifactSummary.failureCount());
        assertEquals(List.of("artifact"), artifactSummary.failedCategories());
        assertEquals("artifact", artifactSummary.primaryFailureCategory().orElseThrow());
        String tamperedArtifactJunitXml =
                Gollek.DL.trainingReportQualityProfileCiGateManifestJUnitXml(tamperedArtifact);
        assertTrue(TrainingReportXml.isWellFormed(tamperedArtifactJunitXml));
        assertTrue(tamperedArtifactJunitXml.contains("tests=\"6\" failures=\"1\""));
        assertTrue(tamperedArtifactJunitXml.contains("name=\"artifacts\""));
        assertTrue(tamperedArtifactJunitXml.contains("manifest.failures.artifact\" value=\"2\""));
    }

    @Test
    void rejectsManifestWhenDeclaredArtifactCountDoesNotMatchArtifactList() throws IOException {
        Path outputDirectory = tempDir.resolve("artifact-count-ci");
        TrainingReportQualityProfileCiGateManifest.ManifestBundle bundle = writeLocalManifest(outputDirectory);
        rewriteManifestJson(bundle.jsonFile(), manifest -> manifest.put("artifactCount", 99));

        TrainingReportQualityProfileCiGateManifest.ManifestInspection refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileCiGateManifest(outputDirectory);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification verification =
                TrainingReportQualityProfileCiGateManifest.verify(refreshed, null, null);

        assertFalse(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertFalse(verification.structureValid());
        assertTrue(verification.checksumFailures().isEmpty());
        assertTrue(verification.formatFailures().isEmpty());
        assertTrue(verification.profileFailures().isEmpty());
        assertTrue(verification.markdownFailures().isEmpty());
        assertTrue(verification.artifactFailures().isEmpty());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("artifact count mismatch")));
        assertTrue(verification.structureFailures().stream()
                .anyMatch(failure -> failure.contains("artifact count mismatch")));
        assertEquals(
                1,
                verification.failureCountsByCategory()
                        .get(TrainingReportQualityProfileCiGateManifestFailureCategory.STRUCTURE));
        TrainingReportQualityProfileCiGateManifestSummary summary = verification.summary();
        assertEquals("failed", summary.status());
        assertFalse(summary.readyForRelease());
        assertEquals(List.of("structure"), summary.failedCategories());
        assertEquals("structure", summary.primaryFailureCategory().orElseThrow());
        assertEquals(Boolean.FALSE, verification.toMap().get("structureValid"));
        String junitXml = Gollek.DL.trainingReportQualityProfileCiGateManifestJUnitXml(verification);
        assertTrue(TrainingReportXml.isWellFormed(junitXml));
        assertTrue(junitXml.contains("tests=\"6\" failures=\"1\""));
        assertTrue(junitXml.contains("name=\"structure\""));
        assertTrue(junitXml.contains("artifact count mismatch"));
    }

    @Test
    void rejectsManifestWhenArtifactNamesAreDuplicated() throws IOException {
        Path outputDirectory = tempDir.resolve("duplicate-artifact-ci");
        TrainingReportQualityProfileCiGateManifest.ManifestBundle bundle = writeLocalManifest(outputDirectory);
        rewriteManifestJson(bundle.jsonFile(), manifest -> {
            List<Map<String, Object>> artifacts = mutableArtifactMaps(manifest);
            artifacts.get(1).put("name", artifacts.get(0).get("name"));
            manifest.put("artifacts", artifacts);
        });

        TrainingReportQualityProfileCiGateManifest.ManifestInspection refreshed =
                Gollek.DL.refreshTrainingReportQualityProfileCiGateManifest(outputDirectory);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification verification =
                TrainingReportQualityProfileCiGateManifest.verify(refreshed, null, null);

        assertFalse(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertFalse(verification.structureValid());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("artifact name is duplicated")));
        assertTrue(verification.structureFailures().stream()
                .anyMatch(failure -> failure.contains("artifact name is duplicated")));
    }

    private TrainingReportQualityProfileCiGateManifest.ManifestBundle writeLocalManifest(Path outputDirectory)
            throws IOException {
        TrainingReportQualityProfileCiGate.Result result =
                Gollek.DL.runTrainingReportQualityProfileCiGate(
                        TrainingReportQualityProfileTestFixtures.reportFiles(tempDir, Map.of(), Map.of()),
                        "baseline",
                        "local-experiment",
                        outputDirectory);
        return Gollek.DL.writeTrainingReportQualityProfileCiGateManifest(outputDirectory, result);
    }

    @SuppressWarnings("unchecked")
    private static void rewriteManifestJson(Path jsonFile, Consumer<Map<String, Object>> mutation)
            throws IOException {
        Object parsed = TrainerJsonParser.parse(Files.readString(jsonFile, StandardCharsets.UTF_8));
        Map<String, Object> manifest = new LinkedHashMap<>((Map<String, Object>) parsed);
        mutation.accept(manifest);
        Files.writeString(jsonFile, TrainerJson.toJson(manifest) + "\n", StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> mutableArtifactMaps(Map<String, Object> manifest) {
        List<Map<String, Object>> artifacts = new ArrayList<>();
        for (Map<String, Object> artifact : (List<Map<String, Object>>) manifest.get("artifacts")) {
            artifacts.add(new LinkedHashMap<>(artifact));
        }
        return artifacts;
    }

    private static Map<String, Object> stringKeyMap(Map<?, ?> raw) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            copy.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return copy;
    }
}
