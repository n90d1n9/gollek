package tech.kayys.gollek.ml.train;

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
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportPromotionGateArtifactPackageTest {
    @TempDir
    Path tempDir;

    @Test
    void writesReadsAndVerifiesCompleteGateArtifactPackage() throws IOException {
        TrainingReportPromotionGate.Result result = gateResult("package");
        Instant generatedAt = Instant.parse("2026-05-21T09:10:11Z");

        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("package-artifacts"),
                        result,
                        generatedAt);

        assertTrue(bundle.passed());
        assertTrue(bundle.promotable());
        assertTrue(Files.isRegularFile(bundle.review().jsonFile()));
        assertTrue(Files.isRegularFile(bundle.review().markdownFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().jsonFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().markdownFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().junitXmlFile()));
        assertTrue(Files.isRegularFile(bundle.manifest().manifestFile()));
        assertEquals(generatedAt, bundle.manifest().generatedAt());
        assertEquals(bundle.review().directory(), bundle.directory());
        assertEquals(bundle.artifacts().directory(), bundle.directory());
        bundle.requirePassed();

        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifactPackage(bundle.directory());
        assertTrue(inspection.passed());
        assertTrue(inspection.promotable());
        assertEquals("candidate", inspection.review().decisionCandidate().orElseThrow());
        assertEquals("candidate", inspection.artifacts().decisionCandidate().orElseThrow());
        assertEquals("PROMOTE", inspection.manifest().decisionStatus());
        assertEquals(
                bundle.review().jsonSha256(),
                inspection.manifest().artifact("reviewJson").orElseThrow().sha256());
        assertEquals(bundle.artifacts().jsonSha256(), inspection.manifest().artifact("json").orElseThrow().sha256());
        assertTrue(Files.isRegularFile(inspection.manifest()
                .artifact("sourceReport.baseline.baseline")
                .orElseThrow()
                .file()));
        assertTrue(Files.isRegularFile(inspection.manifest()
                .artifact("sourceReport.candidate.candidate")
                .orElseThrow()
                .file()));
        assertEquals(2, inspection.sourceReportSnapshots().size());
        assertEquals(2, Gollek.DL.trainingReportPromotionGatePackageSourceSnapshots(inspection).size());

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertTrue(verification.artifactBytesMatch());
        assertTrue(verification.artifactSha256Match());
        assertTrue(verification.artifactVerification().junitXmlWellFormed());
        verification.requirePassed();

        TrainingReportPromotionGateArtifactPackage.PackageVerification completeVerification =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(bundle);
        assertTrue(completeVerification.passed());
        assertTrue(completeVerification.manifestVerification().passed());
        assertTrue(completeVerification.sourceSnapshotVerification().passed());
        assertTrue(completeVerification.toMap().containsKey("sourceSnapshotVerification"));
        completeVerification.requirePassed();

        TrainingReportPromotionGateArtifactPackage.VerificationReport report =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationReport(
                        tempDir.resolve("package-verification.json"),
                        completeVerification);
        assertTrue(report.passed());
        assertTrue(Files.isRegularFile(report.reportFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(report.reportFile()), report.reportSha256());
        String reportJson = Files.readString(report.reportFile(), StandardCharsets.UTF_8);
        assertTrue(reportJson.contains("\"passed\":true"));
        assertTrue(reportJson.contains("\"format\":\"gollek.training.promotion.package.verification.report.v1\""));
        assertTrue(reportJson.contains("\"expectedSourceReportArtifacts\""));
        TrainingReportPromotionGateArtifactPackage.VerificationReportInspection reportInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationReport(report.reportFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT,
                reportInspection.format());
        assertEquals(bundle.directory(), reportInspection.packageDirectory());
        assertEquals(bundle.manifest().manifestSha256(), reportInspection.manifestSha256());
        TrainingReportPromotionGateArtifactPackage.VerificationReportVerification reportVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReport(
                        report.reportFile(),
                        report.reportSha256());
        assertTrue(reportVerification.passed(), reportVerification.failures().toString());
        assertTrue(reportVerification.reportSha256Matches());
        assertTrue(reportVerification.schemaValid());
        assertTrue(reportVerification.packageRevalidated());
        assertTrue(reportVerification.packageVerification().passed());
        reportVerification.requirePassed();

        TrainingReportPromotionGateArtifactPackage.VerificationReport defaultReport =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(bundle.directory());
        assertTrue(defaultReport.passed());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationReportFile(bundle.directory()),
                defaultReport.reportFile());
        assertTrue(Files.isRegularFile(defaultReport.reportFile()));
        String defaultReportJson = Files.readString(defaultReport.reportFile(), StandardCharsets.UTF_8);
        assertTrue(defaultReportJson.contains("\"sourceSnapshotVerification\""));
        assertTrue(defaultReportJson.contains("\"unexpectedSourceReportArtifacts\""));

        TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport markdownReport =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationMarkdownReport(
                        tempDir.resolve("package-verification.md"),
                        completeVerification);
        assertTrue(markdownReport.passed());
        assertEquals(TrainerCheckpointIO.sha256Hex(markdownReport.markdownFile()), markdownReport.markdownSha256());
        String markdown = Files.readString(markdownReport.markdownFile(), StandardCharsets.UTF_8);
        assertTrue(markdown.contains("# Gollek Promotion Gate Package Verification"));
        assertTrue(markdown.contains("| Status | PASS |"));
        assertTrue(markdown.contains("| Source report snapshots | PASS |"));
        assertTrue(markdown.contains("## Package Artifacts"));
        assertTrue(markdown.contains("## Source Report Snapshots"));

        TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport defaultMarkdownReport =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                        bundle.directory());
        assertTrue(defaultMarkdownReport.passed());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationMarkdownFile(bundle.directory()),
                defaultMarkdownReport.markdownFile());
        assertTrue(Files.readString(defaultMarkdownReport.markdownFile(), StandardCharsets.UTF_8)
                .contains("## Failures"));

        TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport junitXmlReport =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                        tempDir.resolve("package-verification.junit.xml"),
                        completeVerification);
        assertTrue(junitXmlReport.passed());
        assertEquals(TrainerCheckpointIO.sha256Hex(junitXmlReport.junitXmlFile()), junitXmlReport.junitXmlSha256());
        String junitXml = Files.readString(junitXmlReport.junitXmlFile(), StandardCharsets.UTF_8);
        assertTrue(junitXml.contains("<testsuite name=\"gollek.training.promotion.package\" tests=\"3\" failures=\"0\""));
        assertTrue(junitXml.contains("name=\"manifest\""));
        assertTrue(junitXml.contains("name=\"source-report-snapshots\""));
        assertTrue(junitXml.contains("name=\"complete-package\""));
        assertTrue(junitXml.contains("property name=\"sourceSnapshots.expected\" value=\"2\""));

        TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport defaultJunitXmlReport =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                        bundle.directory());
        assertTrue(defaultJunitXmlReport.passed());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationJunitXmlFile(bundle.directory()),
                defaultJunitXmlReport.junitXmlFile());
        assertTrue(Files.readString(defaultJunitXmlReport.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("property name=\"package.passed\" value=\"true\""));

        Path reportBundleDirectory = tempDir.resolve("package-verification-bundle");
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundle reportBundle =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationReports(
                        reportBundleDirectory,
                        completeVerification);
        assertTrue(reportBundle.passed());
        assertEquals(completeVerification, reportBundle.verification());
        assertEquals(reportBundleDirectory.toAbsolutePath().normalize(), reportBundle.directory());
        assertTrue(Files.isRegularFile(reportBundle.json().reportFile()));
        assertTrue(Files.isRegularFile(reportBundle.markdown().markdownFile()));
        assertTrue(Files.isRegularFile(reportBundle.junitXml().junitXmlFile()));
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationReportFile(reportBundleDirectory),
                reportBundle.json().reportFile());
        assertTrue(reportBundle.toMap().containsKey("junitXml"));

        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection reportBundleInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationReports(reportBundleDirectory);
        assertEquals(reportBundleDirectory.toAbsolutePath().normalize(), reportBundleInspection.directory());
        assertEquals(reportBundle.json().reportFile(), reportBundleInspection.json().reportFile());
        assertEquals(reportBundle.markdown().markdownFile(), reportBundleInspection.markdownFile());
        assertEquals(reportBundle.junitXml().junitXmlFile(), reportBundleInspection.junitXmlFile());

        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification reportBundleVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReports(
                        reportBundleDirectory,
                        reportBundle.json().reportSha256());
        assertTrue(reportBundleVerification.passed(), reportBundleVerification.failures().toString());
        assertTrue(reportBundleVerification.jsonReportVerified());
        assertTrue(reportBundleVerification.markdownMatchesRendered());
        assertTrue(reportBundleVerification.junitXmlMatchesRendered());
        assertTrue(reportBundleVerification.jsonVerification().passed());
        reportBundleVerification.requirePassed();

        TrainingReportPromotionGateArtifactPackage.VerificationReportBundle defaultReportBundle =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(bundle.directory());
        assertTrue(defaultReportBundle.passed());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationReportFile(bundle.directory()),
                defaultReportBundle.json().reportFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationMarkdownFile(bundle.directory()),
                defaultReportBundle.markdown().markdownFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationJunitXmlFile(bundle.directory()),
                defaultReportBundle.junitXml().junitXmlFile());

        TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification sourceVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageSourceSnapshots(inspection);
        assertTrue(sourceVerification.passed());
        assertEquals(2, sourceVerification.snapshots().size());
        assertEquals(
                List.of("sourceReport.baseline.baseline", "sourceReport.candidate.candidate"),
                sourceVerification.expectedSourceReportArtifactNames());
        assertEquals(
                Set.copyOf(sourceVerification.expectedSourceReportArtifactNames()),
                Set.copyOf(sourceVerification.presentSourceReportArtifactNames()));
        assertTrue(sourceVerification.missingSourceReportArtifactNames().isEmpty());
        assertTrue(sourceVerification.unexpectedSourceReportArtifactNames().isEmpty());
        assertTrue(sourceVerification.toMap().containsKey("snapshots"));
        assertTrue(sourceVerification.toMap().containsKey("unexpectedSourceReportArtifacts"));
        sourceVerification.requirePassed();
    }

    @Test
    void refreshCompletePackageRebuildsDerivedGateArtifactsAndManifest() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("refresh-package-artifacts"),
                        gateResult("refresh-package"),
                        Instant.parse("2026-05-21T09:10:11Z"));
        String originalManifestSha256 = bundle.manifest().manifestSha256();

        Files.writeString(
                bundle.artifacts().markdownFile(),
                "\nmanual stale package markdown edit\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        Files.writeString(
                bundle.artifacts().junitXmlFile(),
                "\n<!-- manual stale package junit edit -->\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.PackageVerification stale =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(bundle.directory());
        assertFalse(stale.passed());
        assertFalse(stale.manifestVerification().artifactSha256Match());
        assertFalse(stale.manifestVerification().artifactVerification().markdownMatchesJson());
        assertFalse(stale.manifestVerification().artifactVerification().junitXmlMatchesJson());

        TrainingReportPromotionGateArtifactPackage.PackageRefresh refresh =
                Gollek.DL.refreshTrainingReportPromotionGateArtifactPackage(bundle.directory());
        TrainingReportPromotionGateArtifactPackage.PackageVerification verification =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(bundle.directory());

        assertTrue(refresh.passed(), () -> refresh.toMap().toString());
        assertTrue(refresh.reportVerification().passed(), () -> refresh.reportVerification().failures().toString());
        assertTrue(verification.passed(), () -> verification.failures().toString());
        assertTrue(verification.manifestVerification().artifactSha256Match());
        assertTrue(verification.manifestVerification().artifactVerification().markdownMatchesJson());
        assertTrue(verification.manifestVerification().artifactVerification().junitXmlMatchesJson());
        assertTrue(Files.isRegularFile(refresh.reports().json().reportFile()));
        assertTrue(!originalManifestSha256.equals(refresh.manifest().manifestSha256()));
        assertTrue(!Files.readString(bundle.artifacts().markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale package markdown edit"));
        assertTrue(!Files.readString(bundle.artifacts().junitXmlFile(), StandardCharsets.UTF_8)
                .contains("manual stale package junit edit"));
    }

    @Test
    void refreshCompletePackageRejectsChangedNonRefreshableArtifact() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("refresh-reject-package-artifacts"),
                        gateResult("refresh-reject-package"),
                        Instant.parse("2026-05-21T09:10:11Z"));

        Files.writeString(
                bundle.review().markdownFile(),
                "\nmanual review edit must not be blessed\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        IOException error = assertThrows(
                IOException.class,
                () -> Gollek.DL.refreshTrainingReportPromotionGateArtifactPackage(bundle.directory()));
        assertTrue(error.getMessage().contains("non-refreshable artifact"));
    }

    @Test
    void writesPackageAndVerificationReportsInOneCall() throws IOException {
        Instant generatedAt = Instant.parse("2026-05-21T09:10:11Z");
        Path outputDirectory = tempDir.resolve("verified-package-artifacts");
        Path reportDirectory = tempDir.resolve("verified-package-reports");

        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        outputDirectory,
                        gateResult("verified-package"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        generatedAt,
                        reportDirectory);

        assertTrue(verified.passed(), () -> verified.toMap().toString());
        assertTrue(verified.promotable());
        verified.requirePassed();
        assertEquals(outputDirectory.toAbsolutePath().normalize(), verified.directory());
        assertEquals(generatedAt, verified.packageBundle().manifest().generatedAt());
        assertEquals(outputDirectory.toAbsolutePath().normalize(), verified.verification().inspection().directory());
        assertEquals(reportDirectory.toAbsolutePath().normalize(), verified.reports().directory());
        assertTrue(Files.isRegularFile(verified.packageBundle().manifest().manifestFile()));
        assertTrue(Files.isRegularFile(verified.reports().json().reportFile()));
        assertTrue(Files.isRegularFile(verified.reports().markdown().markdownFile()));
        assertTrue(Files.isRegularFile(verified.reports().junitXml().junitXmlFile()));
        Path generatedReportBundleReceiptFile =
                TrainingReportPromotionGateArtifactPackage.defaultVerificationReportBundleReceiptFile(
                        reportDirectory);
        assertTrue(verified.reportBundleReceipt() != null);
        assertEquals(generatedReportBundleReceiptFile, verified.reportBundleReceipt().receiptFile());
        assertTrue(Files.isRegularFile(verified.reportBundleReceipt().receiptFile()));
        assertTrue(verified.reportBundleReceipt().passed());
        String reportBundleReceiptJson = Files.readString(
                verified.reportBundleReceipt().receiptFile(),
                StandardCharsets.UTF_8);
        assertTrue(reportBundleReceiptJson.contains(
                "\"format\":\"gollek.training.promotion.package.verification.reports.receipt.v1\""));
        assertTrue(reportBundleReceiptJson.contains("\"markdownMatchesRendered\":true"));
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection
                reportBundleReceiptInspection =
                        Gollek.DL.readTrainingReportPromotionGatePackageVerificationReportsReceipt(
                                verified.reportBundleReceipt().receiptFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT,
                reportBundleReceiptInspection.format());
        assertEquals(verified.reports().directory(), reportBundleReceiptInspection.reportDirectory());
        assertEquals(verified.reports().json().reportFile(), reportBundleReceiptInspection.jsonReportFile());
        assertEquals(verified.reports().json().reportSha256(), reportBundleReceiptInspection.jsonReportSha256());
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification
                reportBundleReceiptVerification =
                        Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReportsReceipt(
                                verified.reportBundleReceipt().receiptFile(),
                                verified.reportBundleReceipt().receiptSha256());
        assertTrue(reportBundleReceiptVerification.passed(), reportBundleReceiptVerification.failures().toString());
        assertTrue(reportBundleReceiptVerification.receiptSha256Matches());
        assertTrue(reportBundleReceiptVerification.schemaValid());
        assertTrue(reportBundleReceiptVerification.reportBundleRevalidated());
        assertTrue(reportBundleReceiptVerification.reportBundleVerification().passed());
        reportBundleReceiptVerification.requirePassed();
        Path manualReportBundleReceiptFile =
                reportDirectory.resolve("manual-promotion-gate-package-verification.reports.receipt.json");
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt manualReportBundleReceipt =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationReportsReceipt(
                        verified.reports().directory(),
                        verified.reports().json().reportSha256(),
                        manualReportBundleReceiptFile);
        assertTrue(manualReportBundleReceipt.passed());
        assertEquals(manualReportBundleReceiptFile, manualReportBundleReceipt.receiptFile());
        assertTrue(Files.isRegularFile(manualReportBundleReceipt.receiptFile()));
        assertTrue(verified.index() != null);
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexFile(reportDirectory),
                verified.index().indexFile());
        assertTrue(Files.isRegularFile(verified.index().indexFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(verified.index().indexFile()), verified.index().indexSha256());
        String indexJson = Files.readString(verified.index().indexFile(), StandardCharsets.UTF_8);
        assertTrue(indexJson.contains("\"format\":\"gollek.training.promotion.package.verification.index.v1\""));
        assertTrue(indexJson.contains("\"reportDirectory\""));
        assertTrue(indexJson.contains("\"receipt\""));
        assertTrue(indexJson.contains("\"sourceReportSnapshots\""));
        TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection indexInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationIndex(verified.index().indexFile());
        assertEquals(TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT, indexInspection.format());
        assertEquals(outputDirectory.toAbsolutePath().normalize(), indexInspection.packageDirectory());
        assertEquals(reportDirectory.toAbsolutePath().normalize(), indexInspection.reportDirectory());
        assertEquals("candidate", indexInspection.decisionCandidate().orElseThrow());
        TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification indexVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());
        assertTrue(indexVerification.passed());
        assertTrue(indexVerification.indexSha256Matches());
        assertTrue(indexVerification.schemaValid());
        assertTrue(indexVerification.referencedSha256Match());
        assertTrue(indexVerification.toMap().containsKey("inspection"));
        indexVerification.requirePassed();
        Path generatedReceiptFile =
                TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexReceiptFile(reportDirectory);
        assertTrue(verified.receipt() != null);
        assertEquals(generatedReceiptFile, verified.receipt().receiptFile());
        assertTrue(verified.receipt().passed());
        Path receiptFile = reportDirectory.resolve("manual-promotion-gate-package-verification.index.receipt.json");
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt receipt =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        verified.index().indexFile(),
                        verified.index().indexSha256(),
                        receiptFile);
        assertTrue(receipt.passed());
        assertEquals(receiptFile, receipt.receiptFile());
        assertTrue(Files.isRegularFile(receipt.receiptFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(receipt.receiptFile()), receipt.receiptSha256());
        String receiptJson = Files.readString(receipt.receiptFile(), StandardCharsets.UTF_8);
        assertTrue(receiptJson.contains("\"format\":\"gollek.training.promotion.package.verification.index.receipt.v1\""));
        assertTrue(receiptJson.contains("\"referencedSha256Match\":true"));
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection receiptInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationIndexReceipt(receipt.receiptFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_RECEIPT_FORMAT,
                receiptInspection.format());
        assertEquals(verified.index().indexFile(), receiptInspection.indexFile());
        assertEquals(verified.index().indexSha256(), receiptInspection.indexSha256());
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification receiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        receipt.receiptFile(),
                        receipt.receiptSha256());
        assertTrue(receiptVerification.passed(), receiptVerification.failures().toString());
        assertTrue(receiptVerification.receiptSha256Matches());
        assertTrue(receiptVerification.schemaValid());
        assertTrue(receiptVerification.indexRevalidated());
        assertTrue(receiptVerification.indexVerification().passed());
        receiptVerification.requirePassed();
        receipt.requirePassed();
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit packageAudit =
                Gollek.DL.auditTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());
        assertTrue(packageAudit.passed());
        assertTrue(packageAudit.indexVerification().passed());
        assertTrue(packageAudit.packageVerification().passed());
        assertTrue(packageAudit.toMap().containsKey("packageVerification"));
        packageAudit.requirePassed();
        Path generatedAuditReportFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexPackageAuditFile(
                reportDirectory);
        assertTrue(verified.packageAuditReport() != null);
        assertEquals(generatedAuditReportFile, verified.packageAuditReport().reportFile());
        assertTrue(verified.packageAuditReport().passed());
        Path auditReportFile = reportDirectory.resolve("manual-promotion-gate-package-verification.index.package-audit.json");
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport auditReport =
                Gollek.DL.auditAndWriteTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                        verified.index().indexFile(),
                        verified.index().indexSha256(),
                        auditReportFile);
        assertTrue(auditReport.passed());
        assertTrue(Files.isRegularFile(auditReport.reportFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(auditReport.reportFile()), auditReport.reportSha256());
        String auditReportJson = Files.readString(auditReport.reportFile(), StandardCharsets.UTF_8);
        assertTrue(auditReportJson.contains(
                "\"format\":\"gollek.training.promotion.package.verification.index.package-audit.v1\""));
        assertTrue(auditReportJson.contains("\"packagePassed\":true"));
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection
                auditReportInspection =
                        Gollek.DL.readTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                                auditReport.reportFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT,
                auditReportInspection.format());
        assertEquals(verified.index().indexFile(), auditReportInspection.indexFile());
        assertEquals(verified.index().indexSha256(), auditReportInspection.indexSha256());
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification
                auditReportVerification =
                        Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                                auditReport.reportFile(),
                                auditReport.reportSha256());
        assertTrue(auditReportVerification.passed(), auditReportVerification.failures().toString());
        assertTrue(auditReportVerification.reportSha256Matches());
        assertTrue(auditReportVerification.schemaValid());
        assertTrue(auditReportVerification.auditRevalidated());
        assertTrue(auditReportVerification.audit().passed());
        auditReportVerification.requirePassed();
        auditReport.requirePassed();
        Path evidenceFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationEvidenceFile(reportDirectory);
        assertTrue(verified.evidence() != null);
        assertEquals(evidenceFile, verified.evidence().evidenceFile());
        assertTrue(Files.isRegularFile(verified.evidence().evidenceFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(verified.evidence().evidenceFile()), verified.evidence().evidenceSha256());
        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8);
        assertTrue(evidenceJson.contains("\"format\":\"gollek.training.promotion.package.verification.evidence.v1\""));
        assertTrue(evidenceJson.contains("\"evidenceFiles\""));
        assertTrue(evidenceJson.contains("\"verificationReportBundleReceipt\""));
        assertTrue(evidenceJson.contains("\"packageArtifacts\""));
        assertTrue(evidenceJson.contains("\"sourceReportSnapshots\""));
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection evidenceInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT,
                evidenceInspection.format());
        assertEquals(verified.directory(), evidenceInspection.packageDirectory());
        assertTrue(evidenceInspection.passed());
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification evidenceVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile(),
                        verified.evidence().evidenceSha256());
        assertTrue(evidenceVerification.passed(), evidenceVerification.failures().toString());
        assertTrue(evidenceVerification.evidenceSha256Matches());
        assertTrue(evidenceVerification.schemaValid());
        assertTrue(evidenceVerification.evidenceFilesSha256Match());
        assertTrue(evidenceVerification.packageArtifactsSha256Match());
        assertTrue(evidenceVerification.toMap().containsKey("inspection"));
        evidenceVerification.requirePassed();
        Path evidenceReceiptFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationEvidenceReceiptFile(
                reportDirectory);
        assertTrue(verified.evidenceReceipt() != null);
        assertEquals(evidenceReceiptFile, verified.evidenceReceipt().receiptFile());
        assertTrue(Files.isRegularFile(verified.evidenceReceipt().receiptFile()));
        assertEquals(
                TrainerCheckpointIO.sha256Hex(verified.evidenceReceipt().receiptFile()),
                verified.evidenceReceipt().receiptSha256());
        assertTrue(verified.evidenceReceipt().passed());
        String evidenceReceiptJson = Files.readString(verified.evidenceReceipt().receiptFile(), StandardCharsets.UTF_8);
        assertTrue(evidenceReceiptJson.contains(
                "\"format\":\"gollek.training.promotion.package.verification.evidence.receipt.v1\""));
        assertTrue(evidenceReceiptJson.contains("\"packageArtifactsSha256Match\":true"));
        verified.evidenceReceipt().requirePassed();
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection evidenceReceiptInspection =
                Gollek.DL.readTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile());
        assertEquals(
                TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT,
                evidenceReceiptInspection.format());
        assertEquals(verified.evidence().evidenceFile(), evidenceReceiptInspection.evidenceFile());
        assertEquals(verified.evidence().evidenceSha256(), evidenceReceiptInspection.evidenceSha256());
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification
                evidenceReceiptVerification =
                        Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                                verified.evidenceReceipt().receiptFile(),
                                verified.evidenceReceipt().receiptSha256());
        assertTrue(evidenceReceiptVerification.passed(), evidenceReceiptVerification.failures().toString());
        assertTrue(evidenceReceiptVerification.receiptSha256Matches());
        assertTrue(evidenceReceiptVerification.schemaValid());
        assertTrue(evidenceReceiptVerification.evidenceRevalidated());
        assertTrue(evidenceReceiptVerification.evidenceVerification().passed());
        evidenceReceiptVerification.requirePassed();
        Path manualEvidenceReceiptFile =
                reportDirectory.resolve("manual-promotion-gate-package-verification.evidence.receipt.json");
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt manualEvidenceReceipt =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidence().evidenceFile(),
                        verified.evidence().evidenceSha256(),
                        manualEvidenceReceiptFile);
        assertTrue(manualEvidenceReceipt.passed());
        assertEquals(manualEvidenceReceiptFile, manualEvidenceReceipt.receiptFile());
        assertTrue(Files.isRegularFile(manualEvidenceReceipt.receiptFile()));
        assertTrue(verified.toMap().containsKey("reports"));
        assertTrue(verified.toMap().containsKey("reportBundleReceipt"));
        assertTrue(verified.toMap().containsKey("index"));
        assertTrue(verified.toMap().containsKey("receipt"));
        assertTrue(verified.toMap().containsKey("packageAuditReport"));
        assertTrue(verified.toMap().containsKey("evidence"));
        assertTrue(verified.toMap().containsKey("evidenceReceipt"));
    }

    @Test
    void verificationIndexReceiptRejectsEmbeddedVerificationMismatch() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("index-receipt-embedded-mismatch-artifacts"),
                        gateResult("index-receipt-embedded-mismatch"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("index-receipt-embedded-mismatch-reports"));

        String wrongSha256 = "0".repeat(64);
        String receiptJson = Files.readString(verified.receipt().receiptFile(), StandardCharsets.UTF_8)
                .replaceFirst(
                        "\"actualIndexSha256\":\"[A-Fa-f0-9]{64}\"",
                        "\"actualIndexSha256\":\"" + wrongSha256 + "\"");
        Files.writeString(verified.receipt().receiptFile(), receiptJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        verified.receipt().receiptFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.receiptSha256Matches());
        assertTrue(verification.schemaValid(), () -> String.join("\n", verification.failures()));
        assertTrue(verification.indexRevalidated());
        assertTrue(verification.indexVerification().passed());
        assertTrue(failures.contains("embedded verification actualIndexSha256 is stale"));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void detectsVerificationIndexReferencedReportTampering() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("tamper-package-artifacts"),
                        gateResult("tamper-package"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("tamper-package-reports"));

        Files.writeString(
                verified.reports().markdown().markdownFile(),
                "\nTAMPERED\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());

        assertFalse(verification.passed());
        assertTrue(verification.indexSha256Matches());
        assertTrue(verification.schemaValid());
        assertFalse(verification.referencedSha256Match());
        assertTrue(String.join("\n", verification.failures()).contains("reports.markdown checksum mismatch"));
        assertThrows(IllegalStateException.class, verification::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification
                staleReportBundleReceiptVerification =
                        Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReportsReceipt(
                                verified.reportBundleReceipt().receiptFile(),
                                verified.reportBundleReceipt().receiptSha256());
        assertFalse(staleReportBundleReceiptVerification.passed());
        assertTrue(staleReportBundleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReportBundleReceiptVerification.schemaValid());
        assertFalse(staleReportBundleReceiptVerification.reportBundleRevalidated());
        assertFalse(staleReportBundleReceiptVerification.reportBundleVerification().markdownMatchesRendered());
        assertTrue(String.join("\n", staleReportBundleReceiptVerification.failures())
                .contains("markdown content mismatch"));
        assertThrows(IllegalStateException.class, staleReportBundleReceiptVerification::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        verified.receipt().receiptFile(),
                        verified.receipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.indexRevalidated());
        assertFalse(staleReceiptVerification.indexVerification().passed());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("reports.markdown checksum mismatch"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
        Path receiptFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexReceiptFile(
                verified.reports().directory());
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt receipt =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        receiptFile,
                        verification);
        assertFalse(receipt.passed());
        assertTrue(Files.isRegularFile(receipt.receiptFile()));
        assertTrue(Files.readString(receipt.receiptFile(), StandardCharsets.UTF_8)
                .contains("\"referencedSha256Match\":false"));
        assertThrows(IllegalStateException.class, receipt::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit audit =
                Gollek.DL.auditTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());
        assertFalse(audit.passed());
        assertFalse(audit.indexVerification().passed());
        assertTrue(audit.packageVerification().passed());
        assertTrue(String.join("\n", audit.failures()).contains("reports.markdown checksum mismatch"));
        assertThrows(IllegalStateException.class, audit::requirePassed);
    }

    @Test
    void detectsVerificationIndexReferencedReportBundleReceiptTampering() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("tamper-report-receipt-artifacts"),
                        gateResult("tamper-report-receipt"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("tamper-report-receipt-reports"));

        Files.writeString(
                verified.reportBundleReceipt().receiptFile(),
                "\nTAMPERED REPORT BUNDLE RECEIPT\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());

        assertFalse(verification.passed());
        assertTrue(verification.indexSha256Matches());
        assertTrue(verification.schemaValid());
        assertFalse(verification.referencedSha256Match());
        assertTrue(String.join("\n", verification.failures()).contains("reports.receipt checksum mismatch"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(
                        verified.receipt().receiptFile(),
                        verified.receipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.indexRevalidated());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("reports.receipt checksum mismatch"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void auditVerificationIndexDetectsPackageArtifactTampering() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("package-audit-artifacts"),
                        gateResult("package-audit"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("package-audit-reports"));

        Files.writeString(
                verified.packageBundle().artifacts().markdownFile(),
                "\nTAMPERED PACKAGE ARTIFACT\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.VerificationReportVerification staleReportVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReport(
                        verified.reports().json().reportFile(),
                        verified.reports().json().reportSha256());
        assertFalse(staleReportVerification.passed());
        assertTrue(staleReportVerification.reportSha256Matches());
        assertTrue(staleReportVerification.schemaValid());
        assertFalse(staleReportVerification.packageRevalidated());
        assertFalse(staleReportVerification.packageVerification().passed());
        assertTrue(String.join("\n", staleReportVerification.failures()).contains("artifact SHA-256 mismatch"));
        assertThrows(IllegalStateException.class, staleReportVerification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification staleReportBundleVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationReports(
                        verified.reports().directory(),
                        verified.reports().json().reportSha256());
        assertFalse(staleReportBundleVerification.passed());
        assertFalse(staleReportBundleVerification.jsonReportVerified());
        assertFalse(staleReportBundleVerification.markdownMatchesRendered());
        assertFalse(staleReportBundleVerification.junitXmlMatchesRendered());
        assertTrue(String.join("\n", staleReportBundleVerification.failures())
                .contains("artifact SHA-256 mismatch"));
        assertTrue(String.join("\n", staleReportBundleVerification.failures())
                .contains("markdown content mismatch"));
        assertTrue(String.join("\n", staleReportBundleVerification.failures())
                .contains("JUnit XML content mismatch"));
        assertThrows(IllegalStateException.class, staleReportBundleVerification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit audit =
                Gollek.DL.auditTrainingReportPromotionGatePackageVerificationIndex(
                        verified.index().indexFile(),
                        verified.index().indexSha256());

        assertFalse(audit.passed());
        assertTrue(audit.indexVerification().passed());
        assertFalse(audit.packageVerification().passed());
        assertTrue(String.join("\n", audit.failures()).contains("artifact SHA-256 mismatch"));
        assertThrows(IllegalStateException.class, audit::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification
                staleAuditReportVerification =
                        Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                                verified.packageAuditReport().reportFile(),
                                verified.packageAuditReport().reportSha256());
        assertFalse(staleAuditReportVerification.passed());
        assertTrue(staleAuditReportVerification.reportSha256Matches());
        assertTrue(staleAuditReportVerification.schemaValid());
        assertFalse(staleAuditReportVerification.auditRevalidated());
        assertFalse(staleAuditReportVerification.audit().packageVerification().passed());
        assertTrue(String.join("\n", staleAuditReportVerification.failures()).contains("artifact SHA-256 mismatch"));
        assertThrows(IllegalStateException.class, staleAuditReportVerification::requirePassed);
        Path auditReportFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexPackageAuditFile(
                verified.reports().directory());
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport auditReport =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                        auditReportFile,
                        audit);
        assertFalse(auditReport.passed());
        assertTrue(Files.isRegularFile(auditReport.reportFile()));
        assertTrue(Files.readString(auditReport.reportFile(), StandardCharsets.UTF_8)
                .contains("\"packagePassed\":false"));
        assertThrows(IllegalStateException.class, auditReport::requirePassed);
    }

    @Test
    void verificationEvidenceDetectsReportBundleReceiptTampering() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-report-receipt-tamper-artifacts"),
                        gateResult("evidence-report-receipt-tamper"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-report-receipt-tamper-reports"));

        Files.writeString(
                verified.reportBundleReceipt().receiptFile(),
                "\nTAMPERED REPORT BUNDLE RECEIPT\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile(),
                        verified.evidence().evidenceSha256());

        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertTrue(verification.schemaValid());
        assertFalse(verification.evidenceFilesSha256Match());
        assertTrue(verification.packageArtifactsSha256Match());
        assertTrue(String.join("\n", verification.failures())
                .contains("evidenceFiles.verificationReportBundleReceipt"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("evidenceFiles.verificationReportBundleReceipt"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsMalformedSourceSnapshotSummary() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-source-snapshot-schema-artifacts"),
                        gateResult("evidence-source-snapshot-schema"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-source-snapshot-schema-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replace("\"expectedSourceReportArtifacts\"", "\"expectedSourceReportArtifactsMissing\"");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertTrue(String.join("\n", verification.failures()).contains("expectedSourceReportArtifacts"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("expectedSourceReportArtifacts"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsSourceSnapshotInventoryMismatch() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-source-snapshot-inventory-artifacts"),
                        gateResult("evidence-source-snapshot-inventory"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-source-snapshot-inventory-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replaceFirst(
                        "\"presentSourceReportArtifacts\":\\[[^\\]]*\\]",
                        "\"presentSourceReportArtifacts\":[\"sourceReport.baseline.baseline\",\"sourceReport.baseline.baseline\"]");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertTrue(failures.contains("presentSourceReportArtifacts contains duplicate entry"), () -> failures);
        assertTrue(failures.contains("presentSourceReportArtifacts does not match snapshots[].snapshotArtifact"));
        assertTrue(failures.contains("missingSourceReportArtifacts does not match expected-minus-present artifacts"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("presentSourceReportArtifacts does not match snapshots[].snapshotArtifact"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsMalformedEmbeddedSourceReport() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-source-report-schema-artifacts"),
                        gateResult("evidence-source-report-schema"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-source-report-schema-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replace("\"role\":\"baseline\"", "\"roleMissing\":\"baseline\"");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertTrue(String.join("\n", verification.failures()).contains("sourceReport is missing string field role"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("sourceReport is missing string field role"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsSourceSnapshotArtifactMismatch() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-source-snapshot-artifact-mismatch-artifacts"),
                        gateResult("evidence-source-snapshot-artifact-mismatch"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-source-snapshot-artifact-mismatch-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replace(
                        "\"snapshotArtifact\":\"sourceReport.baseline.baseline\"",
                        "\"snapshotArtifact\":\"markdown\"");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertFalse(verification.packageArtifactsSha256Match());
        assertTrue(failures.contains("snapshotFile does not match packageArtifacts.markdown.file"));
        assertTrue(failures.contains("snapshotSha256 does not match packageArtifacts.markdown.sha256"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("snapshotFile does not match packageArtifacts.markdown.file"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsMalformedPackageArtifactMetadata() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-package-artifact-schema-artifacts"),
                        gateResult("evidence-package-artifact-schema"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-package-artifact-schema-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replaceFirst("(?s)(\"packageArtifacts\":\\{.*?\"bytes\":)\\d+", "$1-1")
                .replaceFirst("(?s)(\"packageArtifacts\":\\{.*?\"sha256\":\")[A-Fa-f0-9]{64}(\")",
                        "$1not-a-sha256$2");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertFalse(verification.packageArtifactsSha256Match());
        assertTrue(String.join("\n", verification.failures()).contains("negative numeric field bytes"));
        assertTrue(String.join("\n", verification.failures()).contains("invalid SHA-256 field sha256"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertFalse(staleReceiptVerification.evidenceVerification().packageArtifactsSha256Match());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("invalid SHA-256 field sha256"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsMalformedDecisionMetadata() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-decision-schema-artifacts"),
                        gateResult("evidence-decision-schema"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-decision-schema-reports"));

        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8)
                .replaceFirst("\"generatedAt\":\"[^\"]+\"", "\"generatedAt\":\"not-an-instant\"")
                .replaceFirst("\"passed\":true", "\"passed\":\"yes\"")
                .replaceFirst("\"promotable\":true", "\"promotable\":\"yes\"")
                .replaceFirst("\"decisionCandidate\":\"candidate\"", "\"decisionCandidate\":123");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertTrue(failures.contains("invalid instant field generatedAt"));
        assertTrue(failures.contains("missing boolean field passed"));
        assertTrue(failures.contains("missing boolean field promotable"));
        assertTrue(failures.contains("invalid string field decisionCandidate"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("invalid instant field generatedAt"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceRejectsReferencesOutsideDeclaredDirectories() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-reference-root-artifacts"),
                        gateResult("evidence-reference-root"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-reference-root-reports"));

        Path fakePackageDirectory = tempDir.resolve("fake-package-root").toAbsolutePath().normalize();
        Path fakeReportDirectory = tempDir.resolve("fake-report-root").toAbsolutePath().normalize();
        Files.createDirectories(fakePackageDirectory);
        Files.createDirectories(fakeReportDirectory);
        String evidenceJson = Files.readString(verified.evidence().evidenceFile(), StandardCharsets.UTF_8);
        assertTrue(evidenceJson.contains("\"packageDirectory\":\"" + verified.directory() + "\""));
        assertTrue(evidenceJson.contains("\"reportDirectory\":\"" + verified.reports().directory() + "\""));
        evidenceJson = evidenceJson
                .replace(
                        "\"packageDirectory\":\"" + verified.directory() + "\"",
                        "\"packageDirectory\":\"" + fakePackageDirectory + "\"")
                .replace(
                        "\"reportDirectory\":\"" + verified.reports().directory() + "\"",
                        "\"reportDirectory\":\"" + fakeReportDirectory + "\"");
        Files.writeString(verified.evidence().evidenceFile(), evidenceJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertFalse(verification.schemaValid());
        assertFalse(verification.evidenceFilesSha256Match());
        assertFalse(verification.packageArtifactsSha256Match());
        assertTrue(failures.contains("packageArtifacts."));
        assertTrue(failures.contains("is outside packageDirectory"));
        assertTrue(failures.contains("evidenceFiles.verificationJson.file is outside reportDirectory"));
        assertThrows(IllegalStateException.class, verification::requirePassed);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertFalse(staleReceiptVerification.evidenceVerification().schemaValid());
        assertTrue(String.join("\n", staleReceiptVerification.failures())
                .contains("is outside packageDirectory"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
    }

    @Test
    void verificationEvidenceReceiptRejectsEmbeddedVerificationMismatch() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-receipt-embedded-mismatch-artifacts"),
                        gateResult("evidence-receipt-embedded-mismatch"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-receipt-embedded-mismatch-reports"));

        String wrongSha256 = "0".repeat(64);
        String receiptJson = Files.readString(verified.evidenceReceipt().receiptFile(), StandardCharsets.UTF_8)
                .replaceFirst(
                        "\"actualEvidenceSha256\":\"[A-Fa-f0-9]{64}\"",
                        "\"actualEvidenceSha256\":\"" + wrongSha256 + "\"");
        Files.writeString(verified.evidenceReceipt().receiptFile(), receiptJson, StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile());

        String failures = String.join("\n", verification.failures());
        assertFalse(verification.passed());
        assertTrue(verification.receiptSha256Matches());
        assertTrue(verification.schemaValid());
        assertTrue(verification.evidenceRevalidated(), () -> String.join("\n", verification.failures()));
        assertTrue(verification.evidenceVerification().passed());
        assertTrue(failures.contains("embedded verification actualEvidenceSha256 is stale"));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void verificationEvidenceDetectsPackageArtifactTampering() throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("evidence-tamper-artifacts"),
                        gateResult("evidence-tamper"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"),
                        tempDir.resolve("evidence-tamper-reports"));

        Files.writeString(
                verified.packageBundle().artifacts().markdownFile(),
                "\nTAMPERED PACKAGE ARTIFACT\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                        verified.evidence().evidenceFile(),
                        verified.evidence().evidenceSha256());

        assertFalse(verification.passed());
        assertTrue(verification.evidenceSha256Matches());
        assertTrue(verification.schemaValid(), () -> String.join("\n", verification.failures()));
        assertTrue(verification.evidenceFilesSha256Match());
        assertFalse(verification.packageArtifactsSha256Match());
        assertTrue(String.join("\n", verification.failures()).contains("packageArtifacts.markdown"));
        assertThrows(IllegalStateException.class, verification::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification staleReceiptVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        verified.evidenceReceipt().receiptFile(),
                        verified.evidenceReceipt().receiptSha256());
        assertFalse(staleReceiptVerification.passed());
        assertTrue(staleReceiptVerification.receiptSha256Matches());
        assertTrue(staleReceiptVerification.schemaValid());
        assertFalse(staleReceiptVerification.evidenceRevalidated());
        assertTrue(String.join("\n", staleReceiptVerification.failures()).contains("packageArtifacts.markdown"));
        assertThrows(IllegalStateException.class, staleReceiptVerification::requirePassed);
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt receipt =
                Gollek.DL.writeTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                        tempDir.resolve("evidence-tamper-receipt.json"),
                        verification);
        assertFalse(receipt.passed());
        assertTrue(Files.isRegularFile(receipt.receiptFile()));
        assertTrue(Files.readString(receipt.receiptFile(), StandardCharsets.UTF_8)
                .contains("\"packageArtifactsSha256Match\":false"));
        assertThrows(IllegalStateException.class, receipt::requirePassed);
    }

    @Test
    void runsGateAndWritesCompletePackageInOneCall() throws IOException {
        Instant generatedAt = Instant.parse("2026-05-21T09:10:11Z");

        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("run-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("run-package-artifacts"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        generatedAt);

        assertTrue(bundle.passed());
        assertTrue(bundle.promotable());
        assertEquals(generatedAt, bundle.manifest().generatedAt());
        assertTrue(Files.isRegularFile(bundle.review().jsonFile()));
        assertTrue(Files.isRegularFile(bundle.review().markdownFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().jsonFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().markdownFile()));
        assertTrue(Files.isRegularFile(bundle.artifacts().junitXmlFile()));
        assertTrue(Files.isRegularFile(bundle.manifest().manifestFile()));
        assertEquals("candidate", bundle.artifacts().result().decision().candidate().orElseThrow().name());
        assertTrue(bundle.manifest().toMap().containsKey("artifacts"));

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.artifactVerification().junitXmlWellFormed());

        TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                Gollek.DL.runAndVerifyTrainingReportPromotionGateArtifactPackage(
                        promotionReports("run-verified-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("run-verified-package-artifacts"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        generatedAt,
                        tempDir.resolve("run-verified-package-reports"));
        assertTrue(verified.passed(), () -> verified.toMap().toString());
        assertEquals("candidate", verified.packageBundle().artifacts().result().decision().candidate().orElseThrow().name());
        assertTrue(Files.isRegularFile(verified.reports().json().reportFile()));
        assertTrue(Files.isRegularFile(verified.reports().junitXml().junitXmlFile()));
        assertTrue(Files.isRegularFile(verified.index().indexFile()));
        assertTrue(Files.isRegularFile(verified.receipt().receiptFile()));
        assertTrue(verified.receipt().passed());
        assertTrue(Files.isRegularFile(verified.packageAuditReport().reportFile()));
        assertTrue(verified.packageAuditReport().passed());
        assertTrue(Files.isRegularFile(verified.evidence().evidenceFile()));
        assertTrue(Files.readString(verified.index().indexFile(), StandardCharsets.UTF_8)
                .contains("\"decisionCandidate\":\"candidate\""));
    }

    @Test
    void runsSeverityGateAndWritesCustomPackageInOneCall() throws IOException {
        TrainingReportPromotionGateArtifactPackage.Options options =
                new TrainingReportPromotionGateArtifactPackage.Options(
                        new TrainingReportPromotionGateArtifacts.Options(
                                "severity-gate.json",
                                "severity-gate.md",
                                "severity-gate.xml"),
                        new TrainingReportPromotionGateArtifactManifest.Options("severity-gate.manifest"));

        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("severity-package"),
                        "baseline",
                        TrainingReportDiagnostics.Severity.WARNING,
                        tempDir.resolve("severity-package-artifacts"),
                        options,
                        Instant.parse("2026-05-21T09:10:11Z"));

        assertTrue(bundle.passed());
        assertTrue(bundle.artifacts().jsonFile().endsWith("severity-gate.json"));
        assertTrue(bundle.manifest().manifestFile().endsWith("severity-gate.manifest"));

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options);
        assertTrue(verification.passed());

        TrainingReportPromotionGateArtifactPackage.PackageVerification completeVerification =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options);
        assertTrue(completeVerification.passed());

        Path reportFile = tempDir.resolve("severity-package-verification.json");
        TrainingReportPromotionGateArtifactPackage.VerificationReport report =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options,
                        reportFile);
        assertTrue(report.passed());
        assertEquals(reportFile.toAbsolutePath().normalize(), report.reportFile());
        assertTrue(Files.readString(report.reportFile(), StandardCharsets.UTF_8).contains("\"passed\":true"));

        Path markdownFile = tempDir.resolve("severity-package-verification.md");
        TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport markdownReport =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options,
                        markdownFile);
        assertTrue(markdownReport.passed());
        assertEquals(markdownFile.toAbsolutePath().normalize(), markdownReport.markdownFile());
        assertTrue(Files.readString(markdownReport.markdownFile(), StandardCharsets.UTF_8)
                .contains("| Manifest checksum | match |"));

        Path junitXmlFile = tempDir.resolve("severity-package-verification.junit.xml");
        TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport junitXmlReport =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options,
                        junitXmlFile);
        assertTrue(junitXmlReport.passed());
        assertEquals(junitXmlFile.toAbsolutePath().normalize(), junitXmlReport.junitXmlFile());
        assertTrue(Files.readString(junitXmlReport.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("property name=\"manifest.checksumMatches\" value=\"true\""));

        Path reportBundleDirectory = tempDir.resolve("severity-package-verification-bundle");
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundle reportBundle =
                Gollek.DL.verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options,
                        reportBundleDirectory);
        assertTrue(reportBundle.passed());
        assertEquals(reportBundleDirectory.toAbsolutePath().normalize(), reportBundle.directory());
        assertTrue(Files.readString(reportBundle.json().reportFile(), StandardCharsets.UTF_8)
                .contains("\"passed\":true"));
        assertTrue(Files.readString(reportBundle.markdown().markdownFile(), StandardCharsets.UTF_8)
                .contains("| Manifest checksum | match |"));
        assertTrue(Files.readString(reportBundle.junitXml().junitXmlFile(), StandardCharsets.UTF_8)
                .contains("property name=\"sourceSnapshots.expected\" value=\"2\""));
    }

    @Test
    void supportsCustomPackageArtifactAndManifestNames() throws IOException {
        TrainingReportPromotionGateArtifactPackage.Options options =
                new TrainingReportPromotionGateArtifactPackage.Options(
                        new TrainingReportPromotionGateArtifacts.Options(
                                "gate-result.json",
                                "gate-result.md",
                                "gate-result.xml"),
                        new TrainingReportPromotionGateArtifactManifest.Options("gate-result.manifest"));

        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("custom-package-artifacts"),
                        gateResult("custom-package"),
                        options);

        assertTrue(bundle.artifacts().jsonFile().endsWith("gate-result.json"));
        assertTrue(bundle.artifacts().markdownFile().endsWith("gate-result.md"));
        assertTrue(bundle.artifacts().junitXmlFile().endsWith("gate-result.xml"));
        assertTrue(bundle.manifest().manifestFile().endsWith("gate-result.manifest"));

        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifactPackage(bundle.directory(), options);
        assertEquals("candidate", inspection.artifacts().decisionCandidate().orElseThrow());

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options);
        assertTrue(verification.passed());
    }

    @Test
    void supportsCustomReviewGateArtifactAndManifestNames() throws IOException {
        TrainingReportPromotionGateArtifactPackage.Options options =
                new TrainingReportPromotionGateArtifactPackage.Options(
                        new TrainingReportPromotionArtifacts.Options(
                                "review-result.json",
                                "review-result.md"),
                        new TrainingReportPromotionGateArtifacts.Options(
                                "gate-result.json",
                                "gate-result.md",
                                "gate-result.xml"),
                        new TrainingReportPromotionGateArtifactManifest.Options("gate-result.manifest"));

        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("custom-review-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("custom-review-package-artifacts"),
                        options,
                        Instant.parse("2026-05-21T09:10:11Z"));

        assertTrue(bundle.review().jsonFile().endsWith("review-result.json"));
        assertTrue(bundle.review().markdownFile().endsWith("review-result.md"));
        assertTrue(bundle.artifacts().jsonFile().endsWith("gate-result.json"));

        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifactPackage(bundle.directory(), options);
        assertEquals("candidate", inspection.review().decisionCandidate().orElseThrow());
        assertTrue(inspection.manifest().artifact("reviewJson").orElseThrow().file().endsWith("review-result.json"));
        assertTrue(inspection.manifest().artifact("reviewMarkdown").orElseThrow().file().endsWith("review-result.md"));

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(
                        bundle.directory(),
                        bundle.manifest().manifestSha256(),
                        options);
        assertTrue(verification.passed());
    }

    @Test
    void packageVerificationDetectsTamperedArtifact() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifactPackage(
                        tempDir.resolve("tampered-package-artifacts"),
                        gateResult("tampered-package"),
                        Instant.parse("2026-05-21T09:10:11Z"));

        Files.writeString(
                bundle.artifacts().jsonFile(),
                "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle);

        assertFalse(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertFalse(verification.artifactBytesMatch());
        assertFalse(verification.artifactSha256Match());
        assertFalse(verification.artifactVerification().jsonSha256Matches());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("json artifact byte size mismatch")));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void packageVerificationDetectsTamperedReviewArtifact() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("tampered-review-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("tampered-review-package-artifacts"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"));

        Files.writeString(
                bundle.review().markdownFile(),
                "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle);

        assertFalse(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertFalse(verification.artifactBytesMatch());
        assertFalse(verification.artifactSha256Match());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("reviewMarkdown artifact SHA-256 mismatch")));
    }

    @Test
    void packageVerificationDetectsTamperedSourceReportSnapshot() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("tampered-source-snapshot-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("tampered-source-snapshot-package-artifacts"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"));
        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifactPackage(bundle.directory());
        Path sourceSnapshot = inspection.manifest()
                .artifact("sourceReport.candidate.candidate")
                .orElseThrow()
                .file();

        Files.writeString(
                sourceSnapshot,
                "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle);

        assertFalse(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertFalse(verification.artifactBytesMatch());
        assertFalse(verification.artifactSha256Match());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "sourceReport.candidate.candidate artifact SHA-256 mismatch")));

        TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification sourceVerification =
                Gollek.DL.verifyTrainingReportPromotionGatePackageSourceSnapshots(inspection);
        assertFalse(sourceVerification.passed());
        assertTrue(sourceVerification.failures().stream()
                .anyMatch(failure -> failure.contains("Packaged source report snapshot SHA-256 mismatch")));

        TrainingReportPromotionGateArtifactPackage.PackageVerification completeVerification =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(bundle);
        assertFalse(completeVerification.passed());
        assertTrue(completeVerification.manifestVerification().failures().stream()
                .anyMatch(failure -> failure.contains(
                        "sourceReport.candidate.candidate artifact SHA-256 mismatch")));
        assertTrue(completeVerification.sourceSnapshotVerification().failures().stream()
                .anyMatch(failure -> failure.contains("Packaged source report snapshot SHA-256 mismatch")));
        assertThrows(IllegalStateException.class, completeVerification::requirePassed);
    }

    @Test
    void completePackageVerificationDetectsUnexpectedSourceReportSnapshot() throws IOException {
        TrainingReportPromotionGateArtifactPackage.PackageBundle bundle =
                Gollek.DL.runTrainingReportPromotionGateArtifactPackage(
                        promotionReports("unexpected-source-snapshot-package"),
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("unexpected-source-snapshot-package-artifacts"),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults(),
                        Instant.parse("2026-05-21T09:10:11Z"));
        Path shadowReport = Files.writeString(
                bundle.directory().resolve("shadow-source-report.json"),
                "{\"shadow\":true}",
                StandardCharsets.UTF_8);
        String shadowSha256 = TrainerCheckpointIO.sha256Hex(shadowReport);
        String extraSourceSnapshotEntry = "\n"
                + "artifact.sourceReport.candidate.shadow.file=" + shadowReport.getFileName() + "\n"
                + "artifact.sourceReport.candidate.shadow.bytes=" + Files.size(shadowReport) + "\n"
                + "artifact.sourceReport.candidate.shadow.sha256=" + shadowSha256 + "\n";
        Files.writeString(
                bundle.manifest().manifestFile(),
                extraSourceSnapshotEntry,
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestOnlyVerification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactPackage(bundle.directory());
        assertTrue(manifestOnlyVerification.passed());

        TrainingReportPromotionGateArtifactPackage.PackageVerification completeVerification =
                Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(bundle.directory());
        assertFalse(completeVerification.passed());
        assertTrue(completeVerification.sourceSnapshotVerification()
                .presentSourceReportArtifactNames()
                .contains("sourceReport.candidate.shadow"));
        assertEquals(
                List.of("sourceReport.candidate.shadow"),
                completeVerification.sourceSnapshotVerification().unexpectedSourceReportArtifactNames());
        assertEquals(
                List.of("sourceReport.candidate.shadow"),
                Gollek.DL.unexpectedTrainingReportPromotionGatePackageSourceArtifacts(
                        completeVerification.inspection()));
        assertTrue(completeVerification.sourceSnapshotVerification().failures().stream()
                .anyMatch(failure -> failure.contains(
                        "Unexpected packaged source report snapshot in manifest: sourceReport.candidate.shadow")));
    }

    private TrainingReportPromotionGate.Result gateResult(String prefix) throws IOException {
        return Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(prefix),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve(prefix + "-review-artifacts"));
    }

    private Map<String, Path> promotionReports(String prefix) throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                prefix + "-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                prefix + "-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        return reports;
    }

    private Path writeReport(String fileName, long durationMs, List<Map<String, Object>> epochHistory)
            throws IOException {
        Path reportFile = tempDir.resolve(fileName);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestLoss(epochHistory, "validationLoss"),
                bestEpoch(epochHistory, "validationLoss"),
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                durationMs,
                Map.of("epochHistory", epochHistory));
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-21T09:10:11Z"))),
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
            Object epoch = row.get("epoch");
            if (value instanceof Number number && number.doubleValue() < best) {
                best = number.doubleValue();
                bestEpoch = epoch instanceof Number epochNumber ? epochNumber.intValue() : bestEpoch;
            }
        }
        return bestEpoch;
    }
}
