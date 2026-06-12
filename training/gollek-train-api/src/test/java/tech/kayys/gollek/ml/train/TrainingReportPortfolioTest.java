package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportPortfolioTest {
    @TempDir
    Path tempDir;

    @Test
    void ranksReportsAndSelectsBestPassingDiagnosticGate() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                120L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate-warning", writeReport(
                "candidate-warning.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.5, "validationLoss", 0.3),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.9))));
        reports.put("candidate-clean", writeReport(
                "candidate-clean.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPortfolio portfolio = Gollek.DL.trainingReportPortfolio(reports);

        assertEquals(
                List.of("candidate-warning", "candidate-clean", "baseline"),
                portfolio.rankedByValidationScore().stream().map(TrainingReportPortfolio.Entry::name).toList());
        assertEquals("candidate-warning", portfolio.bestByValidationScore().orElseThrow().name());
        assertEquals("candidate-clean", portfolio.bestPassingDiagnostics(
                TrainingReportDiagnostics.Severity.INFO).orElseThrow().name());
        assertEquals(
                List.of("candidate-clean", "baseline"),
                portfolio.rankedPassingDiagnostics(TrainingReportDiagnostics.Severity.INFO)
                        .stream()
                        .map(TrainingReportPortfolio.Entry::name)
                        .toList());

        TrainingReportPortfolio.Entry warning = portfolio.entry("candidate-warning").orElseThrow();
        assertEquals("WARNING", warning.diagnosticSummary().highestSeverity());
        assertEquals(0.3, warning.validationScore().orElseThrow(), 1e-12);
        assertTrue(warning.source().toString().endsWith("candidate-warning.json"));
        assertEquals(Files.size(reports.get("candidate-warning")), warning.sourceBytes());
        assertEquals(TrainerCheckpointIO.sha256Hex(reports.get("candidate-warning")), warning.sourceSha256());
        assertEquals(warning.sourceSha256(), warning.toMap().get("sourceSha256"));
        assertEquals(warning.sourceBytes(), warning.toMap().get("sourceBytes"));

        assertEquals(3, portfolio.toMap().get("total"));
        assertEquals("candidate-warning", portfolio.toMap().get("bestByValidationScore"));

        TrainingReportPortfolioExport export = portfolio.exportAgainst("baseline");
        assertTrue(export.available());
        assertTrue(export.hasComparisons());
        assertTrue(export.hasComparisonFindings());
        assertEquals(3, export.entryCount());
        assertEquals(16, export.comparisonMetricCount());
        assertEquals(3, export.comparisonFindingCount());
        assertEquals(TrainingReportPortfolioExport.SCHEMA, export.toMap().get("schema"));
        assertTrue(export.toJson().contains("\"entryCount\":3"));
        assertTrue(export.leaderboardCsv().startsWith(
                "rank,name,validationScore,bestValidationLoss,latestValidationLoss"));
        assertTrue(export.leaderboardCsv().contains("1,candidate-warning,0.3,0.3,0.9,0.3"));
        assertTrue(export.leaderboardCsv().contains("dataHealthStatus,dataHealthAvailable"));
        assertTrue(export.comparisonMetricsCsv().startsWith(
                "baselineReport,candidateReport,metric,direction,available"));
        assertTrue(export.comparisonMetricsCsv().contains(
                "baseline,candidate-clean,bestValidationLoss,LOWER_IS_BETTER,true,0.7,0.45"));
        assertTrue(export.comparisonFindingsCsv().contains("candidate-warning"));
        assertTrue(export.comparisonFindingsCsv().contains("comparison.diagnostics.regressed"));
        assertTrue(Gollek.DL.trainingReportPortfolioExport(reports, "baseline")
                .comparisonFindingsCsv()
                .contains("comparison.latest_validation_loss.regressed"));
        assertThrows(UnsupportedOperationException.class,
                () -> export.leaderboardRows().get(0).put("extra", "value"));
        assertThrows(UnsupportedOperationException.class,
                () -> export.comparisonMetricRows().add(Map.of("metric", "extra")));

        TrainingReportPortfolio.PromotionDecision decision = portfolio.promotionDecision(
                "baseline",
                TrainingReportDiagnostics.Severity.INFO);
        assertEquals(TrainingReportPortfolio.PromotionStatus.PROMOTE, decision.status());
        assertTrue(decision.promotable());
        assertEquals("candidate-clean", decision.candidate().orElseThrow().name());
        assertTrue(decision.reasons().isEmpty());
        assertEquals("candidate-clean", decision.toMap().get("candidate"));
        assertEquals("INFO", decision.maxAllowedDiagnosticSeverity().name());
        assertEquals(
                "INFO",
                ((Map<?, ?>) decision.toMap().get("policy")).get("maxCandidateDiagnosticSeverity"));

        TrainingReportPortfolio.PromotionDecision facadeDecision = Gollek.DL.trainingReportPromotionDecision(
                reports,
                "baseline",
                TrainingReportDiagnostics.Severity.INFO);
        assertEquals(TrainingReportPortfolio.PromotionStatus.PROMOTE, facadeDecision.status());
    }

    @Test
    void portfolioLeaderboardSurfacesDataHealthForPromotionReview() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "data-health-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "data-health-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45)),
                warningDataHealthMetadata()));

        TrainingReportPortfolio portfolio = Gollek.DL.trainingReportPortfolio(reports);
        TrainingReportPortfolio.Entry candidate = portfolio.entry("candidate").orElseThrow();
        TrainingReportPortfolioExport export = portfolio.exportAgainst("baseline");
        Map<String, Object> candidateRow = export.leaderboardRows().stream()
                .filter(row -> "candidate".equals(row.get("name")))
                .findFirst()
                .orElseThrow();

        assertEquals("warning", candidateRow.get("dataHealthStatus"));
        assertEquals(Boolean.TRUE, candidateRow.get("dataHealthAvailable"));
        assertEquals(Boolean.TRUE, candidateRow.get("dataHealthGatePassed"));
        assertEquals(1, candidateRow.get("dataHealthIssueCount"));
        assertEquals(1, candidateRow.get("dataHealthWarningCount"));
        assertEquals(0, candidateRow.get("dataHealthErrorCount"));
        assertEquals(
                "data-loader-train-drop-last-discarded-samples",
                candidateRow.get("dataHealthIssueCodes"));
        assertEquals(
                "warning",
                ((Map<?, ?>) candidate.toMap().get("dataHealth")).get("status"));
        assertTrue(export.toJson().contains("\"dataHealthStatus\":\"warning\""));
        assertTrue(export.leaderboardCsv().contains(
                "candidate,0.45,0.45,0.45,0.4,2,90,WARNING,warning,true,true,1,1,0,"
                        + "data-loader-train-drop-last-discarded-samples"));
        assertTrue(TrainingReportPortfolioMarkdown.render(export)
                .contains("| `candidate` | 0.45 | 0.45 | 0.45 | 0.4 | 2 | 90 | `WARNING` | `warning` | 1 |"));
    }

    @Test
    void writesPortfolioExportArtifactsWithChecksums() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        Path directory = tempDir.resolve("portfolio-artifacts");
        TrainingReportPortfolioArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPortfolioExportArtifacts(directory, reports, "baseline");

        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.leaderboardCsvFile()));
        assertTrue(Files.exists(bundle.comparisonMetricsCsvFile()));
        assertTrue(Files.exists(bundle.comparisonFindingsCsvFile()));
        assertTrue(bundle.hasComparisons());
        assertFalse(bundle.hasComparisonFindings());
        assertEquals(2, bundle.export().entryCount());
        assertEquals(8, bundle.export().comparisonMetricCount());
        assertEquals(TrainerCheckpointIO.sha256Hex(bundle.jsonFile()), bundle.jsonSha256());
        assertEquals(TrainerCheckpointIO.sha256Hex(bundle.markdownFile()), bundle.markdownSha256());
        assertEquals(TrainerCheckpointIO.sha256Hex(bundle.leaderboardCsvFile()), bundle.leaderboardCsvSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(bundle.comparisonMetricsCsvFile()),
                bundle.comparisonMetricsCsvSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(bundle.comparisonFindingsCsvFile()),
                bundle.comparisonFindingsCsvSha256());

        String json = Files.readString(bundle.jsonFile(), StandardCharsets.UTF_8);
        String markdown = Files.readString(bundle.markdownFile(), StandardCharsets.UTF_8);
        String leaderboardCsv = Files.readString(bundle.leaderboardCsvFile(), StandardCharsets.UTF_8);
        String comparisonMetricsCsv = Files.readString(bundle.comparisonMetricsCsvFile(), StandardCharsets.UTF_8);
        String comparisonFindingsCsv = Files.readString(bundle.comparisonFindingsCsvFile(), StandardCharsets.UTF_8);

        assertTrue(json.contains(TrainingReportPortfolioExport.SCHEMA));
        assertTrue(markdown.contains("# Gollek Training Portfolio Export"));
        assertTrue(markdown.contains("| `candidate` |"));
        assertTrue(markdown.contains("Baseline Comparison Summary"));
        assertTrue(Gollek.DL.trainingReportPortfolioMarkdown(reports, "baseline").contains("Comparison metrics"));
        assertTrue(leaderboardCsv.contains("1,candidate,0.45,0.45,0.45,0.4"));
        assertTrue(comparisonMetricsCsv.contains("baseline,candidate,bestValidationLoss"));
        assertEquals("baselineReport,candidateReport,severity,code,message,metric,evidence\n", comparisonFindingsCsv);
        assertEquals(bundle.jsonFile().toString(), bundle.toMap().get("jsonFile"));
        assertEquals(bundle.markdownFile().toString(), bundle.toMap().get("markdownFile"));
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(2, bundle.toMap().get("entryCount"));
        assertEquals(8, bundle.toMap().get("comparisonMetricCount"));

        TrainingReportPortfolioArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPortfolioExportArtifacts(directory);
        assertEquals(2, inspection.entryCount());
        assertEquals(8, inspection.comparisonMetricCount());
        assertEquals(0, inspection.comparisonFindingCount());
        assertTrue(inspection.hasComparisons());
        assertFalse(inspection.hasComparisonFindings());
        assertEquals(markdown, inspection.markdown());
        assertEquals(leaderboardCsv, inspection.leaderboardCsv());
        assertEquals(comparisonMetricsCsv, inspection.comparisonMetricsCsv());
        assertEquals(comparisonFindingsCsv, inspection.comparisonFindingsCsv());
        assertEquals(TrainingReportPortfolioExport.SCHEMA, inspection.export().get("schema"));
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertEquals(bundle.markdownSha256(), inspection.markdownSha256());
        assertEquals(bundle.leaderboardCsvSha256(), inspection.leaderboardCsvSha256());
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertEquals(inspection.artifactMap(), inspection.toMap().get("artifact"));
        assertEquals(2, inspection.toMap().get("entryCount"));
        assertTrue(Gollek.DL.trainingReportPortfolioArtifactMarkdown(bundle)
                .contains("Markdown summary"));

        TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(bundle);
        assertTrue(verification.passed());
        verification.requirePassed();
        assertTrue(verification.message().contains("verified"));
        assertEquals(Boolean.TRUE, verification.toMap().get("jsonSha256Matches"));
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.leaderboardCsvMatchesJson());
        assertTrue(verification.comparisonMetricsCsvMatchesJson());
        assertTrue(verification.comparisonFindingsCsvMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        assertTrue(Gollek.DL.trainingReportPortfolioArtifactVerificationMarkdown(verification)
                .contains("Portfolio Artifact Verification"));

        TrainingReportPortfolioArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPortfolioExportArtifactManifest(bundle);
        assertTrue(Files.exists(manifest.manifestFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(manifest.manifestFile()), manifest.manifestSha256());
        assertEquals(2, manifest.entryCount());
        assertEquals(8, manifest.comparisonMetricCount());
        assertFalse(manifest.hasComparisonFindings());
        assertEquals(bundle.markdownFile(), manifest.artifacts().markdownFile());

        TrainingReportPortfolioArtifactManifest.ManifestInspection manifestInspection =
                Gollek.DL.readTrainingReportPortfolioExportArtifactManifest(directory);
        assertEquals(TrainingReportPortfolioArtifactManifest.FORMAT_VERSION, manifestInspection.formatVersion());
        assertEquals(2, manifestInspection.entryCount());
        assertEquals(8, manifestInspection.comparisonMetricCount());
        assertTrue(manifestInspection.hasComparisons());
        assertEquals(bundle.markdownFile(), manifestInspection.artifact("markdown").orElseThrow().file());
        assertEquals(bundle.markdownSha256(), manifestInspection.artifact("markdown").orElseThrow().sha256());
        assertEquals(manifest.manifestSha256(), manifestInspection.manifestSha256());
        assertEquals(markdown, TrainingReportPortfolioArtifactManifest.readArtifacts(manifestInspection).markdown());

        TrainingReportPortfolioArtifactManifest.ManifestVerification manifestVerification =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactManifest(manifest);
        assertTrue(manifestVerification.passed());
        manifestVerification.requirePassed();
        assertTrue(manifestVerification.manifestSha256Matches());
        assertTrue(manifestVerification.artifactBytesMatch());
        assertTrue(manifestVerification.artifactSha256Match());
        assertTrue(manifestVerification.artifactVerificationOptional().orElseThrow().markdownSha256Matches());

        Path packageDirectory = tempDir.resolve("portfolio-package");
        TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle verifiedPackage =
                Gollek.DL.writeAndVerifyTrainingReportPortfolioExportArtifactPackage(
                        packageDirectory,
                        reports,
                        "baseline");
        assertTrue(verifiedPackage.passed());
        verifiedPackage.requirePassed();
        assertEquals(2, verifiedPackage.bundle().entryCount());
        assertTrue(Files.exists(verifiedPackage.bundle().artifacts().markdownFile()));
        assertTrue(Files.exists(verifiedPackage.bundle().manifest().manifestFile()));
        assertTrue(verifiedPackage.toMap().containsKey("verification"));

        TrainingReportPortfolioArtifactPackage.PackageInspection packageInspection =
                Gollek.DL.readTrainingReportPortfolioExportArtifactPackage(packageDirectory);
        assertEquals(2, packageInspection.entryCount());
        assertEquals(8, packageInspection.comparisonMetricCount());
        assertEquals(TrainingReportPortfolioExport.SCHEMA, packageInspection.artifacts().export().get("schema"));
        assertEquals(
                verifiedPackage.bundle().manifest().manifestSha256(),
                packageInspection.manifest().manifestSha256());

        TrainingReportPortfolioArtifactManifest.ManifestVerification packageVerification =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(verifiedPackage.bundle());
        assertTrue(packageVerification.passed());
        assertTrue(packageVerification.artifactVerificationOptional().orElseThrow().markdownSha256Matches());
        assertTrue(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationJson(packageVerification)
                .contains(TrainingReportPortfolioArtifactPackageReport.FORMAT));
        assertTrue(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationMarkdown(packageVerification)
                .contains("`PASS`"));
        String packageJunitXml = Gollek.DL.trainingReportPortfolioArtifactPackageVerificationJunitXml(
                packageVerification);
        assertTrue(packageJunitXml.contains("tests=\"4\""));
        assertTrue(packageJunitXml.contains("failures=\"0\""));
        assertTrue(packageJunitXml.contains("name=\"complete-package\""));
        assertWellFormedXml(packageJunitXml);

        TrainingReportPortfolioArtifactPackageReport.ReportBundle verificationReport =
                Gollek.DL.writeTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        packageVerification);
        assertTrue(verificationReport.passed());
        assertTrue(Files.exists(verificationReport.jsonFile()));
        assertTrue(Files.exists(verificationReport.markdownFile()));
        assertTrue(Files.exists(verificationReport.junitXmlFile()));
        assertEquals(TrainerCheckpointIO.sha256Hex(verificationReport.jsonFile()), verificationReport.jsonSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(verificationReport.markdownFile()),
                verificationReport.markdownSha256());
        assertEquals(
                TrainerCheckpointIO.sha256Hex(verificationReport.junitXmlFile()),
                verificationReport.junitXmlSha256());
        assertEquals(verificationReport.artifactMap(), verificationReport.toMap().get("artifact"));
        assertTrue(Files.readString(verificationReport.jsonFile(), StandardCharsets.UTF_8)
                .contains(TrainingReportPortfolioArtifactPackageReport.FORMAT));
        assertTrue(Files.readString(verificationReport.markdownFile(), StandardCharsets.UTF_8)
                .contains("Artifact checksums"));
        String reportJunitXml = Files.readString(verificationReport.junitXmlFile(), StandardCharsets.UTF_8);
        assertTrue(reportJunitXml.contains("failures=\"0\""));
        assertTrue(reportJunitXml.contains("name=\"artifact.json.sha256\""));
        assertTrue(reportJunitXml.contains("name=\"artifact.comparisonMetricsCsv.file\""));
        assertWellFormedXml(reportJunitXml);
        TrainingReportPortfolioArtifactPackageReport.ReportInspection reportInspection =
                Gollek.DL.readTrainingReportPortfolioArtifactPackageVerificationReport(packageDirectory);
        assertEquals(TrainingReportPortfolioArtifactPackageReport.FORMAT, reportInspection.format());
        assertEquals(TrainingReportPortfolioArtifactPackageReport.SCHEMA_VERSION, reportInspection.schemaVersion());
        assertEquals(
                TrainingReportPortfolioArtifactPackageReport.contentFingerprint(packageVerification),
                reportInspection.contentFingerprint());
        assertEquals(
                Gollek.DL.trainingReportPortfolioArtifactPackageContentFingerprint(packageVerification),
                reportInspection.contentFingerprint());
        assertEquals(64, reportInspection.contentFingerprint().length());
        assertTrue(reportInspection.packagePassed());
        assertEquals(0, reportInspection.failureCount());
        assertEquals(verificationReport.artifactMap(), reportInspection.artifactMap());
        assertEquals(reportInspection.artifactMap(), reportInspection.toMap().get("artifact"));
        TrainingReportPortfolioArtifactPackageReport.ReportVerification reportVerification =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(verificationReport);
        assertTrue(reportVerification.passed());
        assertTrue(reportVerification.jsonFormatMatches());
        assertTrue(reportVerification.schemaVersionMatches());
        assertTrue(reportVerification.jsonShapeValid());
        assertTrue(reportVerification.contentFingerprintConsistent());
        assertTrue(reportVerification.packageInspectionConsistent());
        assertTrue(reportVerification.artifactInventoryConsistent());
        assertTrue(reportVerification.artifactVerificationConsistent());
        assertTrue(reportVerification.markdownMatchesJson());
        assertTrue(reportVerification.junitXmlWellFormed());
        assertTrue(reportVerification.junitXmlMatchesJson());
        assertEquals(verificationReport.artifactMap(), reportVerification.artifactMap());
        assertEquals(reportVerification.artifactMap(), reportVerification.toMap().get("artifact"));
        reportVerification.requirePassed();
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency reportConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        verificationReport,
                        packageDirectory);
        assertTrue(reportConsistency.passed());
        assertTrue(reportConsistency.packagePassed());
        assertTrue(reportConsistency.gatePassed());
        assertTrue(reportConsistency.reportMatchesPackage());
        reportConsistency.requirePassed();
        reportConsistency.requireGatePassed();
        assertEquals(Boolean.TRUE, reportConsistency.toMap().get("gatePassed"));

        String packageMetricsCsv = Files.readString(
                verifiedPackage.bundle().artifacts().comparisonMetricsCsvFile(),
                StandardCharsets.UTF_8);
        Files.writeString(
                verifiedPackage.bundle().artifacts().comparisonMetricsCsvFile(),
                packageMetricsCsv + "# tampered\n",
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactManifest.ManifestVerification tamperedPackage =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(verifiedPackage.bundle());
        assertFalse(tamperedPackage.passed());
        assertFalse(tamperedPackage.artifactSha256Match());
        assertTrue(tamperedPackage.failures().stream()
                .anyMatch(failure -> failure.contains("comparisonMetricsCsv artifact")));
        assertTrue(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationMarkdown(tamperedPackage)
                .contains("`FAIL`"));
        String tamperedJunitXml = Gollek.DL.trainingReportPortfolioArtifactPackageVerificationJunitXml(
                tamperedPackage);
        assertTrue(tamperedJunitXml.contains("failures=\"3\""));
        assertTrue(tamperedJunitXml.contains("type=\"ARTIFACT_BYTES\""));
        assertTrue(tamperedJunitXml.contains("type=\"ARTIFACT_CHECKSUM\""));
        assertTrue(tamperedJunitXml.contains("comparisonMetricsCsv artifact"));
        assertWellFormedXml(tamperedJunitXml);
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency staleReportConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        verificationReport,
                        packageDirectory);
        assertFalse(staleReportConsistency.passed());
        assertFalse(staleReportConsistency.gatePassed());
        assertTrue(staleReportConsistency.reportVerification().passed());
        assertFalse(staleReportConsistency.reportMatchesPackage());
        assertTrue(staleReportConsistency.failures().stream()
                .anyMatch(failure -> failure.contains("Verification report field 'passed' is stale")));
        assertThrows(IllegalStateException.class, staleReportConsistency::requireGatePassed);
        TrainingReportPortfolioArtifactPackageReport.ReportBundle tamperedReport =
                Gollek.DL.writeTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        tamperedPackage);
        assertFalse(tamperedReport.passed());
        assertTrue(Files.readString(tamperedReport.markdownFile(), StandardCharsets.UTF_8)
                .contains("comparisonMetricsCsv artifact"));
        assertTrue(Files.readString(tamperedReport.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("failures=\"3\""));
        TrainingReportPortfolioArtifactPackageReport.ReportVerification tamperedReportVerification =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(tamperedReport);
        assertTrue(tamperedReportVerification.passed());
        assertTrue(tamperedReportVerification.jsonShapeValid());
        assertTrue(tamperedReportVerification.contentFingerprintConsistent());
        assertTrue(tamperedReportVerification.packageInspectionConsistent());
        assertTrue(tamperedReportVerification.artifactInventoryConsistent());
        assertTrue(tamperedReportVerification.artifactVerificationConsistent());
        assertTrue(tamperedReportVerification.markdownMatchesJson());
        assertTrue(tamperedReportVerification.junitXmlMatchesJson());
        assertFalse(tamperedReportVerification.inspection().packagePassed());
        assertEquals(4, tamperedReportVerification.inspection().failureCount());
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency tamperedReportConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        tamperedReport,
                        packageDirectory);
        assertTrue(tamperedReportConsistency.passed());
        assertFalse(tamperedReportConsistency.packagePassed());
        assertFalse(tamperedReportConsistency.gatePassed());
        assertTrue(tamperedReportConsistency.reportMatchesPackage());
        tamperedReportConsistency.requirePassed();
        IllegalStateException gateFailure =
                assertThrows(IllegalStateException.class, tamperedReportConsistency::requireGatePassed);
        assertTrue(gateFailure.getMessage().contains("current package did not pass"));

        String tamperedReportJson = Files.readString(tamperedReport.jsonFile(), StandardCharsets.UTF_8);
        String originalTamperedReportMarkdown =
                Files.readString(tamperedReport.markdownFile(), StandardCharsets.UTF_8);
        String tamperedReportJunitXml = Files.readString(tamperedReport.junitXmlFile(), StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace("\"hasComparisons\":true", "\"hasComparisons\":false"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(
                        "name=\"package.hasComparisons\" value=\"true\"",
                        "name=\"package.hasComparisons\" value=\"false\""),
                StandardCharsets.UTF_8);
        refreshPortfolioVerificationReportContentFingerprint(
                tamperedReport.jsonFile(),
                tamperedReport.markdownFile(),
                tamperedReport.junitXmlFile());
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency staleMetadataConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        packageDirectory,
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(staleMetadataConsistency.passed());
        assertTrue(staleMetadataConsistency.reportVerification().passed());
        assertTrue(staleMetadataConsistency.reportVerification().packageInspectionConsistent());
        assertTrue(staleMetadataConsistency.reportVerification().artifactInventoryConsistent());
        assertFalse(staleMetadataConsistency.reportMatchesPackage());
        assertTrue(staleMetadataConsistency.failures().stream()
                .anyMatch(failure -> failure.contains("Verification report field 'hasComparisons' is stale")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.markdownFile(), originalTamperedReportMarkdown, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        int originalEntryCount = tamperedPackage.inspection().entryCount();
        String originalSummaryFingerprint = Gollek.DL.readTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory)
                .contentFingerprint();
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(
                        "\"contentFingerprint\":\"" + originalSummaryFingerprint
                                + "\",\"entryCount\":" + originalEntryCount,
                        "\"contentFingerprint\":\"" + originalSummaryFingerprint
                                + "\",\"entryCount\":999"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                originalTamperedReportMarkdown.replace(
                        "Entries: `" + originalEntryCount + "`",
                        "Entries: `999`"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(
                        "name=\"package.entryCount\" value=\"" + originalEntryCount + "\"",
                        "name=\"package.entryCount\" value=\"999\""),
                StandardCharsets.UTF_8);
        refreshPortfolioVerificationReportContentFingerprint(
                tamperedReport.jsonFile(),
                tamperedReport.markdownFile(),
                tamperedReport.junitXmlFile());
        TrainingReportPortfolioArtifactPackageReport.ReportVerification mismatchedPackageInspection =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(mismatchedPackageInspection.passed());
        assertTrue(mismatchedPackageInspection.jsonShapeValid());
        assertTrue(mismatchedPackageInspection.contentFingerprintConsistent());
        assertFalse(mismatchedPackageInspection.packageInspectionConsistent());
        assertTrue(mismatchedPackageInspection.artifactInventoryConsistent());
        assertTrue(mismatchedPackageInspection.markdownMatchesJson());
        assertTrue(mismatchedPackageInspection.junitXmlMatchesJson());
        assertTrue(mismatchedPackageInspection.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "Verification report field 'entryCount' does not match package inspection field")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.markdownFile(), originalTamperedReportMarkdown, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        String actualArtifactJsonSha256 = tamperedPackage.artifactVerificationOptional()
                .orElseThrow()
                .inspection()
                .jsonSha256();
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(
                        "\"name\":\"json\",\"sha256\":\"" + actualArtifactJsonSha256 + "\"",
                        "\"name\":\"json\",\"sha256\":\"" + "4".repeat(64) + "\""),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                originalTamperedReportMarkdown.replace(
                        "`" + shortSha(actualArtifactJsonSha256) + "`",
                        "`" + shortSha("4".repeat(64)) + "`"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(actualArtifactJsonSha256, "4".repeat(64)),
                StandardCharsets.UTF_8);
        refreshPortfolioVerificationReportContentFingerprint(
                tamperedReport.jsonFile(),
                tamperedReport.markdownFile(),
                tamperedReport.junitXmlFile());
        TrainingReportPortfolioArtifactPackageReport.ReportVerification mismatchedArtifactInventory =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(mismatchedArtifactInventory.passed());
        assertTrue(mismatchedArtifactInventory.jsonShapeValid());
        assertTrue(mismatchedArtifactInventory.contentFingerprintConsistent());
        assertTrue(mismatchedArtifactInventory.packageInspectionConsistent());
        assertFalse(mismatchedArtifactInventory.artifactInventoryConsistent());
        assertTrue(mismatchedArtifactInventory.artifactVerificationConsistent());
        assertTrue(mismatchedArtifactInventory.markdownMatchesJson());
        assertTrue(mismatchedArtifactInventory.junitXmlMatchesJson());
        assertTrue(mismatchedArtifactInventory.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "artifact inventory field 'inspection.artifacts.json.sha256'")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.markdownFile(), originalTamperedReportMarkdown, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(
                        "\"actualJsonSha256\":\"" + actualArtifactJsonSha256 + "\"",
                        "\"actualJsonSha256\":\"" + "0".repeat(64) + "\""),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification inconsistentArtifactVerification =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(inconsistentArtifactVerification.passed());
        assertTrue(inconsistentArtifactVerification.jsonShapeValid());
        assertFalse(inconsistentArtifactVerification.contentFingerprintConsistent());
        assertTrue(inconsistentArtifactVerification.artifactInventoryConsistent());
        assertFalse(inconsistentArtifactVerification.artifactVerificationConsistent());
        assertTrue(inconsistentArtifactVerification.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "Artifact verification field 'actualJsonSha256' does not match inspection field")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        String internallyConsistentStaleArtifactVerificationJson = tamperedReportJson
                .replace(actualArtifactJsonSha256, "0".repeat(64));
        Files.writeString(
                tamperedReport.jsonFile(),
                internallyConsistentStaleArtifactVerificationJson,
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                originalTamperedReportMarkdown.replace(
                        "`" + shortSha(actualArtifactJsonSha256) + "`",
                        "`" + shortSha("0".repeat(64)) + "`"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(actualArtifactJsonSha256, "0".repeat(64)),
                StandardCharsets.UTF_8);
        refreshPortfolioVerificationReportContentFingerprint(
                tamperedReport.jsonFile(),
                tamperedReport.markdownFile(),
                tamperedReport.junitXmlFile());
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency staleArtifactVerificationConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        packageDirectory,
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(staleArtifactVerificationConsistency.passed());
        assertTrue(staleArtifactVerificationConsistency.reportVerification().passed());
        assertTrue(staleArtifactVerificationConsistency.reportVerification().artifactInventoryConsistent());
        assertTrue(staleArtifactVerificationConsistency.reportVerification().artifactVerificationConsistent());
        assertFalse(staleArtifactVerificationConsistency.reportMatchesPackage());
        assertTrue(staleArtifactVerificationConsistency.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "Verification report field 'artifactVerification.actualJsonSha256' is stale")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.markdownFile(), originalTamperedReportMarkdown, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        String actualInspectionLeaderboardSha256 = tamperedPackage.artifactVerificationOptional()
                .orElseThrow()
                .inspection()
                .leaderboardCsvSha256();
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(actualInspectionLeaderboardSha256, "1".repeat(64)),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                originalTamperedReportMarkdown.replace(
                        "`" + shortSha(actualInspectionLeaderboardSha256) + "`",
                        "`" + shortSha("1".repeat(64)) + "`"),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(actualInspectionLeaderboardSha256, "1".repeat(64)),
                StandardCharsets.UTF_8);
        refreshPortfolioVerificationReportContentFingerprint(
                tamperedReport.jsonFile(),
                tamperedReport.markdownFile(),
                tamperedReport.junitXmlFile());
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency staleArtifactInspectionConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        packageDirectory,
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(staleArtifactInspectionConsistency.passed());
        assertTrue(staleArtifactInspectionConsistency.reportVerification().passed());
        assertTrue(staleArtifactInspectionConsistency.reportVerification().artifactInventoryConsistent());
        assertTrue(staleArtifactInspectionConsistency.reportVerification().artifactVerificationConsistent());
        assertFalse(staleArtifactInspectionConsistency.reportMatchesPackage());
        assertTrue(staleArtifactInspectionConsistency.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "Verification report field 'artifactVerification.inspection.leaderboardCsvSha256' is stale")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.markdownFile(), originalTamperedReportMarkdown, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        String staleManifestSha256 = "3".repeat(64);
        String tamperedReportManifestSha256 = tamperedPackage.inspection().manifestSha256();
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(tamperedReportManifestSha256, staleManifestSha256),
                StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace(tamperedReportManifestSha256, staleManifestSha256),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification staleManifestFingerprint =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(staleManifestFingerprint.passed());
        assertTrue(staleManifestFingerprint.jsonShapeValid());
        assertFalse(staleManifestFingerprint.contentFingerprintConsistent());
        assertTrue(staleManifestFingerprint.markdownMatchesJson());
        assertTrue(staleManifestFingerprint.junitXmlMatchesJson());
        assertTrue(staleManifestFingerprint.failures().stream()
                .anyMatch(failure -> failure.contains("contentFingerprint")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        String originalContentFingerprint = Gollek.DL.readTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory)
                .contentFingerprint();
        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(
                        "\"contentFingerprint\":\"" + originalContentFingerprint + "\"",
                        "\"contentFingerprint\":\"" + "2".repeat(64) + "\""),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification staleContentFingerprint =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(staleContentFingerprint.passed());
        assertFalse(staleContentFingerprint.contentFingerprintConsistent());
        assertTrue(staleContentFingerprint.failures().stream()
                .anyMatch(failure -> failure.contains("contentFingerprint")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace("\"manifestSha256\":", "\"manifestSha256Missing\":"),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification malformedReportInspection =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(malformedReportInspection.passed());
        assertFalse(malformedReportInspection.jsonShapeValid());
        assertTrue(malformedReportInspection.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "verification report inspection is missing required string field 'manifestSha256'")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace("\"leaderboardCsvFile\":", "\"leaderboardCsvFileMissing\":"),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification malformedArtifactInspection =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(malformedArtifactInspection.passed());
        assertFalse(malformedArtifactInspection.jsonShapeValid());
        assertTrue(malformedArtifactInspection.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "artifact inspection is missing required string field 'leaderboardCsvFile'")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        Files.writeString(
                tamperedReport.jsonFile(),
                tamperedReportJson.replace(
                        "\"schemaVersion\":" + TrainingReportPortfolioArtifactPackageReport.SCHEMA_VERSION,
                        "\"schemaVersion\":999"),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification unsupportedSchemaVersion =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(unsupportedSchemaVersion.passed());
        assertTrue(unsupportedSchemaVersion.jsonFormatMatches());
        assertFalse(unsupportedSchemaVersion.schemaVersionMatches());
        assertTrue(unsupportedSchemaVersion.jsonShapeValid());
        assertTrue(unsupportedSchemaVersion.failures().stream()
                .anyMatch(failure -> failure.contains("schemaVersion")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        String tamperedReportMarkdown = Files.readString(tamperedReport.markdownFile(), StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                tamperedReportMarkdown.replace("**Verification:** `FAIL`", "**Verification:** `PASS`"),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification driftedMarkdown =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(driftedMarkdown.passed());
        assertTrue(driftedMarkdown.jsonShapeValid());
        assertFalse(driftedMarkdown.markdownMatchesJson());
        assertTrue(driftedMarkdown.junitXmlMatchesJson());
        assertTrue(driftedMarkdown.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown report does not match JSON field")));
        Files.writeString(tamperedReport.markdownFile(), tamperedReportMarkdown, StandardCharsets.UTF_8);

        String markdownArtifactTableDrift = Files.readString(tamperedReport.markdownFile(), StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.markdownFile(),
                markdownArtifactTableDrift.replace(
                        "`" + shortSha(actualArtifactJsonSha256) + "`",
                        "`" + shortSha("5".repeat(64)) + "`"),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification driftedMarkdownArtifact =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(driftedMarkdownArtifact.passed());
        assertTrue(driftedMarkdownArtifact.jsonShapeValid());
        assertFalse(driftedMarkdownArtifact.markdownMatchesJson());
        assertTrue(driftedMarkdownArtifact.junitXmlMatchesJson());
        assertTrue(driftedMarkdownArtifact.failures().stream()
                .anyMatch(failure -> failure.contains("artifact json row")));
        Files.writeString(tamperedReport.markdownFile(), markdownArtifactTableDrift, StandardCharsets.UTF_8);

        tamperedReportJunitXml = Files.readString(tamperedReport.junitXmlFile(), StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                tamperedReportJunitXml.replace("failures=\"3\"", "failures=\"0\""),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification driftedJunitXml =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(driftedJunitXml.passed());
        assertTrue(driftedJunitXml.jsonShapeValid());
        assertTrue(driftedJunitXml.markdownMatchesJson());
        assertTrue(driftedJunitXml.junitXmlWellFormed());
        assertFalse(driftedJunitXml.junitXmlMatchesJson());
        assertTrue(driftedJunitXml.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML report does not match JSON field 'failures'")));
        Files.writeString(tamperedReport.junitXmlFile(), tamperedReportJunitXml, StandardCharsets.UTF_8);

        String artifactJunitXmlDrift = Files.readString(tamperedReport.junitXmlFile(), StandardCharsets.UTF_8);
        Files.writeString(
                tamperedReport.junitXmlFile(),
                artifactJunitXmlDrift.replace(actualArtifactJsonSha256, "6".repeat(64)),
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification driftedJunitXmlArtifact =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(driftedJunitXmlArtifact.passed());
        assertTrue(driftedJunitXmlArtifact.jsonShapeValid());
        assertTrue(driftedJunitXmlArtifact.markdownMatchesJson());
        assertTrue(driftedJunitXmlArtifact.junitXmlWellFormed());
        assertFalse(driftedJunitXmlArtifact.junitXmlMatchesJson());
        assertTrue(driftedJunitXmlArtifact.failures().stream()
                .anyMatch(failure -> failure.contains(
                        "JUnit XML report does not match JSON field 'artifact.json.sha256'")));
        Files.writeString(tamperedReport.junitXmlFile(), artifactJunitXmlDrift, StandardCharsets.UTF_8);

        Files.writeString(
                tamperedReport.jsonFile(),
                "{\"format\":\"" + TrainingReportPortfolioArtifactPackageReport.FORMAT + "\"}\n",
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification malformedJsonShape =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                        packageDirectory,
                        null,
                        null,
                        null);
        assertFalse(malformedJsonShape.passed());
        assertTrue(malformedJsonShape.jsonFormatMatches());
        assertFalse(malformedJsonShape.jsonShapeValid());
        assertTrue(malformedJsonShape.failures().stream()
                .anyMatch(failure -> failure.contains("boolean field 'passed'")));
        Files.writeString(tamperedReport.jsonFile(), tamperedReportJson, StandardCharsets.UTF_8);

        Files.writeString(tamperedReport.junitXmlFile(), "<testsuite>", StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification corruptedReportVerification =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(tamperedReport);
        assertFalse(corruptedReportVerification.passed());
        assertFalse(corruptedReportVerification.junitXmlSha256Matches());
        assertTrue(corruptedReportVerification.jsonShapeValid());
        assertTrue(corruptedReportVerification.artifactVerificationConsistent());
        assertTrue(corruptedReportVerification.markdownMatchesJson());
        assertFalse(corruptedReportVerification.junitXmlWellFormed());
        assertFalse(corruptedReportVerification.junitXmlMatchesJson());
        assertTrue(corruptedReportVerification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML is not well-formed")));
        assertThrows(IllegalStateException.class, corruptedReportVerification::requirePassed);

        Files.writeString(
                bundle.leaderboardCsvFile(),
                leaderboardCsv + "# tampered\n",
                StandardCharsets.UTF_8);
        TrainingReportPortfolioArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(bundle);
        assertFalse(tampered.passed());
        assertFalse(tampered.leaderboardCsvSha256Matches());
        assertTrue(tampered.failures().stream()
                .anyMatch(failure -> failure.contains("Leaderboard CSV checksum mismatch")));
        assertThrows(IllegalStateException.class, tampered::requirePassed);
        TrainingReportPortfolioArtifactManifest.ManifestVerification tamperedManifest =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactManifest(manifest);
        assertFalse(tamperedManifest.passed());
        assertFalse(tamperedManifest.artifactBytesMatch());
        assertFalse(tamperedManifest.artifactSha256Match());
        assertTrue(tamperedManifest.failures().stream()
                .anyMatch(failure -> failure.contains("leaderboardCsv artifact")));

        TrainingReportPortfolio portfolio = Gollek.DL.trainingReportPortfolio(reports);
        assertThrows(IllegalArgumentException.class, () -> TrainingReportPortfolioArtifacts.write(
                tempDir.resolve("bad-portfolio-artifacts"),
                portfolio,
                new TrainingReportPortfolioArtifacts.Options("../bad.json", null, null, null)));
    }

    @Test
    void refreshPortfolioExportArtifactsRebuildsDerivedFilesFromJsonExport() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "refresh-portfolio-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "refresh-portfolio-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        Path directory = tempDir.resolve("refresh-portfolio-artifacts");
        TrainingReportPortfolioArtifacts.ArtifactBundle original =
                Gollek.DL.writeTrainingReportPortfolioExportArtifacts(directory, reports, "baseline");
        Files.writeString(
                original.markdownFile(),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                original.leaderboardCsvFile(),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainingReportPortfolioArtifacts.ArtifactVerification stale =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(directory, null, null, null, null, null);

        assertFalse(stale.passed());
        assertFalse(stale.markdownMatchesJson());
        assertFalse(stale.leaderboardCsvMatchesJson());

        TrainingReportPortfolioArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportPortfolioExportArtifacts(directory);
        TrainingReportPortfolioArtifacts.ArtifactVerification refreshedVerification =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(refreshed);

        assertEquals(original.jsonSha256(), refreshed.jsonSha256());
        assertFalse(Files.readString(refreshed.markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertFalse(Files.readString(refreshed.leaderboardCsvFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertTrue(refreshedVerification.passed());
        assertTrue(refreshedVerification.markdownMatchesJson());
        assertTrue(refreshedVerification.leaderboardCsvMatchesJson());
        assertTrue(refreshedVerification.comparisonMetricsCsvMatchesJson());
        assertTrue(refreshedVerification.comparisonFindingsCsvMatchesJson());
    }

    @Test
    void refreshPortfolioPackageRebuildsDerivedArtifactsAndManifest() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "refresh-portfolio-package-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "refresh-portfolio-package-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPortfolioArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPortfolioExportArtifactPackage(
                        tempDir.resolve("refresh-portfolio-package"),
                        reports,
                        "baseline");
        String originalManifestSha256 = bundle.manifest().manifestSha256();
        TrainingReportPortfolioArtifactPackageReport.ReportBundle originalReport =
                Gollek.DL.writeTrainingReportPortfolioArtifactPackageVerificationReport(
                        bundle.directory(),
                        Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(bundle.directory()));

        Files.writeString(
                bundle.artifacts().markdownFile(),
                "\nmanual stale package markdown edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                bundle.artifacts().leaderboardCsvFile(),
                "\nmanual stale package leaderboard edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                originalReport.markdownFile(),
                "\nmanual stale package verification report edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainingReportPortfolioArtifactManifest.ManifestVerification stale =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(bundle.directory());

        assertFalse(stale.passed());
        assertFalse(stale.artifactSha256Match());
        assertFalse(stale.artifactVerification().markdownMatchesJson());
        assertFalse(stale.artifactVerification().leaderboardCsvMatchesJson());

        TrainingReportPortfolioArtifactPackage.PackageRefresh refresh =
                Gollek.DL.refreshTrainingReportPortfolioExportArtifactPackage(bundle.directory());
        TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(bundle.directory());

        assertTrue(refresh.passed(), () -> refresh.toMap().toString());
        assertTrue(verification.passed(), () -> verification.failures().toString());
        assertTrue(verification.artifactSha256Match());
        assertTrue(verification.artifactVerification().markdownMatchesJson());
        assertTrue(verification.artifactVerification().leaderboardCsvMatchesJson());
        assertTrue(verification.artifactVerification().comparisonMetricsCsvMatchesJson());
        assertTrue(verification.artifactVerification().comparisonFindingsCsvMatchesJson());
        assertTrue(!originalManifestSha256.equals(refresh.manifest().manifestSha256()));
        assertTrue(Files.isRegularFile(refresh.reports().jsonFile()));
        assertTrue(Files.isRegularFile(refresh.reports().markdownFile()));
        assertTrue(Files.isRegularFile(refresh.reports().junitXmlFile()));
        assertTrue(refresh.reportVerification().passed(), () -> refresh.reportVerification().failures().toString());
        TrainingReportPortfolioArtifactPackageReport.ReportVerification reportVerification =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReport(refresh.reports());
        assertTrue(reportVerification.passed(), () -> reportVerification.failures().toString());
        TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency reportConsistency =
                Gollek.DL.verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                        refresh.reports(),
                        bundle.directory());
        assertTrue(reportConsistency.gatePassed(), () -> reportConsistency.failures().toString());
        assertTrue(!Files.readString(bundle.artifacts().markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale package markdown edit"));
        assertTrue(!Files.readString(bundle.artifacts().leaderboardCsvFile(), StandardCharsets.UTF_8)
                .contains("manual stale package leaderboard edit"));
        assertTrue(!Files.readString(refresh.reports().markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale package verification report edit"));
    }

    @Test
    void refreshPortfolioPackageRejectsChangedNonRefreshableArtifact() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "refresh-portfolio-package-reject-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "refresh-portfolio-package-reject-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        TrainingReportPortfolioArtifactPackage.PackageBundle bundle =
                Gollek.DL.writeTrainingReportPortfolioExportArtifactPackage(
                        tempDir.resolve("refresh-portfolio-package-reject"),
                        reports,
                        "baseline");

        Files.writeString(
                bundle.artifacts().jsonFile(),
                "\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        IOException error = assertThrows(
                IOException.class,
                () -> Gollek.DL.refreshTrainingReportPortfolioExportArtifactPackage(bundle.directory()));
        assertTrue(error.getMessage().contains("non-refreshable artifact"));
    }

    @Test
    void promotionPolicyCanRequireLargerValidationImprovement() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                120L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        TrainingReportPortfolio.PromotionPolicy strictPolicy =
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMinimumValidationImprovement(0.30);

        TrainingReportPortfolio.PromotionDecision decision = Gollek.DL.trainingReportPromotionDecision(
                reports,
                "baseline",
                strictPolicy);

        assertEquals(TrainingReportPortfolio.PromotionStatus.HOLD, decision.status());
        assertFalse(decision.promotable());
        assertEquals(0.30, decision.policy().minimumValidationImprovement(), 1e-12);
        assertTrue(decision.reasons().stream()
                .anyMatch(reason -> reason.contains("by at least 0.3")));
    }

    @Test
    void promotionPolicyCanAllowWarningCandidatesWhenExplicitlyRequested() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                120L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate-warning", writeReport(
                "candidate-warning.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.5, "validationLoss", 0.3),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.9))));
        TrainingReportPortfolio.PromotionPolicy warningTolerantPolicy =
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withMaxComparisonFindingSeverity(TrainingReportDiagnostics.Severity.WARNING);

        TrainingReportPortfolio.PromotionDecision decision = Gollek.DL.trainingReportPromotionDecision(
                reports,
                "baseline",
                warningTolerantPolicy);

        assertEquals(TrainingReportPortfolio.PromotionStatus.PROMOTE, decision.status());
        assertEquals("candidate-warning", decision.candidate().orElseThrow().name());
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, decision.policy().maxCandidateDiagnosticSeverity());
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, decision.policy().maxComparisonFindingSeverity());
    }

    @Test
    void promotionSkipsTopCandidateWhenComparisonPolicyFails() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate-slow", writeReport(
                "candidate-slow.json",
                200L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.6, "validationLoss", 0.5),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.3))));
        reports.put("candidate-clean", writeReport(
                "candidate-clean.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPortfolio portfolio = Gollek.DL.trainingReportPortfolio(reports);
        assertEquals("candidate-slow", portfolio.rankedPassingDiagnostics(
                TrainingReportDiagnostics.Severity.INFO).get(0).name());

        TrainingReportPortfolio.PromotionDecision decision = portfolio.promotionDecision(
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy());

        assertEquals(TrainingReportPortfolio.PromotionStatus.PROMOTE, decision.status());
        assertEquals("candidate-clean", decision.candidate().orElseThrow().name());
        assertTrue(portfolio.compare("baseline", "candidate-slow").findings().stream()
                .anyMatch(finding -> finding.code().equals("comparison.duration.regressed")));
    }

    @Test
    void promotionReviewAuditsEveryCandidateAndSelectsFirstPromotable() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate-warning", writeReport(
                "candidate-warning.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.5, "validationLoss", 0.2),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.9))));
        reports.put("candidate-slow", writeReport(
                "candidate-slow.json",
                200L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.6, "validationLoss", 0.5),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.3))));
        reports.put("candidate-clean", writeReport(
                "candidate-clean.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPortfolio.PromotionReview review = Gollek.DL.trainingReportPromotionReview(
                reports,
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy());

        assertEquals(
                List.of("candidate-warning", "candidate-slow", "candidate-clean"),
                review.candidates().stream()
                        .map(candidate -> candidate.candidate().name())
                        .toList());
        assertEquals(2, review.heldCandidates().size());
        assertEquals(1, review.promotableCandidates().size());
        assertEquals("candidate-clean", review.promotableCandidates().get(0).candidate().name());
        assertEquals("candidate-clean", review.decision().candidate().orElseThrow().name());

        TrainingReportPortfolio.PromotionCandidateReview warning =
                review.candidate("candidate-warning").orElseThrow();
        assertFalse(warning.diagnosticsPassed());
        assertTrue(warning.reasons().stream()
                .anyMatch(reason -> reason.contains("diagnostics exceed INFO")));

        TrainingReportPortfolio.PromotionCandidateReview slow =
                review.candidate("candidate-slow").orElseThrow();
        assertTrue(slow.diagnosticsPassed());
        assertFalse(slow.promotable());
        assertTrue(slow.reasons().stream()
                .anyMatch(reason -> reason.contains("comparison.duration.regressed")));

        Map<?, ?> reviewMap = review.toMap();
        assertEquals(3, reviewMap.get("candidateCount"));
        assertEquals(1, reviewMap.get("promotableCount"));
        assertEquals(
                "candidate-clean",
                ((Map<?, ?>) reviewMap.get("decision")).get("candidate"));
    }

    @Test
    void comparesNamedReportsAgainstBaseline() throws IOException {
        Map<String, TrainingReport> reports = new LinkedHashMap<>();
        reports.put("baseline", TrainingReportReader.readReport(writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)))));
        reports.put("candidate", TrainingReportReader.readReport(writeReport(
                "candidate.json",
                105L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55)))));

        TrainingReportPortfolio portfolio = Gollek.DL.trainingReportPortfolioFromReports(reports);

        assertEquals("candidate", portfolio.bestByValidationScore().orElseThrow().name());
        assertTrue(portfolio.compare("baseline", "candidate").improvedMetrics().stream()
                .anyMatch(metric -> metric.name().equals("bestValidationLoss")));
        assertEquals(1, portfolio.compareAllAgainst("baseline").size());
        assertEquals(2, Gollek.DL.trainingReportPortfolioExport(portfolio).entryCount());
        assertEquals(8, Gollek.DL.trainingReportPortfolioExport(portfolio, "baseline").comparisonMetricCount());
        assertTrue(portfolio.export().comparisonMetricRows().isEmpty());
        assertTrue(portfolio.exportAgainst("baseline")
                .comparisonMetricsCsv()
                .contains("baseline,candidate,bestValidationLoss"));
        assertThrows(IllegalArgumentException.class, () -> portfolio.compare("missing", "candidate"));
        assertThrows(IllegalArgumentException.class, () -> portfolio.exportAgainst("missing"));

        TrainingReportPortfolio.Entry candidate = portfolio.entry("candidate").orElseThrow();
        assertNull(candidate.source());
        assertNull(candidate.sourceBytes());
        assertNull(candidate.sourceSha256());
        assertFalse(candidate.toMap().containsKey("sourceSha256"));
    }

    @Test
    void holdsPromotionWhenEligibleCandidateDoesNotBeatBaseline() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.6, "validationLoss", 0.65),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.35))));
        reports.put("candidate", writeReport(
                "candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.75),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.6))));

        TrainingReportPortfolio.PromotionDecision decision = Gollek.DL.trainingReportPortfolio(reports)
                .promotionDecision("baseline", TrainingReportDiagnostics.Severity.INFO);

        assertEquals(TrainingReportPortfolio.PromotionStatus.HOLD, decision.status());
        assertFalse(decision.promotable());
        assertEquals("candidate", decision.candidate().orElseThrow().name());
        assertTrue(decision.reasons().stream()
                .anyMatch(reason -> reason.contains("validation score does not beat")));
        assertThrows(IllegalStateException.class, decision::requirePromotable);
    }

    @Test
    void reportsNoEligibleCandidateWhenOnlyBaselinePassesGate() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.6))));

        TrainingReportPortfolio.PromotionDecision decision = Gollek.DL.trainingReportPortfolio(reports)
                .promotionDecision("baseline", TrainingReportDiagnostics.Severity.INFO);

        assertEquals(TrainingReportPortfolio.PromotionStatus.NO_ELIGIBLE_CANDIDATE, decision.status());
        assertTrue(decision.candidate().isEmpty());
        assertTrue(decision.comparison().isEmpty());
        assertTrue(decision.message().contains("No candidate passed"));
    }

    private Path writeReport(String fileName, long durationMs, List<Map<String, Object>> epochHistory)
            throws IOException {
        return writeReport(fileName, durationMs, epochHistory, Map.of());
    }

    private Path writeReport(
            String fileName,
            long durationMs,
            List<Map<String, Object>> epochHistory,
            Map<String, Object> extraMetadata)
            throws IOException {
        Path reportFile = tempDir.resolve(fileName);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("epochHistory", epochHistory);
        metadata.putAll(extraMetadata);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestLoss(epochHistory, "validationLoss"),
                bestEpoch(epochHistory, "validationLoss"),
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                durationMs,
                metadata);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-19T09:10:11Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
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

    private static void refreshPortfolioVerificationReportContentFingerprint(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile) throws IOException {
        TrainingReportPortfolioArtifactPackageReport.ReportInspection inspection =
                TrainingReportPortfolioArtifactPackageReport.readFiles(jsonFile, markdownFile, junitXmlFile);
        String oldFingerprint = inspection.contentFingerprint();
        String refreshedFingerprint = TrainingReportPortfolioArtifactPackageReport.contentFingerprint(inspection);
        Files.writeString(
                jsonFile,
                Files.readString(jsonFile, StandardCharsets.UTF_8).replace(
                        "\"contentFingerprint\":\"" + oldFingerprint + "\"",
                        "\"contentFingerprint\":\"" + refreshedFingerprint + "\""),
                StandardCharsets.UTF_8);
        Files.writeString(
                markdownFile,
                Files.readString(markdownFile, StandardCharsets.UTF_8).replace(
                        "Content fingerprint: `" + oldFingerprint + "`",
                        "Content fingerprint: `" + refreshedFingerprint + "`"),
                StandardCharsets.UTF_8);
        Files.writeString(
                junitXmlFile,
                Files.readString(junitXmlFile, StandardCharsets.UTF_8).replace(
                        "name=\"package.contentFingerprint\" value=\"" + oldFingerprint + "\"",
                        "name=\"package.contentFingerprint\" value=\"" + refreshedFingerprint + "\""),
                StandardCharsets.UTF_8);
    }

    private static String shortSha(String sha256) {
        return sha256.length() <= 12 ? sha256 : sha256.substring(0, 12);
    }

    private static void assertWellFormedXml(String xml) {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
            factory.newDocumentBuilder().parse(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception exception) {
            throw new AssertionError("Expected well-formed XML", exception);
        }
    }
}
