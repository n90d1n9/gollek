package tech.kayys.gollek.cli.commands;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;
import tech.kayys.gollek.ml.train.TrainingReportComparisonArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportReader;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactPackageReport;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportValidationArtifacts;

class TrainCommandTest {
    @TempDir
    Path tempDir;

    @Test
    void picocliParsesValidateReportCommand() throws Exception {
        Path report = writeReport("picocli-report.json", true);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("validate-report", report.toString(), "--format", "summary"));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training report validation: PASS"));
    }

    @Test
    void smokeRuntimeRunsCheckpointResumeDiagnostics() throws Exception {
        Path smokeDir = tempDir.resolve("trainer-runtime-smoke");
        Path smokeArtifacts = tempDir.resolve("reports");
        Path smokeReport = smokeArtifacts.resolve("trainer-runtime-smoke.json");
        Path smokeMarkdown = smokeArtifacts.resolve("trainer-runtime-smoke.md");
        Path smokeJunit = smokeArtifacts.resolve("trainer-runtime-smoke.junit.xml");

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute(
                        "smoke-runtime",
                        "--checkpoint-dir",
                        smokeDir.toString(),
                        "--artifact-dir",
                        smokeArtifacts.toString(),
                        "--device",
                        "cpu"));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Trainer runtime smoke: PASS"));
        assertTrue(captured.out().contains("Smoke report: " + smokeReport.toAbsolutePath().normalize()));
        assertTrue(captured.out().contains("Smoke Markdown: " + smokeMarkdown.toAbsolutePath().normalize()));
        assertTrue(captured.out().contains("Smoke JUnit XML: " + smokeJunit.toAbsolutePath().normalize()));
        assertTrue(captured.out().contains("Smoke artifacts verified: PASS"));
        assertTrue(captured.out().contains("Requested device: cpu"));
        assertTrue(captured.out().contains("Execution backend: cpu"));
        assertTrue(captured.out().contains("- resumedFromCheckpoint: PASS"));
        assertTrue(Files.isRegularFile(smokeDir.resolve("canonical-runtime.state")));
        assertTrue(Files.isRegularFile(smokeDir.resolve("canonical-checkpoints.metadata")));
        String reportJson = Files.readString(smokeReport, StandardCharsets.UTF_8);
        assertTrue(reportJson.contains("\"passed\":true"));
        assertTrue(reportJson.contains("\"resumedFromCheckpoint\""));
        String markdown = Files.readString(smokeMarkdown, StandardCharsets.UTF_8);
        assertTrue(markdown.contains("# Gollek Trainer Runtime Smoke"));
        assertTrue(markdown.contains("- Status: PASS"));
        assertTrue(markdown.contains("| resumedFromCheckpoint | PASS |"));
        String junitXml = Files.readString(smokeJunit, StandardCharsets.UTF_8);
        assertTrue(junitXml.contains("<testsuite name=\"gollek.trainer.runtime.smoke\""));
        assertTrue(junitXml.contains("failures=\"0\""));
        assertTrue(junitXml.contains("name=\"resumedFromCheckpoint\""));

        Captured<Integer> verifyCaptured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-smoke-runtime-artifacts", smokeArtifacts.toString()));

        assertEquals(0, verifyCaptured.value());
        assertTrue(verifyCaptured.out().contains("Trainer runtime smoke artifact verification: PASS"));
        assertTrue(verifyCaptured.out().contains("Smoke passed: PASS"));
        assertTrue(verifyCaptured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(verifyCaptured.out().contains("JUnit XML matches JSON: PASS"));

        Captured<Integer> autoVerifyCaptured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", smokeArtifacts.toString()));

        assertEquals(0, autoVerifyCaptured.value());
        assertTrue(autoVerifyCaptured.out().contains("Training artifact verification: PASS"));
        assertTrue(autoVerifyCaptured.out().contains("Type: smoke-runtime"));
        assertTrue(autoVerifyCaptured.out().contains("Smoke passed: PASS"));
        assertTrue(autoVerifyCaptured.out().contains("Markdown matches JSON: PASS"));

        Files.writeString(smokeMarkdown, "# stale smoke report\n");
        Files.writeString(smokeJunit, "<stale></stale>\n");

        Captured<Integer> refreshCaptured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-smoke-runtime-artifacts", smokeArtifacts.toString()));

        assertEquals(0, refreshCaptured.value());
        assertTrue(refreshCaptured.out().contains("Trainer runtime smoke artifacts refreshed: PASS"));
        assertTrue(refreshCaptured.out().contains("Smoke passed: PASS"));
        assertTrue(refreshCaptured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(refreshCaptured.out().contains("JUnit XML matches JSON: PASS"));
        assertTrue(Files.readString(smokeMarkdown, StandardCharsets.UTF_8)
                .contains("# Gollek Trainer Runtime Smoke"));
        assertTrue(Files.readString(smokeJunit, StandardCharsets.UTF_8)
                .contains("gollek.trainer.runtime.smoke"));

        Files.writeString(smokeMarkdown, "# stale again\n");
        Files.writeString(smokeJunit, "<stale-again></stale-again>\n");

        Captured<Integer> autoRefreshCaptured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", smokeArtifacts.toString()));

        assertEquals(0, autoRefreshCaptured.value());
        assertTrue(autoRefreshCaptured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(autoRefreshCaptured.out().contains("Type: smoke-runtime"));
        assertTrue(autoRefreshCaptured.out().contains("Smoke passed: PASS"));
        assertTrue(autoRefreshCaptured.out().contains("Markdown matches JSON: PASS"));
    }

    @Test
    void validateReportWritesCiArtifactsAndReturnsSuccess() throws Exception {
        Path report = writeReport("report.json", true);
        Path artifacts = tempDir.resolve("artifacts");
        TrainCommand.ValidateReport command = new TrainCommand.ValidateReport();
        command.reportFile = report;
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training report validation: PASS"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_JUNIT_XML_FILE_NAME)));
    }

    @Test
    void validateReportReturnsFailureCodeForStrictValidationFailures() throws Exception {
        Path report = writeReport("missing-validation.json", false);
        TrainCommand.ValidateReport command = new TrainCommand.ValidateReport();
        command.reportFile = report;
        command.formatName = "json";

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("\"failed\" : true"));
        assertTrue(captured.out().contains("validation.missing"));
    }

    @Test
    void validateReportCanBeUsedAsAdvisoryGate() throws Exception {
        Path report = writeReport("advisory.json", false);
        TrainCommand.ValidateReport command = new TrainCommand.ValidateReport();
        command.reportFile = report;
        command.noFail = true;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training report validation: FAIL"));
        assertTrue(captured.out().contains("validation.missing"));
    }

    @Test
    void verifyValidationArtifactsChecksBundleIntegrity() throws Exception {
        Path artifacts = writeValidationArtifacts();
        TrainingReportValidationArtifacts.ArtifactInspection inspection =
                TrainingReportValidationArtifacts.read(artifacts);
        TrainCommand.VerifyValidationArtifacts command = new TrainCommand.VerifyValidationArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedJunitXmlSha256 = inspection.junitXmlSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training validation artifact verification: PASS"));
        assertTrue(captured.out().contains("JUnit XML well-formed: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
    }

    @Test
    void verifyArtifactsAutoDetectsValidationArtifacts() throws Exception {
        Path artifacts = writeValidationArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: validation"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
    }

    @Test
    void verifyValidationArtifactsFailsWhenArtifactsAreTampered() throws Exception {
        Path artifacts = writeValidationArtifacts();
        TrainingReportValidationArtifacts.ArtifactInspection inspection =
                TrainingReportValidationArtifacts.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual edit after validation\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyValidationArtifacts command = new TrainCommand.VerifyValidationArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedJunitXmlSha256 = inspection.junitXmlSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training validation artifact verification: FAIL"));
        assertTrue(captured.out().contains("Markdown checksum: FAIL"));
        assertTrue(captured.out().contains("Markdown matches JSON: FAIL"));
    }

    @Test
    void refreshValidationArtifactsRegeneratesDerivedFilesFromJson() throws Exception {
        Path artifacts = writeValidationArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.RefreshValidationArtifacts command = new TrainCommand.RefreshValidationArtifacts();
        command.artifactDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);
        TrainingReportValidationArtifacts.ArtifactVerification verification =
                TrainingReportValidationArtifacts.verify(artifacts, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training validation artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void refreshArtifactsAutoDetectsValidationArtifacts() throws Exception {
        Path artifacts = writeValidationArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportValidationArtifacts.ArtifactVerification verification =
                TrainingReportValidationArtifacts.verify(artifacts, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: validation"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportValidationArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void actionPlanPrintsReadyStatusForHealthyReport() throws Exception {
        Path report = writeReport("action-plan-ready.json", true);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("action-plan", report.toString(), "--format", "summary"));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training action plan: READY"));
        assertTrue(captured.out().contains("Recommendations: 0"));
    }

    @Test
    void compareReportsPassesWhenCandidateImproves() throws Exception {
        Path baseline = writeReport("compare-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("compare-candidate.json", 0.35, 0.4);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute(
                        "compare-reports",
                        "--baseline", baseline.toString(),
                        "--candidate", candidate.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training report comparison: PASS"));
        assertTrue(captured.out().contains("Improved metrics:"));
    }

    @Test
    void compareReportsPrintsMarkdownSummary() throws Exception {
        Path baseline = writeReport("compare-markdown-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("compare-markdown-candidate.json", 0.35, 0.4);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute(
                        "compare-reports",
                        "--baseline", baseline.toString(),
                        "--candidate", candidate.toString(),
                        "--format", "markdown"));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("# Gollek Training Report Comparison"));
        assertTrue(captured.out().contains("| Metric | Direction | Baseline | Candidate | Delta | Relative Delta | Verdict |"));
        assertTrue(captured.out().contains("`IMPROVED`"));
    }

    @Test
    void compareReportsPrintsJunitXmlSummary() throws Exception {
        Path baseline = writeReport("compare-junit-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("compare-junit-candidate.json", 0.35, 0.4);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute(
                        "compare-reports",
                        "--baseline", baseline.toString(),
                        "--candidate", candidate.toString(),
                        "--format", "junit"));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("<testsuite name=\"gollek.training.report.comparison\""));
        assertTrue(captured.out().contains("failures=\"0\""));
        assertTrue(captured.out().contains("comparison passed"));
    }

    @Test
    void compareReportsWritesArtifactsAndFailsOnRegression() throws Exception {
        Path baseline = writeReport("compare-strong-baseline.json", 0.35, 0.4);
        Path candidate = writeReport("compare-weak-candidate.json", 0.8, 0.9);
        Path artifacts = tempDir.resolve("comparison-artifacts");
        TrainCommand.CompareReports command = new TrainCommand.CompareReports();
        command.baselineReportFile = baseline;
        command.candidateReportFile = candidate;
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training report comparison: FAIL"));
        assertTrue(captured.out().contains("Regressed metrics:"));
        assertTrue(captured.out().contains("sha256="));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_MARKDOWN_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_JUNIT_XML_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_METRICS_CSV_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_FINDINGS_CSV_FILE_NAME)));
    }

    @Test
    void verifyComparisonArtifactsFailsWhenArtifactsAreTampered() throws Exception {
        Path artifacts = writeComparisonArtifacts();
        TrainingReportComparisonArtifacts.ArtifactInspection inspection =
                TrainingReportComparisonArtifacts.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_METRICS_CSV_FILE_NAME),
                "\nmanual edit after comparison export\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyComparisonArtifacts command = new TrainCommand.VerifyComparisonArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedJunitXmlSha256 = inspection.junitXmlSha256();
        command.expectedMetricsCsvSha256 = inspection.metricsCsvSha256();
        command.expectedFindingsCsvSha256 = inspection.findingsCsvSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training comparison artifact verification: FAIL"));
        assertTrue(captured.out().contains("JUnit XML well-formed: PASS"));
        assertTrue(captured.out().contains("Metrics CSV checksum: FAIL"));
        assertTrue(captured.out().contains("Metrics CSV matches JSON: FAIL"));
    }

    @Test
    void verifyComparisonArtifactsPrintsJunitXmlReport() throws Exception {
        Path artifacts = writeComparisonArtifacts();
        TrainingReportComparisonArtifacts.ArtifactInspection inspection =
                TrainingReportComparisonArtifacts.read(artifacts);
        TrainCommand.VerifyComparisonArtifacts command = new TrainCommand.VerifyComparisonArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedJunitXmlSha256 = inspection.junitXmlSha256();
        command.expectedMetricsCsvSha256 = inspection.metricsCsvSha256();
        command.expectedFindingsCsvSha256 = inspection.findingsCsvSha256();
        command.formatName = "junit";

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("<testsuite name=\"gollek.training.report.comparison.artifacts\""));
        assertTrue(captured.out().contains("failures=\"0\""));
        assertTrue(captured.out().contains("junit xml well formed"));
    }

    @Test
    void verifyArtifactsAutoDetectsComparisonArtifacts() throws Exception {
        Path artifacts = writeComparisonArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: comparison"));
        assertTrue(captured.out().contains("Metrics CSV matches JSON: PASS"));
        assertTrue(captured.out().contains("Findings CSV matches JSON: PASS"));
    }

    @Test
    void verifyComparisonArtifactsFailsWhenDerivedFilesAreStaleWithoutExpectedChecksums() throws Exception {
        Path artifacts = writeComparisonArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyComparisonArtifacts command = new TrainCommand.VerifyComparisonArtifacts();
        command.artifactDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training comparison artifact verification: FAIL"));
        assertTrue(captured.out().contains("Markdown checksum: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: FAIL"));
        assertTrue(captured.out().contains("Markdown report does not match JSON export"));
    }

    @Test
    void refreshComparisonArtifactsRegeneratesDerivedFilesFromJson() throws Exception {
        Path artifacts = writeComparisonArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_FINDINGS_CSV_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.RefreshComparisonArtifacts command = new TrainCommand.RefreshComparisonArtifacts();
        command.artifactDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);
        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                TrainingReportComparisonArtifacts.verify(artifacts, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training comparison artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("Findings CSV matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void refreshArtifactsAutoDetectsComparisonArtifacts() throws Exception {
        Path artifacts = writeComparisonArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_METRICS_CSV_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                TrainingReportComparisonArtifacts.verify(artifacts, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: comparison"));
        assertTrue(captured.out().contains("Metrics CSV matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_METRICS_CSV_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void portfolioExportWritesLeaderboardAndComparisonArtifacts() throws Exception {
        Path baseline = writeReport("portfolio-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("portfolio-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("portfolio-export");
        TrainCommand.PortfolioExport command = new TrainCommand.PortfolioExport();
        command.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        command.baselineName = "baseline";
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training portfolio export: PASS"));
        assertTrue(captured.out().contains("Comparison metrics:"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_LEADERBOARD_CSV_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPortfolioArtifacts.DEFAULT_COMPARISON_METRICS_CSV_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPortfolioArtifacts.DEFAULT_COMPARISON_FINDINGS_CSV_FILE_NAME)));
    }

    @Test
    void verifyPortfolioArtifactsFailsWhenArtifactsAreTampered() throws Exception {
        Path artifacts = writePortfolioExportArtifacts();
        TrainingReportPortfolioArtifacts.ArtifactInspection inspection =
                TrainingReportPortfolioArtifacts.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_LEADERBOARD_CSV_FILE_NAME),
                "\nmanual edit after portfolio export\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyPortfolioArtifacts command = new TrainCommand.VerifyPortfolioArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedLeaderboardCsvSha256 = inspection.leaderboardCsvSha256();
        command.expectedComparisonMetricsCsvSha256 = inspection.comparisonMetricsCsvSha256();
        command.expectedComparisonFindingsCsvSha256 = inspection.comparisonFindingsCsvSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training portfolio artifact verification: FAIL"));
        assertTrue(captured.out().contains("Leaderboard CSV checksum: FAIL"));
    }

    @Test
    void verifyArtifactsAutoDetectsPortfolioExportArtifacts() throws Exception {
        Path artifacts = writePortfolioExportArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: portfolio-export"));
        assertTrue(captured.out().contains("Leaderboard CSV checksum: PASS"));
        assertTrue(captured.out().contains("Leaderboard CSV matches JSON: PASS"));
        assertTrue(captured.out().contains("Comparison metrics CSV matches JSON: PASS"));
    }

    @Test
    void refreshArtifactsAutoDetectsPortfolioExportArtifacts() throws Exception {
        Path artifacts = writePortfolioExportArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_LEADERBOARD_CSV_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                TrainingReportPortfolioArtifacts.verify(artifacts, null, null, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: portfolio-export"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("Leaderboard CSV matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void portfolioPackageWritesManifestBackedBundle() throws Exception {
        Path baseline = writeReport("portfolio-package-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("portfolio-package-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("portfolio-package");
        TrainCommand.PortfolioPackage command = new TrainCommand.PortfolioPackage();
        command.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        command.baselineName = "baseline";
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training portfolio package: PASS"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifactManifest.DEFAULT_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_LEADERBOARD_CSV_FILE_NAME)));
    }

    @Test
    void verifyPortfolioPackageFailsWhenPackagedArtifactIsTampered() throws Exception {
        Path artifacts = writePortfolioPackageArtifacts();
        TrainingReportPortfolioArtifactPackage.PackageInspection inspection =
                TrainingReportPortfolioArtifactPackage.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual edit after portfolio package\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyPortfolioPackage command = new TrainCommand.VerifyPortfolioPackage();
        command.artifactDirectory = artifacts;
        command.expectedManifestSha256 = inspection.manifest().manifestSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training portfolio package verification: FAIL"));
        assertTrue(captured.out().contains("Artifact sha256: FAIL"));
    }

    @Test
    void verifyArtifactsPrioritizesPortfolioPackageManifest() throws Exception {
        Path artifacts = writePortfolioPackageArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: portfolio-package"));
        assertTrue(captured.out().contains("Manifest checksum: PASS"));
        assertTrue(captured.out().contains("Artifact sha256: PASS"));
    }

    @Test
    void refreshArtifactsAutoDetectsPortfolioPackageArtifacts() throws Exception {
        Path artifacts = writePortfolioPackageArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale package markdown edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_LEADERBOARD_CSV_FILE_NAME),
                "\nmanual stale package leaderboard edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                TrainingReportPortfolioArtifactPackage.verify(artifacts);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: portfolio-package"));
        assertTrue(captured.out().contains("Artifact sha256: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("Leaderboard CSV matches JSON: PASS"));
        assertTrue(captured.out().contains("Verification report: "));
        assertTrue(captured.out().contains("Verification report artifacts: PASS"));
        assertTrue(verification.passed());
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPortfolioArtifactPackageReport.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPortfolioArtifactPackageReport.DEFAULT_MARKDOWN_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPortfolioArtifactPackageReport.DEFAULT_JUNIT_XML_FILE_NAME)));
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportPortfolioArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale package markdown edit"));
    }

    @Test
    void picocliParsesPromotionGateCommand() throws Exception {
        Path baseline = writeReport("picocli-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("picocli-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("picocli-promotion");

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute(
                        "promotion-gate",
                        "--report", "baseline=" + baseline,
                        "--report", "candidate=" + candidate,
                        "--baseline", "baseline",
                        "--output-dir", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training promotion gate: PASS"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME)));
    }

    @Test
    void promotionGatePromotesCandidateAndWritesGateArtifacts() throws Exception {
        Path baseline = writeReport("baseline.json", 0.75, 0.8);
        Path candidate = writeReport("candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("promotion-artifacts");
        TrainCommand.PromotionGate command = new TrainCommand.PromotionGate();
        command.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        command.baselineName = "baseline";
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training promotion gate: PASS"));
        assertTrue(captured.out().contains("Candidate: candidate"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_JUNIT_XML_FILE_NAME)));
    }

    @Test
    void promotionGateReturnsFailureCodeWhenCandidateRegresses() throws Exception {
        Path baseline = writeReport("strong-baseline.json", 0.35, 0.4);
        Path candidate = writeReport("weak-candidate.json", 0.8, 0.9);
        TrainCommand.PromotionGate command = new TrainCommand.PromotionGate();
        command.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        command.baselineName = "baseline";
        command.outputDirectory = tempDir.resolve("promotion-hold");

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training promotion gate: FAIL"));
        assertTrue(captured.out().contains("Status: HOLD"));
        assertTrue(captured.out().contains("Candidate validation score does not beat baseline"));
    }

    @Test
    void verifyPromotionGateArtifactsFailsWhenArtifactsAreTampered() throws Exception {
        Path artifacts = writePromotionGateArtifacts();
        TrainingReportPromotionGateArtifacts.ArtifactInspection inspection =
                TrainingReportPromotionGateArtifacts.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual edit after promotion gate\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyPromotionGateArtifacts command = new TrainCommand.VerifyPromotionGateArtifacts();
        command.artifactDirectory = artifacts;
        command.expectedJsonSha256 = inspection.jsonSha256();
        command.expectedMarkdownSha256 = inspection.markdownSha256();
        command.expectedJunitXmlSha256 = inspection.junitXmlSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Promotion gate artifact verification: FAIL"));
        assertTrue(captured.out().contains("Markdown checksum: FAIL"));
        assertTrue(captured.out().contains("Markdown matches JSON: FAIL"));
    }

    @Test
    void verifyArtifactsAutoDetectsPromotionGateArtifacts() throws Exception {
        Path artifacts = writePromotionGateArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: promotion-gate"));
        assertTrue(captured.out().contains("JUnit XML well-formed: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
    }

    @Test
    void refreshArtifactsAutoDetectsPromotionGateArtifacts() throws Exception {
        Path artifacts = writePromotionGateArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_JUNIT_XML_FILE_NAME),
                "\n<!-- manual stale edit -->\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                TrainingReportPromotionGateArtifacts.verify(artifacts, null, null, null);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: promotion-gate"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
        assertTrue(verification.passed());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale edit"));
    }

    @Test
    void promotionPackageWritesAuditEvidence() throws Exception {
        Path baseline = writeReport("package-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("package-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("promotion-package");
        TrainCommand.PromotionPackage command = new TrainCommand.PromotionPackage();
        command.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        command.baselineName = "baseline";
        command.outputDirectory = artifacts;

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training promotion package: PASS"));
        assertTrue(Files.exists(artifacts.resolve(TrainingReportPromotionGateArtifactManifest.DEFAULT_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPromotionGateArtifactPackage.DEFAULT_VERIFICATION_REPORT_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPromotionGateArtifactPackage.DEFAULT_VERIFICATION_INDEX_FILE_NAME)));
        assertTrue(Files.exists(artifacts.resolve(
                TrainingReportPromotionGateArtifactPackage.DEFAULT_VERIFICATION_EVIDENCE_FILE_NAME)));
    }

    @Test
    void verifyPromotionPackageFailsWhenPackagedArtifactIsTampered() throws Exception {
        Path artifacts = writePromotionPackageArtifacts();
        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection =
                TrainingReportPromotionGateArtifactPackage.read(artifacts);
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual edit after package verification\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        TrainCommand.VerifyPromotionPackage command = new TrainCommand.VerifyPromotionPackage();
        command.artifactDirectory = artifacts;
        command.expectedManifestSha256 = inspection.manifest().manifestSha256();

        Captured<Integer> captured = captureOut(command::call);

        assertEquals(2, captured.value());
        assertTrue(captured.out().contains("Training promotion package verification: FAIL"));
        assertTrue(captured.out().contains("Artifact sha256: FAIL"));
    }

    @Test
    void verifyArtifactsPrioritizesPromotionPackageManifest() throws Exception {
        Path artifacts = writePromotionPackageArtifacts();

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("verify-artifacts", artifacts.toString()));

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifact verification: PASS"));
        assertTrue(captured.out().contains("Type: promotion-package"));
        assertTrue(captured.out().contains("Manifest checksum: PASS"));
        assertTrue(captured.out().contains("Source snapshots: PASS"));
    }

    @Test
    void refreshArtifactsAutoDetectsPromotionPackageArtifacts() throws Exception {
        Path artifacts = writePromotionPackageArtifacts();
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale package edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_JUNIT_XML_FILE_NAME),
                "\n<!-- manual stale package edit -->\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        Captured<Integer> captured = captureOut(() -> new CommandLine(new TrainCommand())
                .execute("refresh-artifacts", artifacts.toString()));
        TrainingReportPromotionGateArtifactPackage.PackageVerification verification =
                TrainingReportPromotionGateArtifactPackage.verifyComplete(artifacts);

        assertEquals(0, captured.value());
        assertTrue(captured.out().contains("Training artifacts refreshed: PASS"));
        assertTrue(captured.out().contains("Type: promotion-package"));
        assertTrue(captured.out().contains("Verification report artifacts: PASS"));
        assertTrue(captured.out().contains("Artifact sha256: PASS"));
        assertTrue(captured.out().contains("Markdown matches JSON: PASS"));
        assertTrue(captured.out().contains("JUnit XML matches JSON: PASS"));
        assertTrue(captured.out().contains("Source snapshots: PASS"));
        assertTrue(verification.passed(), verification.failures().toString());
        assertTrue(!Files.readString(
                        artifacts.resolve(TrainingReportPromotionGateArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                        StandardCharsets.UTF_8)
                .contains("manual stale package edit"));
    }

    private Path writeReport(String fileName, boolean withValidationLoss) throws Exception {
        return writeReport(fileName, 0.5, withValidationLoss ? Double.valueOf(0.55) : null);
    }

    private Path writeReport(String fileName, double trainLoss, double validationLoss) throws Exception {
        return writeReport(fileName, trainLoss, Double.valueOf(validationLoss));
    }

    private Path writeReport(String fileName, double trainLoss, Double validationLoss) throws Exception {
        Path report = tempDir.resolve(fileName);
        Files.writeString(report, canonicalReport(trainLoss, validationLoss), StandardCharsets.UTF_8);
        return report;
    }

    private Path writeValidationArtifacts() throws Exception {
        Path report = writeReport("artifact-source.json", true);
        Path artifacts = tempDir.resolve("validation-artifacts-" + System.nanoTime());
        TrainCommand.ValidateReport validate = new TrainCommand.ValidateReport();
        validate.reportFile = report;
        validate.outputDirectory = artifacts;
        assertEquals(0, captureOut(validate::call).value());
        return artifacts;
    }

    private Path writeComparisonArtifacts() throws Exception {
        Path baseline = writeReport("comparison-artifact-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("comparison-artifact-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("comparison-artifacts-" + System.nanoTime());
        TrainCommand.CompareReports compare = new TrainCommand.CompareReports();
        compare.baselineReportFile = baseline;
        compare.candidateReportFile = candidate;
        compare.outputDirectory = artifacts;
        assertEquals(0, captureOut(compare::call).value());
        return artifacts;
    }

    private Path writePortfolioExportArtifacts() throws Exception {
        Path baseline = writeReport("portfolio-artifact-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("portfolio-artifact-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("portfolio-export-" + System.nanoTime());
        TrainCommand.PortfolioExport portfolio = new TrainCommand.PortfolioExport();
        portfolio.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        portfolio.baselineName = "baseline";
        portfolio.outputDirectory = artifacts;
        assertEquals(0, captureOut(portfolio::call).value());
        return artifacts;
    }

    private Path writePortfolioPackageArtifacts() throws Exception {
        Path baseline = writeReport("portfolio-package-artifact-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("portfolio-package-artifact-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("portfolio-package-" + System.nanoTime());
        TrainCommand.PortfolioPackage portfolioPackage = new TrainCommand.PortfolioPackage();
        portfolioPackage.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        portfolioPackage.baselineName = "baseline";
        portfolioPackage.outputDirectory = artifacts;
        assertEquals(0, captureOut(portfolioPackage::call).value());
        return artifacts;
    }

    private Path writePromotionGateArtifacts() throws Exception {
        Path baseline = writeReport("artifact-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("artifact-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("promotion-artifacts-" + System.nanoTime());
        TrainCommand.PromotionGate promotionGate = new TrainCommand.PromotionGate();
        promotionGate.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        promotionGate.baselineName = "baseline";
        promotionGate.outputDirectory = artifacts;
        assertEquals(0, captureOut(promotionGate::call).value());
        return artifacts;
    }

    private Path writePromotionPackageArtifacts() throws Exception {
        Path baseline = writeReport("package-artifact-baseline.json", 0.75, 0.8);
        Path candidate = writeReport("package-artifact-candidate.json", 0.35, 0.4);
        Path artifacts = tempDir.resolve("promotion-package-" + System.nanoTime());
        TrainCommand.PromotionPackage promotionPackage = new TrainCommand.PromotionPackage();
        promotionPackage.reportSpecs = List.of("baseline=" + baseline, "candidate=" + candidate);
        promotionPackage.baselineName = "baseline";
        promotionPackage.outputDirectory = artifacts;
        assertEquals(0, captureOut(promotionPackage::call).value());
        return artifacts;
    }

    private static String canonicalReport(boolean withValidationLoss) {
        return canonicalReport(0.5, withValidationLoss ? Double.valueOf(0.55) : null);
    }

    private static String canonicalReport(double trainLoss, Double validationLoss) {
        String validationFields = validationLoss == null ? "" : String.format(
                Locale.ROOT,
                """
                  "latestValidationLoss": %.6f,
                  "bestValidationLoss": %.6f,
                """,
                validationLoss.doubleValue(),
                validationLoss.doubleValue());
        String historyValidationLoss = validationLoss == null ? "" : String.format(
                Locale.ROOT,
                ", \"validationLoss\": %.6f",
                validationLoss.doubleValue());
        return String.format(
                Locale.ROOT,
                """
                {
                  "schema": "%s",
                  "generatedAt": "2026-05-28T01:02:03Z",
                  "epochCount": 1,
                  "latestTrainLoss": %.6f,
                  %s
                  "history": [
                    { "epoch": 0, "trainLoss": %.6f%s }
                  ],
                  "historySummary": { "available": true },
                  "runHealth": {
                    "status": "healthy",
                    "healthy": true,
                    "gatePassed": true,
                    "issueDetected": false,
                    "issueCount": 0,
                    "blockingIssueDetected": false,
                    "blockingIssueCount": 0,
                    "recommendedAction": "continue monitoring training",
                    "issueCodes": [],
                    "issueSeverities": [],
                    "issueCountsByKind": {},
                    "issueCountsBySeverity": {},
                    "issues": []
                  },
                  "diagnostics": [],
                  "diagnosticsSummary": {
                    "available": true,
                    "total": 0,
                    "bySeverity": { "INFO": 0, "WARNING": 0, "CRITICAL": 0 },
                    "highestSeverity": "NONE",
                    "codes": []
                  },
                  "metadata": {}
                }
                """,
                TrainingReportReader.CANONICAL_SCHEMA,
                trainLoss,
                validationFields,
                trainLoss,
                historyValidationLoss);
    }

    private static <T> Captured<T> captureOut(ThrowingSupplier<T> supplier) throws Exception {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (PrintStream capture = new PrintStream(buffer, true, StandardCharsets.UTF_8)) {
            System.setOut(capture);
            T value = supplier.get();
            return new Captured<>(value, buffer.toString(StandardCharsets.UTF_8));
        } finally {
            System.setOut(originalOut);
        }
    }

    private record Captured<T>(T value, String out) {
    }

    @FunctionalInterface
    private interface ThrowingSupplier<T> {
        T get() throws Exception;
    }
}
