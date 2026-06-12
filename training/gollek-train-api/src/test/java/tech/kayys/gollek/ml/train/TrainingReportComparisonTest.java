package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportComparisonTest {
    @TempDir
    Path tempDir;

    @Test
    void flagsCandidateLossAndDurationRegressions() throws IOException {
        Path baselineFile = writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.5)));
        Path candidateFile = writeReport(
                "candidate.json",
                180L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.65, "validationLoss", 0.75)));

        TrainingReportComparison comparison = Gollek.DL.compareTrainingReports(baselineFile, candidateFile);

        assertTrue(hasCode(comparison.findings(), "comparison.best_validation_loss.regressed"));
        assertTrue(hasCode(comparison.findings(), "comparison.latest_validation_loss.regressed"));
        assertTrue(hasCode(comparison.findings(), "comparison.latest_train_loss.regressed"));
        assertTrue(hasCode(comparison.findings(), "comparison.duration.regressed"));
        assertEquals("WARNING", comparison.summary().highestSeverity());
        assertEquals(4, comparison.regressedMetrics().size());

        TrainingReportComparison.MetricDelta bestValidation = comparison.metric("bestValidationLoss").orElseThrow();
        assertEquals(0.5, bestValidation.baselineValue().orElseThrow(), 1e-12);
        assertEquals(0.75, bestValidation.candidateValue().orElseThrow(), 1e-12);
        assertEquals(0.25, bestValidation.absoluteDelta().orElseThrow(), 1e-12);
        assertEquals(0.5, bestValidation.relativeDelta().orElseThrow(), 1e-12);

        TrainingReportDiagnostics.GateResult gate = comparison.gate(TrainingReportDiagnostics.Severity.INFO);
        assertFalse(gate.passed());
        assertTrue(gate.failingCodes().contains("comparison.duration.regressed"));
        assertThrows(IllegalStateException.class, gate::requirePassed);
        assertEquals(8, ((List<?>) comparison.toMap().get("metrics")).size());

        TrainingReportComparisonExport export = comparison.export();
        assertTrue(export.available());
        assertEquals(8, export.metricCount());
        assertEquals(comparison.findings().size(), export.findingCount());
        assertTrue(export.hasFindings());
        assertEquals(TrainingReportComparisonExport.SCHEMA, export.toMap().get("schema"));
        assertTrue(export.toJson().contains("\"metricCount\":8"));
        assertTrue(export.metricsCsv().startsWith(
                "metric,direction,available,baseline,candidate,absoluteDelta,relativeDelta,verdict"));
        assertTrue(export.metricsCsv().contains(
                "bestValidationLoss,LOWER_IS_BETTER,true,0.5,0.75,0.25,0.5,REGRESSED"));
        assertTrue(export.findingsCsv().contains("comparison.duration.regressed"));
        assertThrows(UnsupportedOperationException.class,
                () -> export.metricRows().get(0).put("extra", "value"));
        assertThrows(UnsupportedOperationException.class,
                () -> export.findingRows().add(Map.of("code", "extra")));
        String markdown = Gollek.DL.trainingReportComparisonMarkdown(comparison);
        assertTrue(markdown.contains("# Gollek Training Report Comparison"));
        assertTrue(markdown.contains("`REGRESSED`"));
        assertTrue(markdown.contains("comparison.duration.regressed"));
        String junitXml = Gollek.DL.trainingReportComparisonJUnitXml(comparison);
        assertTrue(junitXml.contains("<testsuite name=\"gollek.training.report.comparison\""));
        assertTrue(junitXml.contains("failures=\"4\""));
        assertTrue(junitXml.contains("comparison.duration.regressed"));

        TrainingReportComparisonExport facadeExport =
                Gollek.DL.trainingReportComparisonExport(baselineFile, candidateFile);
        assertEquals(export.metricCount(), facadeExport.metricCount());
        assertEquals(export.findingCount(), facadeExport.findingCount());
    }

    @Test
    void reportsCleanCandidateImprovementsWithoutWarnings() throws IOException {
        TrainingReport baseline = TrainingReportReader.readReport(writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        TrainingReport candidate = TrainingReportReader.readReport(writeReport(
                "candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55))));

        TrainingReportComparison comparison = baseline.compareCandidate(candidate);

        assertFalse(comparison.hasFindings());
        assertEquals("NONE", comparison.summary().highestSeverity());
        assertTrue(comparison.gate(TrainingReportDiagnostics.Severity.INFO).passed());
        assertTrue(comparison.improvedMetrics().stream()
                .anyMatch(metric -> metric.name().equals("bestValidationLoss")));
        assertTrue(comparison.improvedMetrics().stream()
                .anyMatch(metric -> metric.name().equals("latestTrainLoss")));
        assertTrue(comparison.regressedMetrics().isEmpty());
        assertTrue(comparison.export().metricsCsv().contains("bestValidationLoss,LOWER_IS_BETTER,true"));
        assertTrue(comparison.export().metricsCsv().contains("IMPROVED"));
        assertEquals("severity,code,message,metric,evidence\n", comparison.export().findingsCsv());
        assertTrue(Gollek.DL.trainingReportComparisonMarkdown(comparison.export())
                .contains("No regression findings were detected."));
        assertTrue(Gollek.DL.trainingReportComparisonJUnitXml(comparison.export())
                .contains("name=\"comparison passed\""));
        assertEquals(8, Gollek.DL.trainingReportComparisonExport(baseline, candidate).metricCount());
    }

    @Test
    void writesMarkdownBackedComparisonArtifacts() throws IOException {
        Path baselineFile = writeReport(
                "artifact-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)));
        Path candidateFile = writeReport(
                "artifact-candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55)));
        Path artifacts = tempDir.resolve("comparison-artifacts");

        TrainingReportComparisonArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportComparisonArtifacts(artifacts, baselineFile, candidateFile);
        TrainingReportComparisonArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportComparisonArtifacts(artifacts);
        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportComparisonArtifacts(bundle);

        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertTrue(Files.readString(bundle.markdownFile(), StandardCharsets.UTF_8)
                .contains("# Gollek Training Report Comparison"));
        assertTrue(Files.readString(bundle.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("<testsuite name=\"gollek.training.report.comparison\""));
        assertEquals(bundle.markdownSha256(), inspection.markdownSha256());
        assertEquals(bundle.junitXmlSha256(), inspection.junitXmlSha256());
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertEquals(bundle.metricsCsvFile().toString(), bundle.toMap().get("metricsCsvFile"));
        assertEquals(bundle.findingsCsvFile().toString(), inspection.toMap().get("findingsCsvFile"));
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.junitXmlSha256Matches());
        assertTrue(verification.junitXmlWellFormed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertTrue(verification.metricsCsvMatchesJson());
        assertTrue(verification.findingsCsvMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        assertTrue(verification.passed());
    }

    @Test
    void verificationFailsWhenDerivedArtifactsDriftFromJsonExport() throws IOException {
        Path baselineFile = writeReport(
                "stale-artifact-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)));
        Path candidateFile = writeReport(
                "stale-artifact-candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55)));
        Path artifacts = tempDir.resolve("stale-comparison-artifacts");

        Gollek.DL.writeTrainingReportComparisonArtifacts(artifacts, baselineFile, candidateFile);
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_MARKDOWN_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                artifacts.resolve(TrainingReportComparisonArtifacts.DEFAULT_METRICS_CSV_FILE_NAME),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportComparisonArtifacts(artifacts, null, null, null);

        assertFalse(verification.passed());
        assertFalse(verification.markdownMatchesJson());
        assertFalse(verification.metricsCsvMatchesJson());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown report does not match JSON export")));
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("Metrics CSV does not match JSON export")));
    }

    @Test
    void refreshDerivedArtifactsRebuildsFilesFromJsonExport() throws IOException {
        Path baselineFile = writeReport(
                "refresh-artifact-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)));
        Path candidateFile = writeReport(
                "refresh-artifact-candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55)));
        Path artifacts = tempDir.resolve("refresh-comparison-artifacts");
        TrainingReportComparisonArtifacts.ArtifactBundle original =
                Gollek.DL.writeTrainingReportComparisonArtifacts(artifacts, baselineFile, candidateFile);

        Files.writeString(
                original.markdownFile(),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                original.findingsCsvFile(),
                "\nmanual stale edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        TrainingReportComparisonArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportComparisonArtifacts(artifacts);
        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportComparisonArtifacts(refreshed);

        assertEquals(original.jsonSha256(), refreshed.jsonSha256());
        assertFalse(Files.readString(refreshed.markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertFalse(Files.readString(refreshed.findingsCsvFile(), StandardCharsets.UTF_8)
                .contains("manual stale edit"));
        assertTrue(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.findingsCsvMatchesJson());
    }

    @Test
    void comparisonArtifactVerificationRejectsXmlDoctype() throws IOException {
        Path baselineFile = writeReport(
                "xxe-artifact-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7)));
        Path candidateFile = writeReport(
                "xxe-artifact-candidate.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.55)));
        Path artifacts = tempDir.resolve("xxe-comparison-artifacts");
        TrainingReportComparisonArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportComparisonArtifacts(artifacts, baselineFile, candidateFile);

        Files.writeString(bundle.junitXmlFile(), xmlWithDoctype(), StandardCharsets.UTF_8);

        TrainingReportComparisonArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportComparisonArtifacts(artifacts, null, null, null);

        assertFalse(verification.passed());
        assertFalse(verification.junitXmlWellFormed());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML is not well-formed")));
    }

    @Test
    void flagsOptimizerDynamicsRegressionEvenWhenLossImproves() throws IOException {
        TrainingReport baseline = TrainingReportReader.readReport(writeReport(
                "optimizer-baseline.json",
                100L,
                List.of(
                        optimizerRow(0, 0.8, 0.9, 0.20, 0.01),
                        optimizerRow(1, 0.6, 0.7, 0.25, 0.01))));
        TrainingReport candidate = TrainingReportReader.readReport(writeReport(
                "optimizer-candidate.json",
                100L,
                List.of(
                        optimizerRow(0, 0.7, 0.8, 0.50, 0.02),
                        optimizerRow(1, 0.45, 0.5, 1.25, 0.05))));

        TrainingReportComparison comparison = TrainingReportComparison.compare(baseline, candidate);

        assertTrue(hasCode(comparison.findings(), "comparison.gradient_l2_norm.spiked"));
        assertTrue(hasCode(comparison.findings(), "comparison.parameter_update_ratio.regressed"));
        assertEquals("WARNING", comparison.summary().highestSeverity());
        assertEquals(1.25, comparison.metric("latestGradientL2Norm").orElseThrow()
                .candidateValue().orElseThrow(), 1e-12);
        assertEquals(0.05, comparison.metric("latestParameterUpdateToParameterL2Ratio").orElseThrow()
                .candidateValue().orElseThrow(), 1e-12);
        assertTrue(comparison.improvedMetrics().stream()
                .anyMatch(metric -> metric.name().equals("bestValidationLoss")));
        assertFalse(comparison.gate(TrainingReportDiagnostics.Severity.INFO).passed());
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
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T20:21:22Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
    }

    private static boolean hasCode(List<TrainingReportDiagnostics.Finding> findings, String code) {
        return findings.stream().anyMatch(finding -> code.equals(finding.code()));
    }

    private static String xmlWithDoctype() {
        return """
                <?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE testsuite [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
                <testsuite name="unsafe">&xxe;</testsuite>
                """;
    }

    private static Map<String, Object> optimizerRow(
            int epoch,
            double trainLoss,
            double validationLoss,
            double gradientL2Norm,
            double parameterUpdateToParameterL2Ratio) {
        return Map.ofEntries(
                Map.entry("epoch", epoch),
                Map.entry("trainLoss", trainLoss),
                Map.entry("validationLoss", validationLoss),
                Map.entry("gradientL2Norm", gradientL2Norm),
                Map.entry("gradientZeroFraction", 0.0),
                Map.entry("gradientNonFiniteCount", 0L),
                Map.entry("gradientNonFiniteFraction", 0.0),
                Map.entry("parameterL2Norm", 1.0),
                Map.entry("parameterNonFiniteCount", 0L),
                Map.entry("parameterNonFiniteFraction", 0.0),
                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                Map.entry("parameterUpdateL2Norm", parameterUpdateToParameterL2Ratio),
                Map.entry("parameterUpdateToParameterL2Ratio", parameterUpdateToParameterL2Ratio),
                Map.entry("parameterUpdateToGradientL2Ratio", parameterUpdateToParameterL2Ratio / gradientL2Norm),
                Map.entry("parameterUpdateNonFiniteCount", 0L),
                Map.entry("parameterUpdateNonFiniteFraction", 0.0));
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
