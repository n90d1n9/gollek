package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.trainer.api.TrainingSummary;

final class TrainingReportQualityProfileTestFixtures {
    private TrainingReportQualityProfileTestFixtures() {
    }

    static Map<String, Path> reportFiles(
            Path directory,
            Map<String, Object> baselineMetadata,
            Map<String, Object> candidateMetadata) throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                directory,
                "baseline.json",
                baselineMetadata,
                120L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                directory,
                "candidate.json",
                candidateMetadata,
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        return Map.copyOf(reports);
    }

    static Path writeReport(
            Path directory,
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
        Path reportFile = directory.resolve(fileName);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-30T01:02:03Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
    }

    static Map<String, Object> cleanDataHealthMetadata() {
        return Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "healthy"),
                Map.entry("dataLoaderPlanHealthHealthy", true),
                Map.entry("dataLoaderPlanHealthGatePassed", true),
                Map.entry("dataLoaderPlanHealthIssueDetected", false),
                Map.entry("dataLoaderPlanHealthIssueCount", 0),
                Map.entry("dataLoaderPlanHealthWarningCount", 0),
                Map.entry("dataLoaderPlanHealthErrorCount", 0),
                Map.entry("dataLoaderPlanHealthIssues", List.of()),
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

    static Map<String, Object> warningDataHealthMetadata() {
        Map<String, Object> issue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "blocking", false,
                "message", "train loader dropLast discarded samples",
                "action", "adjust batch size or disable dropLast for small datasets");
        Map<String, Object> metadata = new LinkedHashMap<>(cleanDataHealthMetadata());
        metadata.put("dataLoaderPlanHealthStatus", "warning");
        metadata.put("dataLoaderPlanHealthHealthy", false);
        metadata.put("dataLoaderPlanHealthIssueDetected", true);
        metadata.put("dataLoaderPlanHealthIssueCount", 1);
        metadata.put("dataLoaderPlanHealthWarningCount", 1);
        metadata.put("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-drop-last-discarded-samples"));
        metadata.put("dataLoaderPlanHealthIssueSeverities", List.of("warning"));
        metadata.put("dataLoaderPlanHealthRecommendedActions", List.of(issue.get("action")));
        metadata.put("dataLoaderPlanHealthIssues", List.of(issue));
        return Map.copyOf(metadata);
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
}
