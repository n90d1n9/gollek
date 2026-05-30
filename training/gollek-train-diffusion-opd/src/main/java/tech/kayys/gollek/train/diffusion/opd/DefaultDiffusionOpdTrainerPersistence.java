package tech.kayys.gollek.train.diffusion.opd;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Owns trainer artifact persistence and JSON/CSV shaping.
 *
 * <p>This helper stays at the output boundary, while
 * {@link DefaultDiffusionOpdTrainerRuntimeSupport} owns notifications and
 * runtime metadata and {@link DefaultDiffusionOpdTrainerAdaptiveSupport} owns
 * adaptive weighting and round-history row shaping.
 */
final class DefaultDiffusionOpdTrainerPersistence {

    private DefaultDiffusionOpdTrainerPersistence() {
    }

    static void persistHistory(Path historyFile, List<Map<String, Object>> roundHistory) {
        if (historyFile == null) {
            return;
        }
        try {
            Files.createDirectories(historyFile.getParent());
            List<String> lines = new ArrayList<>();
            List<String> extraKeys = historyExtraKeys(roundHistory);
            List<String> headers = new ArrayList<>(List.of(
                    "round",
                    "mean_loss",
                    "optimization_steps",
                    "adaptive_stage_factors",
                    "round_base_stage_mean_loss"));
            headers.addAll(extraKeys);
            lines.add(String.join(",", headers));
            for (Map<String, Object> row : roundHistory) {
                List<String> values = new ArrayList<>();
                values.add(String.valueOf(row.get("round")));
                values.add(String.valueOf(row.get("meanLoss")));
                values.add(String.valueOf(row.get("optimizationSteps")));
                values.add(csvEscape(String.valueOf(row.get("adaptiveTaskStageFactors"))));
                values.add(csvEscape(String.valueOf(row.get("roundBaseStageMeanLoss"))));
                for (String key : extraKeys) {
                    values.add(csvEscape(String.valueOf(row.get(key))));
                }
                lines.add(String.join(",", values));
            }
            Files.write(
                    historyFile,
                    lines,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    static void persistSummary(Path summaryFile, TrainingSummary summary) {
        if (summaryFile == null) {
            return;
        }
        try {
            Files.createDirectories(summaryFile.getParent());
            Files.writeString(
                    summaryFile,
                    toSummaryJson(summary),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    static void persistReport(Path reportFile, TrainingSummary summary, List<Map<String, Object>> roundHistory) {
        if (reportFile == null) {
            return;
        }
        try {
            Files.createDirectories(reportFile.getParent());
            Files.writeString(
                    reportFile,
                    toReportJson(summary, roundHistory),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    private static List<String> historyExtraKeys(List<Map<String, Object>> roundHistory) {
        List<String> keys = new ArrayList<>();
        for (Map<String, Object> row : roundHistory) {
            for (String key : row.keySet()) {
                if (isDefaultHistoryKey(key) || keys.contains(key)) {
                    continue;
                }
                keys.add(key);
            }
        }
        return List.copyOf(keys);
    }

    private static boolean isDefaultHistoryKey(String key) {
        return switch (key) {
            case "round", "meanLoss", "optimizationSteps", "adaptiveTaskStageFactors", "adaptiveStageFactors", "roundBaseStageMeanLoss" -> true;
            default -> false;
        };
    }

    private static String csvEscape(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    private static String toSummaryJson(TrainingSummary summary) {
        return "{\n"
                + "  \"epochCount\": " + summary.epochCount() + ",\n"
                + "  \"bestValidationLoss\": " + summary.bestValidationLoss() + ",\n"
                + "  \"bestValidationEpoch\": " + summary.bestValidationEpoch() + ",\n"
                + "  \"latestTrainLoss\": " + summary.latestTrainLoss() + ",\n"
                + "  \"latestValidationLoss\": " + summary.latestValidationLoss() + ",\n"
                + "  \"durationMs\": " + summary.durationMs() + ",\n"
                + "  \"metadata\": " + metadataToJson(summary.metadata()) + "\n"
                + "}\n";
    }

    private static String toReportJson(TrainingSummary summary, List<Map<String, Object>> roundHistory) {
        Map<String, Object> metadata = summary.metadata();
        DiffusionOpdRunReport run = new DiffusionOpdRunReport(
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.durationMs(),
                metadata.get("samplerType"),
                metadata.get("taskCount"),
                metadata.get("optimizationSteps"),
                metadata.get("roundsCompleted"),
                metadata.get("stopped"));
        DiffusionOpdArtifactsReport artifacts = new DiffusionOpdArtifactsReport(
                String.valueOf(metadata.get("summaryFile")),
                String.valueOf(metadata.get("historyFile")),
                String.valueOf(metadata.get("reportFile")),
                String.valueOf(metadata.get("checkpointDir")));
        DiffusionOpdReport report = new DiffusionOpdReport(
                run,
                artifacts,
                extractMatching(metadata, "teacher"),
                extractMatching(metadata, "stage"),
                extractMatching(metadata, "task"),
                extractMatching(metadata, "conditioning"),
                extractMatching(metadata, "adaptive"),
                Map.of("teacherBindings", metadata.get("teacherBindings")),
                List.copyOf(roundHistory));
        return valueToJson(report.asMap());
    }

    private static Map<String, Object> extractMatching(Map<String, Object> metadata, String needle) {
        Map<String, Object> extracted = new LinkedHashMap<>();
        String lowerNeedle = needle.toLowerCase();
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            if (entry.getKey().toLowerCase().contains(lowerNeedle)) {
                extracted.put(entry.getKey(), entry.getValue());
            }
        }
        return Map.copyOf(extracted);
    }

    private static String metadataToJson(Map<String, Object> metadata) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append("\"").append(entry.getKey()).append("\": ");
            sb.append(valueToJson(entry.getValue()));
        }
        sb.append("}");
        return sb.toString();
    }

    private static String valueToJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> typed = (Map<String, Object>) map;
            return metadataToJson(typed);
        }
        if (value instanceof List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(valueToJson(list.get(i)));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + String.valueOf(value).replace("\"", "\\\"") + "\"";
    }
}
