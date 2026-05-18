package tech.kayys.gollek.ml.bytelatent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import tech.kayys.gollek.trainer.api.TrainingSummary;

final class ByteLatentCheckpointIO {
    static final String SUMMARY_FILE_NAME = "byte-latent-summary.json";
    static final String MANIFEST_FILE_NAME = "byte-latent-checkpoint.metadata";
    static final String HISTORY_FILE_NAME = "byte-latent-history.csv";
    static final String REPORT_FILE_NAME = "byte-latent-report.json";
    static final int FORMAT_VERSION = 1;

    private ByteLatentCheckpointIO() {
    }

    static void persistSummary(
            Path checkpointDir,
            TrainingSummary summary,
            List<ByteLatentHistoryRow> historyRows) throws IOException {
        if (checkpointDir == null) {
            return;
        }
        Files.createDirectories(checkpointDir);
        Path summaryFile = checkpointDir.resolve(SUMMARY_FILE_NAME);
        String json = toJson(summary);
        Files.writeString(summaryFile, json, StandardCharsets.UTF_8);
        Path historyFile = checkpointDir.resolve(HISTORY_FILE_NAME);
        ByteLatentHistoryCsv.write(historyFile, historyRows);
        Path reportFile = checkpointDir.resolve(REPORT_FILE_NAME);
        Files.writeString(reportFile, reportJson(summary, historyRows, checkpointDir), StandardCharsets.UTF_8);

        Properties manifest = new Properties();
        manifest.setProperty("formatVersion", Integer.toString(FORMAT_VERSION));
        manifest.setProperty("summaryFile", summaryFile.getFileName().toString());
        manifest.setProperty("summaryBytes", Long.toString(Files.size(summaryFile)));
        manifest.setProperty("summarySha256", sha256Hex(summaryFile));
        manifest.setProperty("historyFile", historyFile.getFileName().toString());
        manifest.setProperty("historyRows", Integer.toString(historyRows.size()));
        manifest.setProperty("historyBytes", Long.toString(Files.size(historyFile)));
        manifest.setProperty("historySha256", sha256Hex(historyFile));
        manifest.setProperty("reportFile", reportFile.getFileName().toString());
        manifest.setProperty("reportBytes", Long.toString(Files.size(reportFile)));
        manifest.setProperty("reportSha256", sha256Hex(reportFile));
        manifest.setProperty("epochCount", Integer.toString(summary.epochCount()));
        manifest.setProperty("bestValidationLoss", Double.toString(summary.bestValidationLoss()));
        manifest.setProperty("bestValidationEpoch", Integer.toString(summary.bestValidationEpoch()));
        manifest.setProperty("durationMs", Long.toString(summary.durationMs()));
        Double latestTrainLoss = summary.latestTrainLoss();
        if (latestTrainLoss != null) {
            manifest.setProperty("latestTrainLoss", Double.toString(latestTrainLoss));
        }
        Object globalStep = summary.metadata().get("globalStep");
        if (globalStep != null) {
            manifest.setProperty("globalStep", String.valueOf(globalStep));
        }
        try (var writer = Files.newBufferedWriter(checkpointDir.resolve(MANIFEST_FILE_NAME), StandardCharsets.UTF_8)) {
            manifest.store(writer, "Gollek byte-latent trainer checkpoint manifest");
        }
    }

    static ResumeState loadResumeState(Path checkpointDir) {
        if (checkpointDir == null) {
            return ResumeState.notRequested();
        }
        Path manifestFile = checkpointDir.resolve(MANIFEST_FILE_NAME);
        if (!Files.isRegularFile(manifestFile)) {
            return ResumeState.missing("checkpoint-manifest-not-found");
        }
        try {
            Properties properties = new Properties();
            try (var reader = Files.newBufferedReader(manifestFile, StandardCharsets.UTF_8)) {
                properties.load(reader);
            }
            int formatVersion = parseInt(properties, "formatVersion", 0);
            if (formatVersion != FORMAT_VERSION) {
                return ResumeState.failed("unsupported-checkpoint-format-version:" + formatVersion);
            }
            String summaryName = properties.getProperty("summaryFile", SUMMARY_FILE_NAME);
            Path summaryFile = checkpointDir.resolve(summaryName);
            if (!Files.isRegularFile(summaryFile)) {
                return ResumeState.failed("checkpoint-summary-not-found");
            }
            long expectedBytes = parseLong(properties, "summaryBytes", -1L);
            if (expectedBytes >= 0 && Files.size(summaryFile) != expectedBytes) {
                return ResumeState.failed("checkpoint-summary-size-mismatch");
            }
            String expectedSha = properties.getProperty("summarySha256");
            if (expectedSha != null && !expectedSha.equals(sha256Hex(summaryFile))) {
                return ResumeState.failed("checkpoint-summary-sha256-mismatch");
            }
            TrainingSummary summary = new TrainingSummary(
                    parseInt(properties, "epochCount", 0),
                    parseDouble(properties, "bestValidationLoss", Double.NaN),
                    parseInt(properties, "bestValidationEpoch", 0),
                    properties.getProperty("latestTrainLoss") == null
                            ? null
                            : parseDouble(properties, "latestTrainLoss", Double.NaN),
                    null,
                    parseLong(properties, "durationMs", 0L),
                    Map.of());
            int globalStep = parseInt(properties, "globalStep", 0);
            String historyName = properties.getProperty("historyFile", HISTORY_FILE_NAME);
            Path historyFile = checkpointDir.resolve(historyName);
            if (!Files.isRegularFile(historyFile)) {
                return ResumeState.failed("checkpoint-history-not-found");
            }
            int expectedRows = parseInt(properties, "historyRows", -1);
            long expectedHistoryBytes = parseLong(properties, "historyBytes", -1L);
            if (expectedHistoryBytes >= 0 && Files.size(historyFile) != expectedHistoryBytes) {
                return ResumeState.failed("checkpoint-history-size-mismatch");
            }
            String expectedHistorySha = properties.getProperty("historySha256");
            if (expectedHistorySha != null && !expectedHistorySha.equals(sha256Hex(historyFile))) {
                return ResumeState.failed("checkpoint-history-sha256-mismatch");
            }
            List<ByteLatentHistoryRow> historyRows = ByteLatentHistoryCsv.read(historyFile);
            if (expectedRows >= 0 && historyRows.size() != expectedRows) {
                return ResumeState.failed("checkpoint-history-row-count-mismatch");
            }
            return ResumeState.loaded(summary, globalStep, historyRows);
        } catch (Exception error) {
            return ResumeState.failed(error.getMessage());
        }
    }

    private static String toJson(TrainingSummary summary) {
        return simpleJson(summaryMap(summary));
    }

    private static String reportJson(
            TrainingSummary summary,
            List<ByteLatentHistoryRow> historyRows,
            Path checkpointDir) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema", "gollek.byte-latent.report.v1");
        payload.put("summary", summaryMap(summary));
        payload.put("history", historyRows.stream().map(ByteLatentCheckpointIO::historyMap).toList());
        payload.put("historyCount", historyRows.size());
        payload.put("artifacts", Map.of(
                "checkpointDir", checkpointDir.toAbsolutePath().normalize().toString(),
                "summaryFile", checkpointDir.resolve(SUMMARY_FILE_NAME).toAbsolutePath().normalize().toString(),
                "historyFile", checkpointDir.resolve(HISTORY_FILE_NAME).toAbsolutePath().normalize().toString(),
                "reportFile", checkpointDir.resolve(REPORT_FILE_NAME).toAbsolutePath().normalize().toString(),
                "manifestFile", checkpointDir.resolve(MANIFEST_FILE_NAME).toAbsolutePath().normalize().toString()));
        return simpleJson(payload);
    }

    private static Map<String, Object> summaryMap(TrainingSummary summary) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("epochCount", summary.epochCount());
        payload.put("bestValidationLoss", summary.bestValidationLoss());
        payload.put("bestValidationEpoch", summary.bestValidationEpoch());
        payload.put("latestTrainLoss", summary.latestTrainLoss());
        payload.put("latestValidationLoss", summary.latestValidationLoss());
        payload.put("durationMs", summary.durationMs());
        payload.put("metadata", summary.metadata());
        return payload;
    }

    private static Map<String, Object> historyMap(ByteLatentHistoryRow row) {
        return Map.of(
                "epoch", row.epoch(),
                "globalStep", row.globalStep(),
                "batchCount", row.batchCount(),
                "trainLoss", row.trainLoss());
    }

    private static String simpleJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String text) {
            return "\"" + escape(text) + "\"";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            StringBuilder out = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append(simpleJson(String.valueOf(entry.getKey())))
                        .append(':')
                        .append(simpleJson(entry.getValue()));
            }
            return out.append('}').toString();
        }
        if (value instanceof Iterable<?> iterable) {
            StringBuilder out = new StringBuilder("[");
            boolean first = true;
            for (Object item : iterable) {
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append(simpleJson(item));
            }
            return out.append(']').toString();
        }
        return simpleJson(String.valueOf(value));
    }

    private static String escape(String text) {
        return text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String sha256Hex(Path file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] content = Files.readAllBytes(file);
            return HexFormat.of().formatHex(digest.digest(content));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 unavailable", exception);
        }
    }

    private static int parseInt(Properties properties, String key, int defaultValue) {
        String value = properties.getProperty(key);
        return value == null ? defaultValue : Integer.parseInt(value);
    }

    private static long parseLong(Properties properties, String key, long defaultValue) {
        String value = properties.getProperty(key);
        return value == null ? defaultValue : Long.parseLong(value);
    }

    private static double parseDouble(Properties properties, String key, double defaultValue) {
        String value = properties.getProperty(key);
        return value == null ? defaultValue : Double.parseDouble(value);
    }

    record ResumeState(
            boolean requested,
            boolean loaded,
            boolean missing,
            boolean failed,
            String detail,
            TrainingSummary summary,
            int globalStep,
            List<ByteLatentHistoryRow> historyRows) {
        static ResumeState notRequested() {
            return new ResumeState(false, false, false, false, null, null, 0, List.of());
        }

        static ResumeState loaded(TrainingSummary summary, int globalStep, List<ByteLatentHistoryRow> historyRows) {
            return new ResumeState(true, true, false, false, null, summary, globalStep, List.copyOf(historyRows));
        }

        static ResumeState missing(String detail) {
            return new ResumeState(true, false, true, false, detail, null, 0, List.of());
        }

        static ResumeState failed(String detail) {
            return new ResumeState(true, false, false, true, detail, null, 0, List.of());
        }
    }
}
