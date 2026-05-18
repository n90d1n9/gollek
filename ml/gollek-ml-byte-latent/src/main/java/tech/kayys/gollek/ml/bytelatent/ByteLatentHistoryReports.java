package tech.kayys.gollek.ml.bytelatent;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * Compact selector helpers for byte-latent epoch history CSV artifacts.
 */
public final class ByteLatentHistoryReports {
    private ByteLatentHistoryReports() {
    }

    public static List<ByteLatentHistoryRow> load(Path historyFile) {
        try {
            return ByteLatentHistoryCsv.read(historyFile);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load byte-latent history from " + historyFile, exception);
        }
    }

    public static Map<String, Object> sections(List<ByteLatentHistoryRow> rows) {
        Objects.requireNonNull(rows, "rows must not be null");
        Map<String, Object> sections = new LinkedHashMap<>();
        sections.put("history", rows.stream().map(ByteLatentHistoryReports::asMap).toList());
        sections.put("historyCount", rows.size());
        sections.put("summary", summarize(rows));
        return Map.copyOf(sections);
    }

    public static Object select(Path historyFile, String section) {
        return select(load(historyFile), section);
    }

    public static Object select(List<ByteLatentHistoryRow> rows, String section) {
        Objects.requireNonNull(rows, "rows must not be null");
        String normalized = section == null ? "all" : section.toLowerCase(Locale.ROOT);
        if ("all".equals(normalized) || "history".equals(normalized)) {
            return rows.stream().map(ByteLatentHistoryReports::asMap).toList();
        }
        if ("overview".equals(normalized)) {
            return Map.of(
                    "historyCount", rows.size(),
                    "summary", summarize(rows));
        }
        if ("historycount".equals(normalized) || "count".equals(normalized)) {
            return rows.size();
        }
        if ("summary".equals(normalized)) {
            return summarize(rows);
        }
        if (normalized.startsWith("history:")) {
            return selectHistory(rows, normalized.substring("history:".length()));
        }
        throw new IllegalArgumentException(
                "Unknown section '" + section + "'. Use one of: all, overview, history, historyCount, summary, history:last, history:<index>, history:epoch=<n>, history:summary, history:meanLoss, history:avgLoss, history:minLoss, history:maxLoss, history:lastLoss, history:lastEpoch, history:lastGlobalStep, history:totalBatches, history:sort=-trainLoss:top=3.");
    }

    private static Object selectHistory(List<ByteLatentHistoryRow> rows, String selector) {
        if (rows.isEmpty()) {
            return List.of();
        }
        if (selector == null || selector.isBlank() || "all".equals(selector)) {
            return rows.stream().map(ByteLatentHistoryReports::asMap).toList();
        }
        Object current = rows;
        for (String rawSegment : selector.split(":")) {
            String segment = rawSegment == null ? "" : rawSegment.trim();
            if (segment.isEmpty() || "all".equals(segment)) {
                continue;
            }
            current = applyHistorySegment(current, segment, rows.size());
        }
        return normalizeSelection(current);
    }

    private static Object applyHistorySegment(Object current, String segment, int totalSize) {
        if ("last".equals(segment)) {
            ByteLatentHistoryRow row = latestRow(asHistoryList(current, segment));
            return row == null ? Map.of() : asMap(row);
        }
        if ("summary".equals(segment)) {
            return summarize(asHistoryList(current, segment));
        }
        if ("meanloss".equals(segment) || "avgloss".equals(segment)) {
            OptionalDouble mean = asHistoryList(current, segment).stream()
                    .mapToDouble(ByteLatentHistoryRow::trainLoss)
                    .average();
            return mean.isPresent() ? mean.getAsDouble() : null;
        }
        if ("minloss".equals(segment)) {
            return asHistoryList(current, segment).stream()
                    .map(ByteLatentHistoryRow::trainLoss)
                    .min(Comparator.naturalOrder())
                    .orElse(null);
        }
        if ("maxloss".equals(segment)) {
            return asHistoryList(current, segment).stream()
                    .map(ByteLatentHistoryRow::trainLoss)
                    .max(Comparator.naturalOrder())
                    .orElse(null);
        }
        if ("lastloss".equals(segment)) {
            ByteLatentHistoryRow row = latestRow(asHistoryList(current, segment));
            return row == null ? null : row.trainLoss();
        }
        if ("lastepoch".equals(segment)) {
            ByteLatentHistoryRow row = latestRow(asHistoryList(current, segment));
            return row == null ? null : row.epoch();
        }
        if ("lastglobalstep".equals(segment)) {
            ByteLatentHistoryRow row = latestRow(asHistoryList(current, segment));
            return row == null ? null : row.globalStep();
        }
        if ("totalbatches".equals(segment)) {
            return asHistoryList(current, segment).stream()
                    .mapToInt(ByteLatentHistoryRow::batchCount)
                    .sum();
        }
        if ("count".equals(segment)) {
            return asHistoryList(current, segment).size();
        }
        if (segment.startsWith("top=")) {
            int limit = parsePositiveInt("top", segment.substring("top=".length()));
            return asHistoryList(current, segment).stream()
                    .limit(limit)
                    .toList();
        }
        if (segment.startsWith("sort=")) {
            return sortHistory(asHistoryList(current, segment), segment.substring("sort=".length()));
        }
        if (segment.chars().allMatch(Character::isDigit)) {
            List<ByteLatentHistoryRow> rows = asHistoryList(current, segment);
            int index = Integer.parseInt(segment);
            if (index < 0 || index >= rows.size()) {
                throw new IllegalArgumentException(
                        "history index out of range: " + index + " (size=" + rows.size() + ", fullSize=" + totalSize + ")");
            }
            return asMap(rows.get(index));
        }
        int separator = segment.indexOf('=');
        if (separator <= 0 || separator == segment.length() - 1) {
            throw new IllegalArgumentException(
                    "Unsupported history selector '" + segment + "'. Use last, summary, count, meanLoss, avgLoss, minLoss, maxLoss, lastLoss, lastEpoch, lastGlobalStep, totalBatches, top=<n>, sort=<field>, <index>, or epoch=<n>.");
        }
        String key = segment.substring(0, separator).trim();
        String expected = segment.substring(separator + 1).trim();
        if (!"epoch".equals(key)) {
            throw new IllegalArgumentException(
                    "Unsupported history selector key '" + key + "'. Use epoch.");
        }
        return asHistoryList(current, segment).stream()
                .filter(row -> String.valueOf(row.epoch()).equals(expected))
                .toList();
    }

    private static List<ByteLatentHistoryRow> asHistoryList(Object current, String segment) {
        if (current instanceof List<?> list) {
            @SuppressWarnings("unchecked")
            List<ByteLatentHistoryRow> rows = (List<ByteLatentHistoryRow>) list;
            return rows;
        }
        if (current instanceof Map<?, ?>) {
            throw new IllegalArgumentException(
                    "history selector '" + segment + "' cannot be applied after a single-row result. Reorder the selector chain.");
        }
        if (current == null) {
            return List.of();
        }
        throw new IllegalArgumentException(
                "history selector '" + segment + "' cannot be applied to scalar result type " + current.getClass().getSimpleName() + ".");
    }

    private static List<ByteLatentHistoryRow> sortHistory(List<ByteLatentHistoryRow> rows, String sortSpec) {
        if (sortSpec == null || sortSpec.isBlank()) {
            throw new IllegalArgumentException("sort field must not be blank.");
        }
        boolean descending = sortSpec.startsWith("-");
        String field = descending || sortSpec.startsWith("+")
                ? sortSpec.substring(1)
                : sortSpec;
        Comparator<ByteLatentHistoryRow> comparator = switch (field) {
            case "epoch" -> Comparator.comparingInt(ByteLatentHistoryRow::epoch);
            case "globalstep" -> Comparator.comparingInt(ByteLatentHistoryRow::globalStep);
            case "batchcount" -> Comparator.comparingInt(ByteLatentHistoryRow::batchCount);
            case "trainloss" -> Comparator.comparingDouble(ByteLatentHistoryRow::trainLoss);
            default -> throw new IllegalArgumentException(
                    "Unsupported history sort field '" + field + "'. Use epoch, globalStep, batchCount, or trainLoss.");
        };
        if (descending) {
            comparator = comparator.reversed();
        }
        return rows.stream().sorted(comparator).toList();
    }

    private static Map<String, Object> summarize(List<ByteLatentHistoryRow> rows) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("count", rows.size());
        OptionalDouble mean = rows.stream().mapToDouble(ByteLatentHistoryRow::trainLoss).average();
        summary.put("meanLoss", mean.isPresent() ? mean.getAsDouble() : null);
        summary.put("avgLoss", summary.get("meanLoss"));
        summary.put("minLoss", rows.stream().map(ByteLatentHistoryRow::trainLoss).min(Comparator.naturalOrder()).orElse(null));
        summary.put("maxLoss", rows.stream().map(ByteLatentHistoryRow::trainLoss).max(Comparator.naturalOrder()).orElse(null));
        ByteLatentHistoryRow last = latestRow(rows);
        summary.put("lastLoss", last == null ? null : last.trainLoss());
        summary.put("lastEpoch", last == null ? null : last.epoch());
        summary.put("lastGlobalStep", last == null ? null : last.globalStep());
        summary.put("totalBatches", rows.stream().mapToInt(ByteLatentHistoryRow::batchCount).sum());
        return Map.copyOf(summary);
    }

    private static ByteLatentHistoryRow latestRow(List<ByteLatentHistoryRow> rows) {
        return rows.stream().max(Comparator.comparingInt(ByteLatentHistoryRow::epoch)).orElse(null);
    }

    private static Map<String, Object> asMap(ByteLatentHistoryRow row) {
        return Map.of(
                "epoch", row.epoch(),
                "globalStep", row.globalStep(),
                "batchCount", row.batchCount(),
                "trainLoss", row.trainLoss());
    }

    private static Object normalizeSelection(Object current) {
        if (current instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof ByteLatentHistoryRow) {
            @SuppressWarnings("unchecked")
            List<ByteLatentHistoryRow> rows = (List<ByteLatentHistoryRow>) list;
            return rows.stream().map(ByteLatentHistoryReports::asMap).toList();
        }
        if (current instanceof List<?> list && list.isEmpty()) {
            return List.of();
        }
        return current;
    }

    private static int parsePositiveInt(String label, String value) {
        try {
            int parsed = Integer.parseInt(value);
            if (parsed < 0) {
                throw new IllegalArgumentException(label + " must be >= 0 but was " + parsed);
            }
            return parsed;
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(label + " must be an integer but was '" + value + "'.", exception);
        }
    }
}
