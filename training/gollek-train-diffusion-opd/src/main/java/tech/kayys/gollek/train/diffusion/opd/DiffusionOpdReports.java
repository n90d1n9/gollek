package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;

/**
 * Public loader/parser utilities for normalized DiffusionOPD report artifacts.
 *
 * <p>The selector grammar intentionally supports compact terminal queries so
 * callers can pivot from full row dumps to scalar or summary views such as
 * {@code roundHistory:task=ocr:meanLoss},
 * {@code roundHistory:teacher=ocr-early:lastLoss}, or
 * {@code roundHistory:task=ocr:summary}.
 */
public final class DiffusionOpdReports {
    private static final ObjectMapper JSON = new ObjectMapper();

    private DiffusionOpdReports() {
    }

    public static DiffusionOpdReport load(Path reportPath) {
        try {
            JsonNode root = JSON.readTree(reportPath.toFile());
            return fromJson(root);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load DiffusionOPD report from " + reportPath, e);
        }
    }

    public static DiffusionOpdReport fromJson(JsonNode root) {
        JsonNode runNode = root.path("run");
        JsonNode artifactsNode = root.path("artifacts");
        return new DiffusionOpdReport(
                new DiffusionOpdRunReport(
                        runNode.path("epochCount").asInt(),
                        nullableDouble(runNode.get("latestTrainLoss")),
                        runNode.path("durationMs").asLong(),
                        toJsonValue(runNode.get("samplerType")),
                        toJsonValue(runNode.get("taskCount")),
                        toJsonValue(runNode.get("optimizationSteps")),
                        toJsonValue(runNode.get("roundsCompleted")),
                        toJsonValue(runNode.get("stopped"))),
                new DiffusionOpdArtifactsReport(
                        nullableText(artifactsNode.get("summaryFile")),
                        nullableText(artifactsNode.get("historyFile")),
                        nullableText(artifactsNode.get("reportFile")),
                        nullableText(artifactsNode.get("checkpointDir"))),
                objectMap(root.get("teachers")),
                objectMap(root.get("stages")),
                objectMap(root.get("tasks")),
                objectMap(root.get("conditioning")),
                objectMap(root.get("adaptive")),
                objectMap(root.get("bindings")),
                objectList(root.get("roundHistory")));
    }

    public static Map<String, Object> sections(DiffusionOpdReport report) {
        Objects.requireNonNull(report, "report must not be null");
        Map<String, Object> sections = new LinkedHashMap<>();
        sections.put("run", report.run().asMap());
        sections.put("artifacts", report.artifacts().asMap());
        sections.put("teachers", report.teachers());
        sections.put("stages", report.stages());
        sections.put("tasks", report.tasks());
        sections.put("conditioning", report.conditioning());
        sections.put("adaptive", report.adaptive());
        sections.put("bindings", report.bindings());
        sections.put("roundHistory", report.roundHistory());
        sections.put("roundHistoryCount", report.roundHistory().size());
        return Map.copyOf(sections);
    }

    public static Object select(DiffusionOpdReport report, String section) {
        Objects.requireNonNull(report, "report must not be null");
        String normalized = section == null ? "all" : section.toLowerCase(Locale.ROOT);
        Map<String, Object> sections = sections(report);
        if ("all".equals(normalized)) {
            return sections;
        }
        if ("overview".equals(normalized)) {
            return Map.of(
                    "run", report.run().asMap(),
                    "artifacts", report.artifacts().asMap(),
                    "roundHistoryCount", report.roundHistory().size());
        }
        if (normalized.startsWith("tasksummaries")) {
            return selectGroupedRoundHistory(report.roundHistory(), "taskId", normalized.substring("tasksummaries".length()));
        }
        if (normalized.startsWith("teachersummaries")) {
            return selectGroupedRoundHistory(report.roundHistory(), "teacherKey", normalized.substring("teachersummaries".length()));
        }
        if (normalized.startsWith("stagesummaries")) {
            return selectGroupedRoundHistory(report.roundHistory(), "stageName", normalized.substring("stagesummaries".length()));
        }
        if (normalized.startsWith("taskteachersummaries")) {
            return selectGroupedPairedRoundHistory(
                    report.roundHistory(),
                    "taskId",
                    "teacherKey",
                    normalized.substring("taskteachersummaries".length()));
        }
        if (normalized.startsWith("taskstagesummaries")) {
            return selectGroupedPairedRoundHistory(
                    report.roundHistory(),
                    "taskId",
                    "stageName",
                    normalized.substring("taskstagesummaries".length()));
        }
        if (normalized.startsWith("teacherstagesummaries")) {
            return selectGroupedPairedRoundHistory(
                    report.roundHistory(),
                    "teacherKey",
                    "stageName",
                    normalized.substring("teacherstagesummaries".length()));
        }
        if (normalized.startsWith("tasksummary=")) {
            return summarizeRoundHistory(report.roundHistory(), "taskId", normalized.substring("tasksummary=".length()));
        }
        if (normalized.startsWith("teachersummary=")) {
            return summarizeRoundHistory(report.roundHistory(), "teacherKey", normalized.substring("teachersummary=".length()));
        }
        if (normalized.startsWith("stagesummary=")) {
            return summarizeRoundHistory(report.roundHistory(), "stageName", normalized.substring("stagesummary=".length()));
        }
        if (normalized.startsWith("taskteachersummary=")) {
            return summarizePairedRoundHistory(
                    report.roundHistory(),
                    "taskId",
                    "teacherKey",
                    normalized.substring("taskteachersummary=".length()));
        }
        if (normalized.startsWith("taskstagesummary=")) {
            return summarizePairedRoundHistory(
                    report.roundHistory(),
                    "taskId",
                    "stageName",
                    normalized.substring("taskstagesummary=".length()));
        }
        if (normalized.startsWith("teacherstagesummary=")) {
            return summarizePairedRoundHistory(
                    report.roundHistory(),
                    "teacherKey",
                    "stageName",
                    normalized.substring("teacherstagesummary=".length()));
        }
        if (normalized.startsWith("roundhistory:")) {
            return selectRoundHistory(report.roundHistory(), normalized.substring("roundhistory:".length()));
        }
        Object value = sections.get(normalized);
        if (value == null) {
            throw new IllegalArgumentException(
                    "Unknown section '" + section + "'. Use one of: all, overview, run, artifacts, teachers, stages, tasks, conditioning, adaptive, bindings, roundHistory, roundHistoryCount, taskSummaries, teacherSummaries, stageSummaries, taskTeacherSummaries, taskStageSummaries, teacherStageSummaries, taskSummaries:sort=-meanLoss:top=5, taskSummary=<taskId>, teacherSummary=<teacherKey>, stageSummary=<stageName>, taskTeacherSummary=<taskId>,<teacherKey>, taskStageSummary=<taskId>,<stageName>, teacherStageSummary=<teacherKey>,<stageName>, roundHistory:last, roundHistory:<index>, roundHistory:task=<taskId>, roundHistory:teacher=<teacherKey>, roundHistory:stage=<stageName>, roundHistory:round=<round>, roundHistory:task=<taskId>:last, roundHistory:task=<taskId>:meanLoss, roundHistory:task=<taskId>:lastLoss, roundHistory:task=<taskId>:summary, roundHistory:task=<taskId>:sort=-averageLoss:top=3.");
        }
        return value;
    }

    private static Map<String, Object> selectGroupedRoundHistory(
            List<Map<String, Object>> roundHistory,
            String field,
            String suffix) {
        Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
        for (Map<String, Object> row : roundHistory) {
            Object value = row.get(field);
            if (value == null) {
                continue;
            }
            grouped.computeIfAbsent(String.valueOf(value), ignored -> new ArrayList<>()).add(row);
        }
        List<Map<String, Object>> summaries = new ArrayList<>();
        grouped.forEach((key, rows) -> {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("value", key);
            summary.putAll(summarizeRows(rows));
            summaries.add(Map.copyOf(summary));
        });
        List<Map<String, Object>> selected = applyGroupedSummarySegments(summaries, suffix);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("field", field);
        result.put("values", selected);
        return Map.copyOf(result);
    }

    private static List<Map<String, Object>> applyGroupedSummarySegments(
            List<Map<String, Object>> summaries,
            String suffix) {
        if (suffix == null || suffix.isBlank()) {
            return List.copyOf(summaries);
        }
        String normalized = suffix.startsWith(":") ? suffix.substring(1) : suffix;
        if (normalized.isBlank()) {
            return List.copyOf(summaries);
        }
        List<Map<String, Object>> current = List.copyOf(summaries);
        for (String rawSegment : normalized.split(":")) {
            String segment = rawSegment == null ? "" : rawSegment.trim();
            if (segment.isEmpty() || "all".equals(segment)) {
                continue;
            }
            current = applyGroupedSummarySegment(current, segment);
        }
        return current;
    }

    private static List<Map<String, Object>> applyGroupedSummarySegment(
            List<Map<String, Object>> summaries,
            String segment) {
        if (segment.startsWith("top=")) {
            int limit = parsePositiveInt("top", segment.substring("top=".length()));
            return summaries.stream().limit(limit).toList();
        }
        if (segment.startsWith("sort=")) {
            return sortGroupedSummaries(summaries, segment.substring("sort=".length()));
        }
        throw new IllegalArgumentException(
                "Unsupported grouped summary selector '" + segment + "'. Use sort=<field> or top=<n>.");
    }

    private static Map<String, Object> summarizeRoundHistory(
            List<Map<String, Object>> roundHistory,
            String field,
            String expectedValue) {
        List<Map<String, Object>> rows = roundHistory.stream()
                .filter(row -> matchesSelector(row.get(field), expectedValue))
                .toList();
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("field", field);
        summary.put("value", expectedValue);
        summary.putAll(summarizeRows(rows));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> summarizePairedRoundHistory(
            List<Map<String, Object>> roundHistory,
            String firstField,
            String secondField,
            String selector) {
        int separator = selector.indexOf(',');
        if (separator <= 0 || separator == selector.length() - 1) {
            throw new IllegalArgumentException(
                    "Paired summary selector must look like <first>,<second> but was '" + selector + "'.");
        }
        String firstValue = selector.substring(0, separator).trim();
        String secondValue = selector.substring(separator + 1).trim();
        List<Map<String, Object>> rows = roundHistory.stream()
                .filter(row -> matchesSelector(row.get(firstField), firstValue))
                .filter(row -> matchesSelector(row.get(secondField), secondValue))
                .toList();
        Map<String, Object> summary = summarizeRows(rows);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("firstField", firstField);
        result.put("firstValue", firstValue);
        result.put("secondField", secondField);
        result.put("secondValue", secondValue);
        result.putAll(summary);
        return Map.copyOf(result);
    }

    private static Map<String, Object> selectGroupedPairedRoundHistory(
            List<Map<String, Object>> roundHistory,
            String firstField,
            String secondField,
            String suffix) {
        Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
        for (Map<String, Object> row : roundHistory) {
            Object first = row.get(firstField);
            Object second = row.get(secondField);
            if (first == null || second == null) {
                continue;
            }
            String key = String.valueOf(first) + "," + String.valueOf(second);
            grouped.computeIfAbsent(key, ignored -> new ArrayList<>()).add(row);
        }
        List<Map<String, Object>> summaries = new ArrayList<>();
        grouped.forEach((key, rows) -> {
            int separator = key.indexOf(',');
            String firstValue = key.substring(0, separator);
            String secondValue = key.substring(separator + 1);
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("firstValue", firstValue);
            summary.put("secondValue", secondValue);
            summary.put("pair", key);
            summary.putAll(summarizeRows(rows));
            summaries.add(Map.copyOf(summary));
        });
        List<Map<String, Object>> selected = applyGroupedSummarySegments(summaries, suffix);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("firstField", firstField);
        result.put("secondField", secondField);
        result.put("values", selected);
        return Map.copyOf(result);
    }

    private static Map<String, Object> summarizeRows(List<Map<String, Object>> rows) {
        Double meanLoss = rows.stream()
                .map(row -> row.get("averageLoss"))
                .map(DiffusionOpdReports::asDouble)
                .filter(Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .average()
                .stream()
                .boxed()
                .findFirst()
                .orElse(null);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("count", rows.size());
        summary.put("meanLoss", meanLoss);
        summary.put("last", rows.isEmpty()
                ? Map.of()
                : rows.stream().max(Comparator.comparingInt(DiffusionOpdReports::roundNumber)).orElseThrow());
        summary.put("topLosses", sortRoundHistory(rows, "-averageLoss").stream().limit(3).toList());
        summary.put("firstRounds", sortRoundHistory(rows, "round").stream().limit(3).toList());
        return Map.copyOf(summary);
    }

    private static Object selectRoundHistory(List<Map<String, Object>> roundHistory, String selector) {
        if (roundHistory.isEmpty()) {
            return List.of();
        }
        if (selector == null || selector.isBlank() || "all".equals(selector)) {
            return roundHistory;
        }
        List<String> segments = List.of(selector.split(":"));
        Object current = roundHistory;
        for (String rawSegment : segments) {
            String segment = rawSegment == null ? "" : rawSegment.trim();
            if (segment.isEmpty() || "all".equals(segment)) {
                continue;
            }
            current = applyRoundHistorySegment(current, segment, roundHistory.size());
        }
        return current;
    }

    private static Object applyRoundHistorySegment(Object current, String segment, int totalSize) {
        if ("last".equals(segment)) {
            List<Map<String, Object>> rows = asRoundHistoryList(current, segment);
            if (rows.isEmpty()) {
                return Map.of();
            }
            return rows.stream()
                    .max(Comparator.comparingInt(DiffusionOpdReports::roundNumber))
                    .orElseThrow();
        }
        if ("summary".equals(segment)) {
            return summarizeRows(asRoundHistoryList(current, segment));
        }
        if ("meanloss".equals(segment) || "avgloss".equals(segment)) {
            List<Map<String, Object>> rows = asRoundHistoryList(current, segment);
            OptionalDouble meanLoss = rows.stream()
                    .map(row -> row.get("averageLoss"))
                    .map(DiffusionOpdReports::asDouble)
                    .filter(Objects::nonNull)
                    .mapToDouble(Double::doubleValue)
                    .average();
            return meanLoss.isPresent() ? meanLoss.getAsDouble() : null;
        }
        if ("minloss".equals(segment)) {
            return scalarLossAggregate(current, segment, Comparator.naturalOrder());
        }
        if ("maxloss".equals(segment)) {
            return scalarLossAggregate(current, segment, Comparator.reverseOrder());
        }
        if ("lastloss".equals(segment)) {
            Map<String, Object> row = latestRow(asRoundHistoryList(current, segment));
            return row == null ? null : asDouble(row.get("averageLoss"));
        }
        if ("lastround".equals(segment)) {
            Map<String, Object> row = latestRow(asRoundHistoryList(current, segment));
            return row == null ? null : row.get("round");
        }
        if ("count".equals(segment)) {
            return asRoundHistoryList(current, segment).size();
        }
        if (segment.startsWith("top=")) {
            List<Map<String, Object>> rows = asRoundHistoryList(current, segment);
            int limit = parsePositiveInt("top", segment.substring("top=".length()));
            return rows.stream().limit(limit).toList();
        }
        if (segment.startsWith("sort=")) {
            List<Map<String, Object>> rows = asRoundHistoryList(current, segment);
            return sortRoundHistory(rows, segment.substring("sort=".length()));
        }
        if (segment.chars().allMatch(Character::isDigit)) {
            List<Map<String, Object>> rows = asRoundHistoryList(current, segment);
            int index = Integer.parseInt(segment);
            if (index < 0 || index >= rows.size()) {
                throw new IllegalArgumentException(
                        "roundHistory index out of range: " + index + " (size=" + rows.size() + ", fullSize=" + totalSize + ")");
            }
            return rows.get(index);
        }
        int separator = segment.indexOf('=');
        if (separator <= 0 || separator == segment.length() - 1) {
            throw new IllegalArgumentException(
                    "Unsupported roundHistory selector '" + segment + "'. Use last, summary, count, meanLoss, avgLoss, minLoss, maxLoss, lastLoss, lastRound, top=<n>, sort=<field>, <index>, task=<taskId>, teacher=<teacherKey>, stage=<stageName>, or round=<round>.");
        }
        String key = segment.substring(0, separator).trim();
        String expected = segment.substring(separator + 1).trim();
        String field = switch (key) {
            case "task" -> "taskId";
            case "teacher" -> "teacherKey";
            case "stage" -> "stageName";
            case "round" -> "round";
            default -> throw new IllegalArgumentException(
                    "Unsupported roundHistory selector key '" + key + "'. Use task, teacher, stage, or round.");
        };
        return asRoundHistoryList(current, segment).stream()
                .filter(row -> matchesSelector(row.get(field), expected))
                .toList();
    }

    private static List<Map<String, Object>> asRoundHistoryList(Object current, String segment) {
        if (current instanceof List<?> list) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rows = (List<Map<String, Object>>) list;
            return rows;
        }
        if (current instanceof Map<?, ?>) {
            throw new IllegalArgumentException(
                    "roundHistory selector '" + segment + "' cannot be applied after a single-row result. Reorder the selector chain.");
        }
        if (current == null) {
            return List.of();
        }
        throw new IllegalArgumentException(
                "roundHistory selector '" + segment + "' cannot be applied to scalar result type " + current.getClass().getSimpleName() + ".");
    }

    private static Double asDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value == null) {
            return null;
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static Double scalarLossAggregate(Object current, String segment, Comparator<Double> comparator) {
        return asRoundHistoryList(current, segment).stream()
                .map(row -> row.get("averageLoss"))
                .map(DiffusionOpdReports::asDouble)
                .filter(Objects::nonNull)
                .min(comparator)
                .orElse(null);
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

    private static List<Map<String, Object>> sortRoundHistory(List<Map<String, Object>> rows, String sortSpec) {
        if (sortSpec == null || sortSpec.isBlank()) {
            throw new IllegalArgumentException("sort field must not be blank.");
        }
        boolean descending = sortSpec.startsWith("-");
        String field = descending || sortSpec.startsWith("+")
                ? sortSpec.substring(1)
                : sortSpec;
        Comparator<Map<String, Object>> comparator = Comparator.comparing(
                row -> sortableValue(row.get(field)),
                DiffusionOpdReports::compareSortableValues);
        if (descending) {
            comparator = comparator.reversed();
        }
        return rows.stream().sorted(comparator).toList();
    }

    private static List<Map<String, Object>> sortGroupedSummaries(List<Map<String, Object>> rows, String sortSpec) {
        if (sortSpec == null || sortSpec.isBlank()) {
            throw new IllegalArgumentException("grouped summary sort field must not be blank.");
        }
        boolean descending = sortSpec.startsWith("-");
        String field = descending || sortSpec.startsWith("+")
                ? sortSpec.substring(1)
                : sortSpec;
        Comparator<Map<String, Object>> comparator = Comparator.comparing(
                row -> sortableValue(row.get(field)),
                DiffusionOpdReports::compareSortableValues);
        if (descending) {
            comparator = comparator.reversed();
        }
        return rows.stream().sorted(comparator).toList();
    }

    private static Comparable<?> sortableValue(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof Comparable<?> comparable) {
            return comparable;
        }
        return String.valueOf(value);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static int compareSortableValues(Comparable left, Comparable right) {
        if (left == right) {
            return 0;
        }
        if (left == null) {
            return -1;
        }
        if (right == null) {
            return 1;
        }
        if (left.getClass().isInstance(right)) {
            return left.compareTo(right);
        }
        if (left instanceof Number || right instanceof Number) {
            return Double.compare(Double.parseDouble(String.valueOf(left)), Double.parseDouble(String.valueOf(right)));
        }
        return String.valueOf(left).compareTo(String.valueOf(right));
    }

    private static boolean matchesSelector(Object candidate, String expected) {
        return candidate != null && expected.equals(String.valueOf(candidate));
    }

    private static int roundNumber(Map<String, Object> row) {
        Object value = row.get("round");
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(String.valueOf(value));
    }

    private static Map<String, Object> latestRow(List<Map<String, Object>> rows) {
        if (rows.isEmpty()) {
            return null;
        }
        return rows.stream()
                .max(Comparator.comparingInt(DiffusionOpdReports::roundNumber))
                .orElseThrow();
    }

    private static Map<String, Object> objectMap(JsonNode node) {
        if (node == null || node.isNull() || !node.isObject()) {
            return Map.of();
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = node.fields(); it.hasNext();) {
            Map.Entry<String, JsonNode> entry = it.next();
            values.put(entry.getKey(), toJsonValue(entry.getValue()));
        }
        return Map.copyOf(values);
    }

    private static List<Map<String, Object>> objectList(JsonNode node) {
        if (node == null || node.isNull() || !node.isArray()) {
            return List.of();
        }
        List<Map<String, Object>> values = new ArrayList<>();
        for (JsonNode element : node) {
            values.add(objectMap(element));
        }
        return List.copyOf(values);
    }

    private static Object toJsonValue(JsonNode node) {
        if (node == null || node.isNull()) {
            return null;
        }
        if (node.isTextual()) {
            return node.asText();
        }
        if (node.isBoolean()) {
            return node.asBoolean();
        }
        if (node.isInt() || node.isLong()) {
            return node.asLong();
        }
        if (node.isFloat() || node.isDouble() || node.isBigDecimal()) {
            return node.asDouble();
        }
        if (node.isArray()) {
            List<Object> values = new ArrayList<>();
            for (JsonNode element : node) {
                values.add(toJsonValue(element));
            }
            return List.copyOf(values);
        }
        if (node.isObject()) {
            return objectMap(node);
        }
        return node.asText();
    }

    private static String nullableText(JsonNode node) {
        return node == null || node.isNull() ? null : node.asText();
    }

    private static Double nullableDouble(JsonNode node) {
        return node == null || node.isNull() ? null : node.asDouble();
    }
}
