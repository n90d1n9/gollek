package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;

/**
 * Owns JSON loading and normalization for DiffusionOPD report artifacts.
 *
 * <p>This helper converts raw JSON into the stable report and section shapes
 * consumed by {@link DiffusionOpdReports} and {@link DiffusionOpdReportSelectors}.
 */
final class DiffusionOpdReportJsons {
    private static final ObjectMapper JSON = new ObjectMapper();

    private DiffusionOpdReportJsons() {
    }

    static DiffusionOpdReport load(Path reportPath) {
        try {
            JsonNode root = JSON.readTree(reportPath.toFile());
            return fromJson(root);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load DiffusionOPD report from " + reportPath, e);
        }
    }

    static DiffusionOpdReport fromJson(JsonNode root) {
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

    static Map<String, Object> sections(DiffusionOpdReport report) {
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
