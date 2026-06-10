package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Portable export for a baseline-vs-candidate training action plan.
 */
public record TrainingReportComparisonActionPlanExport(
        TrainingReportActionPlan actionPlan,
        TrainingReportRuntimeRegressionSummary runtimeRegression,
        String markdown) {
    public static final String SCHEMA = "gollek.training.report.comparison-action-plan.v1";
    public static final String ARTIFACT_MANIFEST_SCHEMA =
            "gollek.training.report.comparison-action-plan-artifacts.v1";

    public TrainingReportComparisonActionPlanExport {
        actionPlan = Objects.requireNonNull(actionPlan, "actionPlan must not be null");
        runtimeRegression = runtimeRegression == null ? TrainingReportRuntimeRegressionSummary.empty() : runtimeRegression;
        markdown = markdown == null ? "" : markdown;
    }

    public static TrainingReportComparisonActionPlanExport of(
            TrainingReportActionPlan actionPlan,
            TrainingReportRuntimeRegressionSummary runtimeRegression) {
        return new TrainingReportComparisonActionPlanExport(
                actionPlan,
                runtimeRegression,
                TrainingReportActionPlanMarkdown.render(actionPlan, runtimeRegression));
    }

    public static TrainingReportComparisonActionPlanExport fromMap(Map<String, ?> map) {
        if (map == null || map.isEmpty()) {
            throw new IllegalArgumentException("comparison action-plan export must not be empty");
        }
        Object schema = map.get("schema");
        if (schema != null && !SCHEMA.equals(String.valueOf(schema))) {
            throw new IllegalArgumentException("Unsupported comparison action-plan export schema: " + schema);
        }
        TrainingReportActionPlan actionPlan =
                TrainingReportActionPlan.fromMap(TrainingReportValues.mapValue(map, "actionPlan"));
        TrainingReportRuntimeRegressionSummary runtimeRegression =
                TrainingReportRuntimeRegressionSummary.fromMap(TrainingReportValues.mapValue(map, "runtimeRegression"));
        return new TrainingReportComparisonActionPlanExport(
                actionPlan,
                runtimeRegression,
                markdownValue(map.get("markdown")));
    }

    public static TrainingReportComparisonActionPlanExport fromJson(String json) {
        Object parsed = TrainerJsonParser.parse(json == null ? "" : json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("comparison action-plan export JSON must be an object");
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            @SuppressWarnings("unchecked")
            Map<String, Object> typedMap = (Map<String, Object>) snapshotMap;
            return fromMap(typedMap);
        }
        throw new IllegalArgumentException("comparison action-plan export JSON must be an object");
    }

    public static TrainingReportComparisonActionPlanExport readJson(Path jsonFile) throws IOException {
        Objects.requireNonNull(jsonFile, "jsonFile must not be null");
        return fromJson(Files.readString(jsonFile, StandardCharsets.UTF_8));
    }

    public static String readMarkdown(Path markdownFile) throws IOException {
        Objects.requireNonNull(markdownFile, "markdownFile must not be null");
        return Files.readString(markdownFile, StandardCharsets.UTF_8);
    }

    public boolean regressed() {
        return runtimeRegression.regressed();
    }

    private static String markdownValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    public boolean requiresAttention() {
        return actionPlan.requiresAttention();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("schema", SCHEMA);
        map.put("actionPlan", actionPlan.toMap());
        map.put("runtimeRegression", runtimeRegression.toMap());
        map.put("markdown", markdown);
        map.put("regressed", regressed());
        map.put("requiresAttention", requiresAttention());
        return Map.copyOf(map);
    }

    public Map<String, Object> artifactManifest(Path jsonFile, Path markdownFile) throws IOException {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("schema", ARTIFACT_MANIFEST_SCHEMA);
        manifest.put("exportSchema", SCHEMA);
        manifest.put("regressed", regressed());
        manifest.put("requiresAttention", requiresAttention());
        manifest.put("json", TrainingReportArtifactFingerprint.of(jsonFile).toMap());
        manifest.put("markdown", TrainingReportArtifactFingerprint.of(markdownFile).toMap());
        return Map.copyOf(manifest);
    }

    public String toJson() {
        return TrainerJson.toJson(toMap());
    }

    public void writeJson(Path jsonFile) throws IOException {
        Objects.requireNonNull(jsonFile, "jsonFile must not be null");
        TrainerCheckpointIO.writeStringAtomically(jsonFile, toJson() + "\n");
    }

    public void writeMarkdown(Path markdownFile) throws IOException {
        Objects.requireNonNull(markdownFile, "markdownFile must not be null");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, markdown);
    }
}
