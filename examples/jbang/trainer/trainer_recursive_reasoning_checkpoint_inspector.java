///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-recursive-reasoning:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointMetadataJson;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointLineageHealthSnapshot;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointLineageHealthValidationReport;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointResumeCompatibilityMode;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointResumePolicy;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanDiagnosticsPolicy;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanReadinessGate;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanReport;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointResumeExpectation;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointBridge;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointInspectionReport;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointRestorePreflight;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointSelectionPolicy;

/**
 * JBang inspector for recursive-reasoning trainer checkpoint sidecars.
 */
public class trainer_recursive_reasoning_checkpoint_inspector {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            throw new IllegalArgumentException("""
                    Pass a checkpoint root directory.
                    Examples:
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestReady candidates
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints strictLatestResumeReady restore json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineage json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageChain json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageGraph json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealth json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchema json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthValidation json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthValidationSchema json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaCatalog json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemas json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaCatalogSchema json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaLock json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaLockSchema json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaLockValidation schemaLock=/tmp/schema-lock.json json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaLockValidation schemaLock=/tmp/schema-lock.json requireSchemaLockValid=true json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady lineageHealthSchemaLockValidationSchema json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints latestResumeReady overview requireLineageHealthy=true
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints preflight currentReport=/tmp/current-plan-report.json json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints explain currentReport=/tmp/current-plan-report.json
                      jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java /tmp/checkpoints explain currentReport=/tmp/current-plan-report.json resumeMode=force json
                    """);
        }

        CliOptions options = parseArgs(args);
        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                DiscreteTokenDatasetTrainerCheckpointBridge.inspectCheckpoints(
                        options.rootDir(),
                        options.policy());
        Object value = select(report, options);
        String format = normalizeFormat(options.format(), options.outputPath());
        String rendered = renderValue(value, format);

        System.out.println("====================================================");
        System.out.println(" Gollek Recursive Reasoning Checkpoint Inspector");
        System.out.println("====================================================");
        System.out.println("rootDir=" + options.rootDir().toAbsolutePath().normalize());
        System.out.println("section=" + options.section());
        System.out.println("policy=" + options.policy().summary());
        System.out.println("resumeMode=" + options.resumeCompatibilityMode().id());
        System.out.println("selectionSatisfied=" + report.selectionSatisfied());
        System.out.println("selectedRunId=" + selectedRunId(report));
        if (options.currentReportPath() != null) {
            System.out.println("currentReport=" + options.currentReportPath().toAbsolutePath().normalize());
        }
        if (options.schemaLockPath() != null) {
            System.out.println("schemaLock=" + options.schemaLockPath().toAbsolutePath().normalize());
        }
        if (options.requireSchemaLockValid()) {
            System.out.println("requireSchemaLockValid=true");
        }
        System.out.println("format=" + format);

        if (options.outputPath() != null) {
            writeOutput(options.outputPath(), rendered);
            System.out.println("wroteOutput=" + options.outputPath().toAbsolutePath().normalize());
            return;
        }
        System.out.print(rendered);
    }

    private static Object select(
            DiscreteTokenDatasetTrainerCheckpointInspectionReport report,
            CliOptions options) {
        String section = options.section();
        String normalized = section == null || section.isBlank()
                ? "overview"
                : section.toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "all", "report" -> report.toMetadata();
            case "overview", "status", "summary", "ci" -> overview(report, "ci".equals(normalized));
            case "selected" -> selected(report);
            case "restore", "restoreplan", "handoff" -> report.requireRestorePlan().toMetadata();
            case "preflight", "restorepreflight", "resumecheck" -> restorePreflight(report, options).toMetadata();
            case "explain", "explanation", "why" -> restorePreflight(report, options).explanation().toMetadata();
            case "lineage", "ancestry" -> report.requireRestorePlan().lineageMetadata();
            case "lineagechain", "ancestrychain", "chain" -> report.requireSelectedLineageChain().toMetadata();
            case "lineagegraph", "ancestrygraph", "lineages" -> report.lineageGraph().toMetadata();
            case "lineagehealth", "ancestryhealth", "health" -> report.lineageGraph().healthMetadata();
            case "lineagehealthschema", "healthschema", "ancestryhealthschema" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.jsonSchemaMetadata();
            case "lineagehealthvalidation", "healthvalidation", "validatehealth", "validatelineagehealth" ->
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport
                            .fromMetadata(report.lineageGraph().healthMetadata())
                            .toMetadata();
            case "lineagehealthvalidationschema", "healthvalidationschema", "validatehealthschema",
                    "validatelineagehealthschema" ->
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.jsonSchemaMetadata();
            case "lineagehealthschemacatalog", "healthschemacatalog" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(false);
            case "lineagehealthschemas", "healthschemas" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(true);
            case "lineagehealthschemacatalogschema", "healthschemacatalogschema", "lineagehealthschemasschema",
                    "healthschemasschema" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.jsonSchemaMetadata();
            case "lineagehealthschemalock", "healthschemalock", "lineagehealthlock", "healthlock" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata();
            case "lineagehealthschemalockschema", "healthschemalockschema", "lineagehealthlockschema",
                    "healthlockschema" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockJsonSchemaMetadata();
            case "lineagehealthschemalockvalidation", "healthschemalockvalidation", "lineagehealthlockvalidation",
                    "healthlockvalidation" ->
                    schemaLockValidation(options);
            case "lineagehealthschemalockvalidationschema", "healthschemalockvalidationschema",
                    "lineagehealthlockvalidationschema", "healthlockvalidationschema" ->
                    DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockValidationJsonSchemaMetadata();
            case "paths", "restorepaths" -> report.requireRestorePlan().pathsMetadata();
            case "identity", "restoreidentity" -> report.requireRestorePlan().identityMetadata();
            case "candidates", "decisions" -> report.checkpointDecisions().stream()
                    .map(DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision::toMetadata)
                    .toList();
            case "accepted" -> report.acceptedDecisions().stream()
                    .map(DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision::toMetadata)
                    .toList();
            case "rejected", "rejections" -> report.rejectedDecisions().stream()
                    .map(DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision::toMetadata)
                    .toList();
            case "failures" -> report.inventory().failures().stream()
                    .map(failure -> failure.toMetadata())
                    .toList();
            case "inventory" -> report.inventory().toMetadata();
            case "policy" -> report.policy().toMetadata();
            default -> throw new IllegalArgumentException(
                    "Unknown section '" + section + "'. Use overview, ci, selected, restore, preflight, explain, lineage, lineageChain, lineageGraph, lineageHealth, lineageHealthSchema, lineageHealthValidation, lineageHealthValidationSchema, lineageHealthSchemaCatalog, lineageHealthSchemas, lineageHealthSchemaCatalogSchema, lineageHealthSchemaLock, lineageHealthSchemaLockSchema, lineageHealthSchemaLockValidation, lineageHealthSchemaLockValidationSchema, paths, identity, candidates, accepted, rejected, failures, inventory, policy, or report.");
        };
    }

    private static DiscreteTokenDatasetTrainerCheckpointRestorePreflight restorePreflight(
            DiscreteTokenDatasetTrainerCheckpointInspectionReport report,
            CliOptions options) {
        if (options.currentReportPath() == null) {
            throw new IllegalArgumentException(
                    "Section 'preflight' or 'explain' requires currentReport=/path/to/current-plan-report-or-manifest.json");
        }
        DiscreteTokenDatasetPlanReport currentReport = readCurrentPlanReport(options.currentReportPath());
        DiscreteTokenDatasetCheckpointResumePolicy resumePolicy = new DiscreteTokenDatasetCheckpointResumePolicy(
                new DiscreteTokenDatasetPlanReadinessGate(
                        DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults(),
                        currentReport.readiness().failOnWarnings()),
                report.policy().expectation(),
                options.resumeCompatibilityMode());
        return DiscreteTokenDatasetTrainerCheckpointBridge.evaluateRestorePreflight(
                report.requireRestorePlan(),
                currentReport,
                resumePolicy);
    }

    private static Map<String, Object> schemaLockValidation(CliOptions options) {
        Map<String, Object> validation;
        try {
            validation = options.schemaLockPath() == null
                    ? DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock()
                    : DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockPath(
                            options.schemaLockPath());
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read schema lock file: "
                            + options.schemaLockPath().toAbsolutePath().normalize(),
                    e);
        }
        if (options.requireSchemaLockValid()) {
            DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.requireValidLockValidation(validation);
        }
        return validation;
    }

    private static Map<String, Object> overview(
            DiscreteTokenDatasetTrainerCheckpointInspectionReport report,
            boolean ci) {
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("summary", ci ? "short" : report.summary());
        overview.put("rootDir", report.rootDir().toString());
        overview.put("selectionSatisfied", report.selectionSatisfied());
        overview.put("selectedCheckpointPresent", report.selected());
        overview.put("selectedRunId", selectedRunId(report));
        overview.put("acceptedCount", report.acceptedCount());
        overview.put("rejectedCount", report.rejectedCount());
        overview.put("checkpointCount", report.inventory().checkpointCount());
        overview.put("failureCount", report.inventory().failureCount());
        overview.put("policy", report.policy().summary());
        report.selectionFailure().ifPresent(failure -> overview.put("selectionFailure", failure));
        if (!ci) {
            overview.put("inventory", report.inventory().summary());
            overview.put("candidates", report.checkpointDecisions().stream()
                    .map(DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision::summary)
                    .toList());
        }
        return overview;
    }

    private static Map<String, Object> selected(
            DiscreteTokenDatasetTrainerCheckpointInspectionReport report) {
        Map<String, Object> selected = new LinkedHashMap<>();
        selected.put("selectedCheckpointPresent", report.selected());
        selected.put("selectionSatisfied", report.selectionSatisfied());
        selected.put("selectedRunId", selectedRunId(report));
        report.selectionFailure().ifPresent(failure -> selected.put("selectionFailure", failure));
        report.selectedCheckpoint().ifPresent(checkpoint -> selected.put("checkpoint", checkpoint.toMetadata()));
        report.selectedRestorePlan().ifPresent(plan -> selected.put("restorePlan", plan.toMetadata()));
        return selected;
    }

    private static String selectedRunId(DiscreteTokenDatasetTrainerCheckpointInspectionReport report) {
        return report.selectedCheckpoint()
                .map(checkpoint -> checkpoint.manifest().runId())
                .orElse("none");
    }

    private static CliOptions parseArgs(String[] args) {
        Path rootDir = Path.of(args[0]);
        SelectionKind selectionKind = SelectionKind.LATEST_RESUME_READY;
        Boolean requireReady = null;
        Boolean requireResumeReport = null;
        Boolean failOnInventoryFailures = null;
        Boolean failOnLineageIssues = null;
        String section = "overview";
        String format = null;
        Path outputPath = null;
        Path currentReportPath = null;
        Path schemaLockPath = null;
        boolean requireSchemaLockValid = false;
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode resumeCompatibilityMode =
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT;
        DiscreteTokenDatasetCheckpointResumeExpectation.Builder expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder();

        for (int index = 1; index < args.length; index++) {
            String token = args[index];
            String normalized = token.toLowerCase(Locale.ROOT);
            if (isFormatToken(normalized)) {
                format = normalized;
                continue;
            }
            SelectionKind parsedKind = SelectionKind.fromToken(normalized);
            if (parsedKind != null) {
                selectionKind = parsedKind;
                continue;
            }
            if (isSectionToken(normalized)) {
                section = normalized;
                continue;
            }
            if (token.contains("=")) {
                ParsedAssignment assignment = ParsedAssignment.parse(token);
                switch (assignment.key()) {
                    case "selector", "policy" -> selectionKind = SelectionKind.require(assignment.value());
                    case "section" -> section = assignment.value().toLowerCase(Locale.ROOT);
                    case "format" -> format = assignment.value().toLowerCase(Locale.ROOT);
                    case "output", "out" -> outputPath = Path.of(assignment.value());
                    case "currentreport", "currentplanreport", "planreport" ->
                            currentReportPath = Path.of(assignment.value());
                    case "schemalock", "lock", "expectedlock", "lineagehealthschemalock" ->
                            schemaLockPath = Path.of(assignment.value());
                    case "requireschemalockvalid", "requireschemalock", "failonschemalockinvalid",
                            "failschemalockinvalid" ->
                            requireSchemaLockValid = parseBoolean(assignment.value(), assignment.key());
                    case "resumemode", "compatibilitymode", "compatibility" ->
                            resumeCompatibilityMode =
                                    DiscreteTokenDatasetCheckpointResumeCompatibilityMode.fromId(assignment.value());
                    case "requireready" -> requireReady = parseBoolean(assignment.value(), assignment.key());
                    case "requireresumereport" -> requireResumeReport = parseBoolean(assignment.value(), assignment.key());
                    case "failoninventoryfailures" -> failOnInventoryFailures = parseBoolean(assignment.value(), assignment.key());
                    case "failonlineageissues", "failonlineagehealth", "requirehealthylineage", "requirelineagehealthy" ->
                            failOnLineageIssues = parseBoolean(assignment.value(), assignment.key());
                    case "experiment", "experimentname" -> expectation.experimentName(assignment.value());
                    case "run", "runid" -> expectation.runId(assignment.value());
                    case "family", "modelfamily" -> expectation.modelFamily(assignment.value());
                    case "seed" -> expectation.seed(parseLong(assignment.value(), assignment.key()));
                    case "step", "checkpointstep" -> expectation.checkpointStep(parseLong(assignment.value(), assignment.key()));
                    case "minstep", "minimumcheckpointstep" ->
                            expectation.minimumCheckpointStep(parseLong(assignment.value(), assignment.key()));
                    default -> throw new IllegalArgumentException("Unknown option: " + token);
                }
                continue;
            }
            outputPath = Path.of(token);
        }

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy = selectionKind.policy();
        if (requireReady != null) {
            policy = policy.withRequireReady(requireReady);
        }
        if (requireResumeReport != null) {
            policy = policy.withRequireResumeReport(requireResumeReport);
        }
        if (failOnInventoryFailures != null) {
            policy = policy.withFailOnInventoryFailures(failOnInventoryFailures);
        }
        if (failOnLineageIssues != null) {
            policy = policy.withFailOnLineageIssues(failOnLineageIssues);
        }
        DiscreteTokenDatasetCheckpointResumeExpectation builtExpectation = expectation.build();
        if (builtExpectation.active()) {
            policy = policy.withExpectation(builtExpectation);
        }
        return new CliOptions(
                rootDir,
                section,
                format,
                outputPath,
                currentReportPath,
                schemaLockPath,
                requireSchemaLockValid,
                resumeCompatibilityMode,
                policy);
    }

    private static String normalizeFormat(String format, Path outputPath) {
        if (format != null && !format.isBlank()) {
            return format.toLowerCase(Locale.ROOT);
        }
        if (outputPath != null && outputPath.getFileName() != null
                && outputPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".json")) {
            return "json";
        }
        return "text";
    }

    private static String renderValue(Object value, String format) {
        if ("json".equals(format)) {
            return DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(jsonRoot(value))
                    + System.lineSeparator();
        }
        if (value instanceof Map<?, ?> map) {
            StringBuilder builder = new StringBuilder();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                builder.append(entry.getKey())
                        .append('=')
                        .append(entry.getValue())
                        .append(System.lineSeparator());
            }
            return builder.toString();
        }
        if (value instanceof List<?> list) {
            StringBuilder builder = new StringBuilder();
            for (Object item : list) {
                if (item instanceof Map<?, ?> map) {
                    Object summary = map.containsKey("summary") ? map.get("summary") : map;
                    builder.append("- ")
                            .append(summary)
                            .append(System.lineSeparator());
                } else {
                    builder.append("- ").append(item).append(System.lineSeparator());
                }
            }
            return builder.toString();
        }
        return String.valueOf(value) + System.lineSeparator();
    }

    private static Map<String, Object> jsonRoot(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            map.forEach((key, element) -> copy.put(String.valueOf(key), element));
            return copy;
        }
        if (value instanceof List<?> list) {
            return Map.of("items", list);
        }
        return Map.of("value", value);
    }

    private static void writeOutput(Path outputPath, String rendered) {
        try {
            if (outputPath.getParent() != null) {
                Files.createDirectories(outputPath.getParent());
            }
            Files.writeString(outputPath, rendered);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to write output file: " + outputPath.toAbsolutePath().normalize(), e);
        }
    }

    private static boolean isFormatToken(String token) {
        return "text".equals(token) || "json".equals(token);
    }

    private static boolean isSectionToken(String token) {
        return switch (token) {
            case "overview", "status", "summary", "ci", "selected", "candidates", "decisions",
                    "restore", "restoreplan", "handoff", "preflight", "restorepreflight", "resumecheck",
                    "explain", "explanation", "why", "lineage", "ancestry",
                    "lineagechain", "ancestrychain", "chain", "lineagegraph", "ancestrygraph", "lineages",
                    "lineagehealth", "ancestryhealth", "health",
                    "lineagehealthschema", "healthschema", "ancestryhealthschema",
                    "lineagehealthvalidation", "healthvalidation", "validatehealth", "validatelineagehealth",
                    "lineagehealthvalidationschema", "healthvalidationschema", "validatehealthschema",
                    "validatelineagehealthschema",
                    "lineagehealthschemacatalog", "healthschemacatalog",
                    "lineagehealthschemas", "healthschemas",
                    "lineagehealthschemacatalogschema", "healthschemacatalogschema", "lineagehealthschemasschema",
                    "healthschemasschema",
                    "lineagehealthschemalock", "healthschemalock", "lineagehealthlock", "healthlock",
                    "lineagehealthschemalockschema", "healthschemalockschema", "lineagehealthlockschema",
                    "healthlockschema",
                    "lineagehealthschemalockvalidation", "healthschemalockvalidation",
                    "lineagehealthlockvalidation", "healthlockvalidation",
                    "lineagehealthschemalockvalidationschema", "healthschemalockvalidationschema",
                    "lineagehealthlockvalidationschema", "healthlockvalidationschema",
                    "paths", "restorepaths", "identity", "restoreidentity",
                    "accepted", "rejected", "rejections", "failures", "inventory", "policy", "report", "all" -> true;
            default -> false;
        };
    }

    private static boolean parseBoolean(String value, String name) {
        if ("true".equalsIgnoreCase(value)) {
            return true;
        }
        if ("false".equalsIgnoreCase(value)) {
            return false;
        }
        throw new IllegalArgumentException(name + " must be true or false");
    }

    private static long parseLong(String value, String name) {
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be an integer", e);
        }
    }

    private static DiscreteTokenDatasetPlanReport readCurrentPlanReport(Path path) {
        try {
            Map<String, Object> metadata = DiscreteTokenDatasetCheckpointMetadataJson.read(path);
            return DiscreteTokenDatasetPlanReport.fromMetadata(planReportMetadata(metadata));
        } catch (IOException e) {
            throw new IllegalStateException(
                "Failed to read current report file: " + path.toAbsolutePath().normalize(), e);
        }
    }

    private static Map<?, ?> planReportMetadata(Map<String, Object> metadata) {
        if (metadata.containsKey("fingerprint") && metadata.containsKey("readiness")) {
            return metadata;
        }
        Object directReport = firstMap(
                metadata,
                "datasetPlanReport",
                "report",
                "currentPlanReport",
                "planReport");
        if (directReport instanceof Map<?, ?> map) {
            return map;
        }
        Object currentResumeReport = metadata.get("currentResumeReport");
        if (currentResumeReport instanceof Map<?, ?> resumeMap) {
            Object nested = resumeMap.get("currentPlanReport");
            if (nested instanceof Map<?, ?> nestedMap) {
                return nestedMap;
            }
        }
        throw new IllegalArgumentException(
                "currentReport must be a DiscreteTokenDatasetPlanReport JSON, a checkpoint manifest, a provenance spec metadata file, or a restore preflight JSON");
    }

    private static Object firstMap(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Object value = metadata.get(key);
            if (value instanceof Map<?, ?>) {
                return value;
            }
        }
        return null;
    }

    private record CliOptions(
            Path rootDir,
            String section,
            String format,
            Path outputPath,
            Path currentReportPath,
            Path schemaLockPath,
            boolean requireSchemaLockValid,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode resumeCompatibilityMode,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {}

    private record ParsedAssignment(String key, String value) {
        static ParsedAssignment parse(String token) {
            int index = token.indexOf('=');
            String key = token.substring(0, index).trim().toLowerCase(Locale.ROOT).replace("-", "");
            String value = token.substring(index + 1).trim();
            if (key.isBlank() || value.isBlank()) {
                throw new IllegalArgumentException("Malformed option: " + token);
            }
            return new ParsedAssignment(key, value);
        }
    }

    private enum SelectionKind {
        LATEST,
        LATEST_READY,
        LATEST_RESUME_READY,
        STRICT_LATEST_READY,
        STRICT_LATEST_RESUME_READY;

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy() {
            return switch (this) {
                case LATEST -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest();
                case LATEST_READY -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady();
                case LATEST_RESUME_READY -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady();
                case STRICT_LATEST_READY -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestReady();
                case STRICT_LATEST_RESUME_READY ->
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestResumeReady();
            };
        }

        static SelectionKind require(String token) {
            SelectionKind kind = fromToken(token.toLowerCase(Locale.ROOT));
            if (kind == null) {
                throw new IllegalArgumentException("Unknown checkpoint selection policy: " + token);
            }
            return kind;
        }

        static SelectionKind fromToken(String token) {
            return switch (token.replace("-", "")) {
                case "latest", "any" -> LATEST;
                case "latestready", "ready" -> LATEST_READY;
                case "latestresumeready", "resumeready", "resume" -> LATEST_RESUME_READY;
                case "strictlatestready", "strictready" -> STRICT_LATEST_READY;
                case "strictlatestresumeready", "strictresumeready", "strictresume" -> STRICT_LATEST_RESUME_READY;
                default -> null;
            };
        }
    }
}
