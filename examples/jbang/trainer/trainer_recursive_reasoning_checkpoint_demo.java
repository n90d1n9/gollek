///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-recursive-reasoning:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointLineage;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointManifest;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointResumeExpectation;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetCheckpointResumePolicy;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetExample;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlan;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanConfig;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanReadinessGate;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanReport;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetPlanner;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSplitMode;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainEpochMode;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointBridge;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointInspectionReport;
import tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetTrainerCheckpointSelectionPolicy;

/**
 * Generates sample recursive-reasoning checkpoint sidecars for the checkpoint inspector.
 */
public class trainer_recursive_reasoning_checkpoint_demo {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Path rootDir = args.length > 0
                ? Path.of(args[0])
                : Path.of("trainer_checkpoints", "recursive_reasoning_checkpoints");
        boolean includeCorrupt = !hasFlag(args, "clean");
        boolean includeUnhealthyLineage = hasFlag(args, "lineage-unhealthy") || hasFlag(args, "unhealthy-lineage");

        recreateDirectory(rootDir);

        DiscreteTokenDatasetPlan plan = plan(0L);
        writeReadyCheckpoint(rootDir.resolve("001-ready"), plan, "run-ready", 100L, 12L);
        writeManifestOnlyCheckpoint(rootDir.resolve("002-manifest-only"), plan, "run-manifest-only", 200L, 18L);
        writeBlockedCheckpoint(rootDir.resolve("003-blocked"), plan, changedPlan(), "run-blocked", 300L, 24L);
        if (includeCorrupt) {
            writeCorruptCheckpoint(rootDir.resolve("004-corrupt"));
        }
        if (includeUnhealthyLineage) {
            writeMissingParentCheckpoint(rootDir.resolve("005-missing-parent"), plan);
        }

        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                        rootDir,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady());

        System.out.println("====================================================");
        System.out.println(" Gollek Recursive Reasoning Checkpoint Demo");
        System.out.println("====================================================");
        System.out.println("rootDir=" + rootDir.toAbsolutePath().normalize());
        System.out.println("includeCorrupt=" + includeCorrupt);
        System.out.println("includeUnhealthyLineage=" + includeUnhealthyLineage);
        System.out.println("checkpointCount=" + report.inventory().checkpointCount());
        System.out.println("failureCount=" + report.inventory().failureCount());
        System.out.println("selectedRunId=" + report.selectedCheckpoint()
                .map(checkpoint -> checkpoint.manifest().runId())
                .orElse("none"));
        System.out.println("summary=" + report.summary());
        System.out.println();
        System.out.println("Try:");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady overview");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady restore json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineage json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageChain json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageGraph json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealth json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchema json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthValidation json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthValidationSchema json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaCatalog json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemas json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaCatalogSchema json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaLock json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaLockSchema json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaLockValidation json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaLockValidation requireSchemaLockValid=true json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady lineageHealthSchemaLockValidationSchema json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady overview requireLineageHealthy=true");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady preflight currentReport=\""
                + DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(rootDir.resolve("001-ready"))
                        .toAbsolutePath()
                        .normalize()
                + "\" json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady explain currentReport=\""
                + DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(rootDir.resolve("001-ready"))
                        .toAbsolutePath()
                        .normalize()
                + "\"");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady explain currentReport=\""
                + DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(rootDir.resolve("003-blocked"))
                        .toAbsolutePath()
                        .normalize()
                + "\" resumeMode=force json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestResumeReady candidates json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" latestReady selected json");
        System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                + rootDir.toAbsolutePath().normalize()
                + "\" runId=run-ready selected json");
        if (includeCorrupt) {
            System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_inspector.java \""
                    + rootDir.toAbsolutePath().normalize()
                    + "\" strictLatestResumeReady overview");
        }
        if (!includeUnhealthyLineage) {
            System.out.println("  jbang trainer/trainer_recursive_reasoning_checkpoint_demo.java \""
                    + rootDir.toAbsolutePath().normalize()
                    + "\" clean lineage-unhealthy");
        }
    }

    private static void writeReadyCheckpoint(
            Path checkpointDir,
            DiscreteTokenDatasetPlan plan,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt, checkpointStep)
                        .build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(checkpointDir, plan, policy);
    }

    private static void writeManifestOnlyCheckpoint(
            Path checkpointDir,
            DiscreteTokenDatasetPlan plan,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(
                checkpointDir,
                manifest(
                        plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()),
                        runId,
                        createdAt,
                        checkpointStep)
                        .build());
    }

    private static void writeBlockedCheckpoint(
            Path checkpointDir,
            DiscreteTokenDatasetPlan originalPlan,
            DiscreteTokenDatasetPlan changedPlan,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(
                        originalPlan.report(DiscreteTokenDatasetPlanReadinessGate.strict()),
                        runId,
                        createdAt,
                        checkpointStep)
                        .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changedPlan));
    }

    private static void writeCorruptCheckpoint(Path checkpointDir) throws IOException {
        Files.createDirectories(checkpointDir);
        Files.writeString(
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir),
                "{bad-json");
    }

    private static void writeMissingParentCheckpoint(Path checkpointDir, DiscreteTokenDatasetPlan plan)
            throws IOException {
        DiscreteTokenDatasetPlanReport report = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(report, "run-missing-parent", 500L, 36L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-origin",
                        "run-parent-missing",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of("demo", "lineage-unhealthy")))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report,
            String runId,
            long createdAt,
            long checkpointStep) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-structured-reasoning")
                .runId(runId)
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(checkpointStep)
                .createdAtEpochMillis(createdAt)
                .createdBy("trainer_recursive_reasoning_checkpoint_demo")
                .attributes(Map.of("demo", true));
    }

    private static DiscreteTokenDatasetPlan changedPlan() {
        return plan(42L);
    }

    private static DiscreteTokenDatasetPlan plan(long seed) {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("graph-coloring", 0, 1),
                        knownExample("graph-coloring", 1, 2),
                        knownExample("graph-coloring", 2, 1),
                        knownExample("graph-coloring", 3, 2),
                        knownExample("nqueens", 10, 1),
                        knownExample("nqueens", 11, 2),
                        knownExample("nqueens", 12, 1),
                        knownExample("nqueens", 13, 2)),
                new DiscreteTokenDatasetPlanConfig(
                        0.25d,
                        0.25d,
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        seed,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                        0L,
                        false));
    }

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                1,
                Map.of("inputLength", inputLength));
    }

    private static boolean hasFlag(String[] args, String flag) {
        for (int i = 1; i < args.length; i++) {
            if (flag.equalsIgnoreCase(args[i])) {
                return true;
            }
        }
        return false;
    }

    private static void recreateDirectory(Path path) throws IOException {
        if (Files.exists(path)) {
            try (var stream = Files.walk(path)) {
                for (Path entry : stream.sorted(Comparator.reverseOrder()).toList()) {
                    Files.deleteIfExists(entry);
                }
            }
        }
        Files.createDirectories(path);
    }
}
