package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * One-object trainer provenance setup for recursive-reasoning token datasets.
 */
public record DiscreteTokenDatasetTrainerProvenanceSpec(
        Path checkpointDir,
        DiscreteTokenDatasetPlan plan,
        DiscreteTokenDatasetPlanReport report,
        DiscreteTokenDatasetCheckpointManifest manifest,
        DiscreteTokenDatasetCheckpointResumePolicy resumePolicy,
        boolean failOnResumeRejection,
        boolean writeManifestOnStart,
        boolean writeResumeReportOnStart) {

    public DiscreteTokenDatasetTrainerProvenanceSpec {
        checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
        plan = Objects.requireNonNull(plan, "plan must not be null");
        report = Objects.requireNonNull(report, "report must not be null");
        manifest = Objects.requireNonNull(manifest, "manifest must not be null");
        resumePolicy = Objects.requireNonNull(resumePolicy, "resumePolicy must not be null");
        manifest.verifyReport(report).requireMatch();
        report.verifyFingerprint(plan.fingerprint()).requireMatch();
    }

    public static Builder builder(Path checkpointDir, DiscreteTokenDatasetPlan plan) {
        return new Builder(checkpointDir, plan);
    }

    public Path manifestPath() {
        return DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir);
    }

    public Path resumeReportPath() {
        return DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir);
    }

    public Path checkpointRootDir() {
        Path parent = checkpointDir.getParent();
        return parent == null ? checkpointDir : parent;
    }

    public DiscreteTokenDatasetCheckpointResumeExpectation resumeExpectation() {
        return resumePolicy.expectation();
    }

    public DiscreteTokenDatasetPlanReadinessGate currentPlanGate() {
        return resumePolicy.currentPlanGate();
    }

    public DiscreteTokenDatasetCheckpointResumeCompatibilityMode resumeCompatibilityMode() {
        return resumePolicy.compatibilityMode();
    }

    public DiscreteTokenDatasetTrainerProvenanceListener listener() {
        return DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, manifest)
                .currentPlan(plan)
                .resumePolicy(resumePolicy)
                .failOnResumeRejection(failOnResumeRejection)
                .writeManifestOnStart(writeManifestOnStart)
                .writeResumeReportOnStart(writeResumeReportOnStart)
                .build();
    }

    public void writeManifest() throws IOException {
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
    }

    public Map<String, Object> readManifestMetadata() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.readManifestMetadata(checkpointDir);
    }

    public DiscreteTokenDatasetCheckpointManifestSnapshot readSnapshot() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.readSnapshot(checkpointDir);
    }

    public Map<String, Object> readResumeReportMetadata() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.readResumeReportMetadata(checkpointDir);
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot readResumeReportSnapshot() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.readResumeReportSnapshot(checkpointDir);
    }

    public DiscreteTokenDatasetTrainerCheckpointSnapshot readCheckpointSnapshot() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(checkpointDir);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy restoreSelectionPolicy() {
        return restoreSelectionPolicy(DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady());
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy restoreSelectionPolicy(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy basePolicy) {
        Objects.requireNonNull(basePolicy, "basePolicy must not be null");
        return basePolicy.withExpectation(resumeExpectation());
    }

    public DiscreteTokenDatasetTrainerCheckpointInventory scanCheckpointInventory() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.scanCheckpointInventory(checkpointRootDir());
    }

    public DiscreteTokenDatasetTrainerCheckpointInspectionReport inspectRestoreCandidates() throws IOException {
        return inspectRestoreCandidates(restoreSelectionPolicy());
    }

    public DiscreteTokenDatasetTrainerCheckpointInspectionReport inspectRestoreCandidates(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        Objects.requireNonNull(policy, "policy must not be null");
        return DiscreteTokenDatasetTrainerCheckpointBridge.inspectCheckpoints(checkpointRootDir(), policy);
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointRestorePlan> selectRestorePlan() throws IOException {
        return inspectRestoreCandidates().selectedRestorePlan();
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePlan requireRestorePlan() throws IOException {
        return inspectRestoreCandidates().requireRestorePlan();
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluateRestorePreflight() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.evaluateRestorePreflight(
                requireRestorePlan(),
                plan,
                resumePolicy);
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireRestorePreflightReady() throws IOException {
        return evaluateRestorePreflight().requireReady();
    }

    public DiscreteTokenDatasetCheckpointResumeReport evaluateResume() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.evaluateResume(checkpointDir, plan, resumePolicy);
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireResumeReady() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReady(checkpointDir, plan, resumePolicy);
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireResumeReadyAndWriteReport() throws IOException {
        return DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(
                checkpointDir,
                plan,
                resumePolicy);
    }

    public Map<String, Object> trainingReportMetadata() {
        return DiscreteTokenDatasetTrainerCheckpointBridge.trainingReportMetadata(manifest);
    }

    public Map<String, Object> trainingReportMetadata(
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        return DiscreteTokenDatasetTrainerCheckpointBridge.trainingReportMetadata(manifest, resumeReport);
    }

    public Map<String, Object> attachToTrainingMetadata(Map<String, ?> trainingMetadata) {
        return DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingMetadata(
                trainingMetadata,
                manifest);
    }

    public Map<String, Object> attachToTrainingMetadata(
            Map<String, ?> trainingMetadata,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        return DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingMetadata(
                trainingMetadata,
                manifest,
                resumeReport);
    }

    public TrainingSummary attachToTrainingSummary(TrainingSummary summary) {
        return DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(summary, manifest);
    }

    public TrainingSummary attachToTrainingSummary(
            TrainingSummary summary,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        return DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(
                summary,
                manifest,
                resumeReport);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("checkpointDir", checkpointDir.toString());
        metadata.put("checkpointRootDir", checkpointRootDir().toString());
        metadata.put("manifestFileName", DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME);
        metadata.put("resumeReportFileName", DiscreteTokenDatasetTrainerCheckpointBridge.RESUME_REPORT_FILE_NAME);
        metadata.put("report", report.toMetadata());
        metadata.put("manifest", manifest.toMetadata());
        metadata.put("lineage", manifest.lineage().toMetadata());
        metadata.put("resumePolicy", resumePolicy.toMetadata());
        metadata.put("restoreSelectionPolicy", restoreSelectionPolicy().toMetadata());
        metadata.put("restorePreflightAvailable", true);
        metadata.put("failOnResumeRejection", failOnResumeRejection);
        metadata.put("writeManifestOnStart", writeManifestOnStart);
        metadata.put("writeResumeReportOnStart", writeResumeReportOnStart);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public static final class Builder {
        private final Path checkpointDir;
        private final DiscreteTokenDatasetPlan plan;
        private DiscreteTokenDatasetPlanReadinessGate gate = DiscreteTokenDatasetPlanReadinessGate.training();
        private DiscreteTokenDatasetPlanReadinessGate resumeGate;
        private String experimentName = "default";
        private String runId;
        private String modelFamily = "recursive-reasoning";
        private long seed;
        private long checkpointStep;
        private long createdAtEpochMillis = System.currentTimeMillis();
        private String createdBy = "gollek";
        private Map<String, Object> attributes = Map.of();
        private DiscreteTokenDatasetCheckpointLineage lineage;
        private DiscreteTokenDatasetCheckpointResumeExpectation resumeExpectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.none();
        private DiscreteTokenDatasetCheckpointResumeCompatibilityMode resumeCompatibilityMode =
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT;
        private boolean exactResumeExpectation;
        private boolean failOnResumeRejection = true;
        private boolean writeManifestOnStart = true;
        private boolean writeResumeReportOnStart = true;

        private Builder(Path checkpointDir, DiscreteTokenDatasetPlan plan) {
            this.checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
            this.plan = Objects.requireNonNull(plan, "plan must not be null");
        }

        public Builder gate(DiscreteTokenDatasetPlanReadinessGate gate) {
            this.gate = Objects.requireNonNull(gate, "gate must not be null");
            return this;
        }

        public Builder strict() {
            this.gate = DiscreteTokenDatasetPlanReadinessGate.strict();
            this.resumeGate = null;
            this.resumeCompatibilityMode = DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT;
            return this;
        }

        public Builder lenient() {
            this.gate = DiscreteTokenDatasetPlanReadinessGate.lenient();
            this.resumeGate = null;
            this.resumeCompatibilityMode = DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE;
            return this;
        }

        public Builder trainingGate() {
            this.gate = DiscreteTokenDatasetPlanReadinessGate.training();
            this.resumeGate = null;
            this.resumeCompatibilityMode = DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT;
            return this;
        }

        public Builder resumeGate(DiscreteTokenDatasetPlanReadinessGate resumeGate) {
            this.resumeGate = Objects.requireNonNull(resumeGate, "resumeGate must not be null");
            return this;
        }

        public Builder resumeCompatibilityMode(
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode resumeCompatibilityMode) {
            this.resumeCompatibilityMode = Objects.requireNonNull(
                    resumeCompatibilityMode,
                    "resumeCompatibilityMode must not be null");
            return this;
        }

        public Builder compatibleResume() {
            this.resumeCompatibilityMode = DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE;
            if (resumeGate == null) {
                this.resumeGate = DiscreteTokenDatasetPlanReadinessGate.lenient();
            }
            return this;
        }

        public Builder forceResume() {
            this.resumeCompatibilityMode = DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE;
            if (resumeGate == null) {
                this.resumeGate = DiscreteTokenDatasetPlanReadinessGate.lenient();
            }
            return this;
        }

        public Builder experimentName(String experimentName) {
            this.experimentName = experimentName;
            return this;
        }

        public Builder runId(String runId) {
            this.runId = runId;
            return this;
        }

        public Builder modelFamily(String modelFamily) {
            this.modelFamily = modelFamily;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder checkpointStep(long checkpointStep) {
            this.checkpointStep = checkpointStep;
            return this;
        }

        public Builder createdAtEpochMillis(long createdAtEpochMillis) {
            this.createdAtEpochMillis = createdAtEpochMillis;
            return this;
        }

        public Builder createdBy(String createdBy) {
            this.createdBy = createdBy;
            return this;
        }

        public Builder attributes(Map<String, Object> attributes) {
            this.attributes = attributes;
            return this;
        }

        public Builder lineage(DiscreteTokenDatasetCheckpointLineage lineage) {
            this.lineage = Objects.requireNonNull(lineage, "lineage must not be null");
            return this;
        }

        public Builder lineageFrom(DiscreteTokenDatasetCheckpointManifestSnapshot parent) {
            this.lineage = DiscreteTokenDatasetCheckpointLineage.resumedFrom(parent);
            return this;
        }

        public Builder lineageFrom(
                DiscreteTokenDatasetCheckpointManifestSnapshot parent,
                Map<String, Object> attributes) {
            this.lineage = DiscreteTokenDatasetCheckpointLineage.resumedFrom(parent, attributes);
            return this;
        }

        public Builder resumeExpectation(DiscreteTokenDatasetCheckpointResumeExpectation resumeExpectation) {
            this.resumeExpectation = Objects.requireNonNull(
                    resumeExpectation,
                    "resumeExpectation must not be null");
            this.exactResumeExpectation = false;
            return this;
        }

        public Builder exactResumeExpectation() {
            this.exactResumeExpectation = true;
            return this;
        }

        public Builder noResumeExpectation() {
            this.resumeExpectation = DiscreteTokenDatasetCheckpointResumeExpectation.none();
            this.exactResumeExpectation = false;
            return this;
        }

        public Builder failOnResumeRejection(boolean failOnResumeRejection) {
            this.failOnResumeRejection = failOnResumeRejection;
            return this;
        }

        public Builder writeManifestOnStart(boolean writeManifestOnStart) {
            this.writeManifestOnStart = writeManifestOnStart;
            return this;
        }

        public Builder writeResumeReportOnStart(boolean writeResumeReportOnStart) {
            this.writeResumeReportOnStart = writeResumeReportOnStart;
            return this;
        }

        public DiscreteTokenDatasetTrainerProvenanceSpec build() {
            DiscreteTokenDatasetPlanReport report = plan.report(gate);
            DiscreteTokenDatasetCheckpointManifest manifest = DiscreteTokenDatasetCheckpointManifest.builder(report)
                    .experimentName(experimentName)
                    .runId(runId == null ? report.fingerprint().shortValue() : runId)
                    .modelFamily(modelFamily)
                    .seed(seed)
                    .checkpointStep(checkpointStep)
                    .createdAtEpochMillis(createdAtEpochMillis)
                    .createdBy(createdBy)
                    .lineage(lineage == null
                            ? DiscreteTokenDatasetCheckpointLineage.root(runId == null
                                    ? report.fingerprint().shortValue()
                                    : runId)
                            : lineage)
                    .attributes(attributes)
                    .build();
            DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                    exactResumeExpectation
                            ? DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest)
                            : resumeExpectation;
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy =
                    new DiscreteTokenDatasetCheckpointResumePolicy(
                            resumeGate == null ? gate : resumeGate,
                            expectation,
                            resumeCompatibilityMode);
            return new DiscreteTokenDatasetTrainerProvenanceSpec(
                    checkpointDir,
                    plan,
                    report,
                    manifest,
                    resumePolicy,
                    failOnResumeRejection,
                    writeManifestOnStart,
                    writeResumeReportOnStart);
        }
    }
}
