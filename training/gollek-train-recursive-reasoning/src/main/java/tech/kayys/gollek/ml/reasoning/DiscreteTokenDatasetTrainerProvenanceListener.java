package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Training listener that keeps recursive-reasoning dataset provenance beside checkpoints.
 */
public final class DiscreteTokenDatasetTrainerProvenanceListener implements TrainingListener {
    private final Path checkpointDir;
    private final DiscreteTokenDatasetCheckpointManifest manifest;
    private final DiscreteTokenDatasetPlan currentPlan;
    private final DiscreteTokenDatasetCheckpointResumePolicy resumePolicy;
    private final boolean failOnResumeRejection;
    private final boolean writeManifestOnStart;
    private final boolean writeResumeReportOnStart;

    private DiscreteTokenDatasetCheckpointResumeReport lastResumeReport;
    private TrainingSummary lastEnrichedSummary;

    private DiscreteTokenDatasetTrainerProvenanceListener(Builder builder) {
        this.checkpointDir = Objects.requireNonNull(builder.checkpointDir, "checkpointDir must not be null");
        this.manifest = Objects.requireNonNull(builder.manifest, "manifest must not be null");
        this.currentPlan = builder.currentPlan;
        this.resumePolicy = Objects.requireNonNull(builder.resumePolicy, "resumePolicy must not be null");
        this.failOnResumeRejection = builder.failOnResumeRejection;
        this.writeManifestOnStart = builder.writeManifestOnStart;
        this.writeResumeReportOnStart = builder.writeResumeReportOnStart;
    }

    public static Builder builder(
            Path checkpointDir,
            DiscreteTokenDatasetCheckpointManifest manifest) {
        return new Builder(checkpointDir, manifest);
    }

    public static DiscreteTokenDatasetTrainerProvenanceListener create(
            Path checkpointDir,
            DiscreteTokenDatasetCheckpointManifest manifest) {
        return builder(checkpointDir, manifest).build();
    }

    public static DiscreteTokenDatasetTrainerProvenanceListener withResumePreflight(
            Path checkpointDir,
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        return builder(checkpointDir, manifest)
                .currentPlan(currentPlan)
                .resumePolicy(resumePolicy)
                .build();
    }

    public Path checkpointDir() {
        return checkpointDir;
    }

    public DiscreteTokenDatasetCheckpointManifest manifest() {
        return manifest;
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeReport> lastResumeReport() {
        return Optional.ofNullable(lastResumeReport);
    }

    public Optional<TrainingSummary> lastEnrichedSummary() {
        return Optional.ofNullable(lastEnrichedSummary);
    }

    public Map<String, Object> trainingReportMetadata() {
        return DiscreteTokenDatasetTrainerCheckpointBridge.trainingReportMetadata(manifest, lastResumeReport);
    }

    @Override
    public void onTrainingStart(TrainerSession session) {
        try {
            preflightExistingManifest();
            if (writeManifestOnStart) {
                DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
            }
        } catch (IOException error) {
            throw new UncheckedIOException("Failed to persist recursive-reasoning dataset provenance", error);
        }
    }

    @Override
    public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
        lastEnrichedSummary = DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(
                summary,
                manifest,
                lastResumeReport);
    }

    private void preflightExistingManifest() throws IOException {
        if (currentPlan == null || !Files.isRegularFile(
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir))) {
            return;
        }

        lastResumeReport = DiscreteTokenDatasetTrainerCheckpointBridge.evaluateResume(
                checkpointDir,
                currentPlan,
                resumePolicy);
        if (writeResumeReportOnStart) {
            DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(checkpointDir, lastResumeReport);
        }
        if (failOnResumeRejection) {
            lastResumeReport.requireReady();
        }
    }

    public static final class Builder {
        private final Path checkpointDir;
        private final DiscreteTokenDatasetCheckpointManifest manifest;
        private DiscreteTokenDatasetPlan currentPlan;
        private DiscreteTokenDatasetCheckpointResumePolicy resumePolicy =
                DiscreteTokenDatasetCheckpointResumePolicy.training();
        private boolean failOnResumeRejection = true;
        private boolean writeManifestOnStart = true;
        private boolean writeResumeReportOnStart = true;

        private Builder(Path checkpointDir, DiscreteTokenDatasetCheckpointManifest manifest) {
            this.checkpointDir = checkpointDir;
            this.manifest = manifest;
        }

        public Builder currentPlan(DiscreteTokenDatasetPlan currentPlan) {
            this.currentPlan = Objects.requireNonNull(currentPlan, "currentPlan must not be null");
            return this;
        }

        public Builder resumePolicy(DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
            this.resumePolicy = Objects.requireNonNull(resumePolicy, "resumePolicy must not be null");
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

        public DiscreteTokenDatasetTrainerProvenanceListener build() {
            return new DiscreteTokenDatasetTrainerProvenanceListener(this);
        }
    }
}
