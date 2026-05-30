package tech.kayys.gollek.ml;

import static tech.kayys.gollek.ml.GollekTrainingOptionValidators.normalizeBestModelMonitorMetric;
import static tech.kayys.gollek.ml.GollekTrainingOptionValidators.normalizeNonNegativeDouble;

import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Core training controls shared by training option builders.
 */
public abstract class GollekTrainingCoreBuilderSupport<B extends GollekTrainingCoreBuilderSupport<B>>
        extends GollekTrainingMetricsBuilderSupport<B> {
    protected double gradientClip = 1.0;
    protected double gradientClipValue = 0.0;
    protected int earlyStoppingPatience = 0;
    protected double earlyStoppingMinDelta = 0.0;
    protected String earlyStoppingMonitorMetric;
    protected CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode =
            CanonicalTrainer.BestModelMonitorMode.MIN;
    protected Path checkpointDir;
    protected boolean resumeFromCheckpoint = false;
    protected boolean failOnCheckpointLoadError = true;
    protected boolean saveBestModelCheckpoint = true;
    protected boolean restoreBestModelAtEnd = false;
    protected String bestModelMonitorMetric;
    protected CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode =
            CanonicalTrainer.BestModelMonitorMode.MIN;
    protected int gradientAccumulationSteps = 1;
    protected boolean mixedPrecision = false;
    protected GradScaler gradScaler;
    protected String device = "auto";
    protected boolean failOnAcceleratorFallback = false;
    protected boolean dataDistributionDiagnostics = false;

    public B gradientClip(double gradientClip) {
        this.gradientClip = normalizeNonNegativeDouble(gradientClip);
        return self();
    }

    public B gradientClipByValue(double maxAbsValue) {
        this.gradientClipValue = normalizeNonNegativeDouble(maxAbsValue);
        return self();
    }

    public B gradientValueClip(double maxAbsValue) {
        return gradientClipByValue(maxAbsValue);
    }

    public B earlyStopping(int patience) {
        return earlyStopping(patience, earlyStoppingMinDelta);
    }

    public B earlyStopping(int patience, double minDelta) {
        this.earlyStoppingPatience = Math.max(0, patience);
        this.earlyStoppingMinDelta = Math.max(0.0, minDelta);
        return self();
    }

    public B earlyStoppingMonitorMetric(String metricName) {
        return earlyStoppingMonitorMetric(metricName, CanonicalTrainer.BestModelMonitorMode.MAX);
    }

    public B earlyStoppingMonitorMetric(
            String metricName,
            CanonicalTrainer.BestModelMonitorMode mode) {
        this.earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(metricName);
        this.earlyStoppingMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MAX : mode;
        return self();
    }

    public B earlyStoppingMonitorValidationLoss() {
        this.earlyStoppingMonitorMetric = null;
        this.earlyStoppingMonitorMode = CanonicalTrainer.BestModelMonitorMode.MIN;
        return self();
    }

    public B checkpointDir(Path checkpointDir) {
        this.checkpointDir = checkpointDir;
        return self();
    }

    public B resumeFromCheckpoint() {
        return resumeFromCheckpoint(true);
    }

    public B resumeFromCheckpoint(boolean resumeFromCheckpoint) {
        this.resumeFromCheckpoint = resumeFromCheckpoint;
        return self();
    }

    public B failOnCheckpointLoadError(boolean failOnCheckpointLoadError) {
        this.failOnCheckpointLoadError = failOnCheckpointLoadError;
        return self();
    }

    public B saveBestModelCheckpoint(boolean saveBestModelCheckpoint) {
        this.saveBestModelCheckpoint = saveBestModelCheckpoint;
        return self();
    }

    public B bestModelCheckpoint(boolean saveBestModelCheckpoint) {
        return saveBestModelCheckpoint(saveBestModelCheckpoint);
    }

    public B restoreBestModelAtEnd() {
        return restoreBestModelAtEnd(true);
    }

    public B restoreBestModelAtEnd(boolean restoreBestModelAtEnd) {
        this.restoreBestModelAtEnd = restoreBestModelAtEnd;
        if (restoreBestModelAtEnd) {
            this.saveBestModelCheckpoint = true;
        }
        return self();
    }

    public B bestModelMonitorMetric(String metricName) {
        return bestModelMonitorMetric(metricName, CanonicalTrainer.BestModelMonitorMode.MAX);
    }

    public B bestModelMonitorMetric(
            String metricName,
            CanonicalTrainer.BestModelMonitorMode mode) {
        this.bestModelMonitorMetric = normalizeBestModelMonitorMetric(metricName);
        this.bestModelMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MAX : mode;
        return self();
    }

    public B bestModelMonitorValidationLoss() {
        return bestModelMonitorValidationLoss(CanonicalTrainer.BestModelMonitorMode.MIN);
    }

    public B bestModelMonitorValidationLoss(CanonicalTrainer.BestModelMonitorMode mode) {
        this.bestModelMonitorMetric = null;
        this.bestModelMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MIN : mode;
        return self();
    }

    public B gradientAccumulationSteps(int gradientAccumulationSteps) {
        this.gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
        return self();
    }

    public B mixedPrecision() {
        return mixedPrecision(true);
    }

    public B mixedPrecision(boolean mixedPrecision) {
        this.mixedPrecision = mixedPrecision;
        if (!mixedPrecision) {
            this.gradScaler = null;
        }
        return self();
    }

    public B gradScaler(GradScaler gradScaler) {
        this.gradScaler = Objects.requireNonNull(gradScaler, "gradScaler must not be null");
        this.mixedPrecision = true;
        return self();
    }

    public B mixedPrecision(GradScaler gradScaler) {
        return gradScaler(gradScaler);
    }

    public B device(String device) {
        this.device = device == null || device.isBlank() ? "auto" : device.trim();
        return self();
    }

    public B accelerator(String device) {
        return device(device);
    }

    public B failOnAcceleratorFallback(boolean failOnAcceleratorFallback) {
        this.failOnAcceleratorFallback = failOnAcceleratorFallback;
        return self();
    }

    public B requireAccelerator() {
        return failOnAcceleratorFallback(true);
    }

    public B dataDistributionDiagnostics() {
        return dataDistributionDiagnostics(true);
    }

    public B dataDistributionDiagnostics(boolean enabled) {
        this.dataDistributionDiagnostics = enabled;
        return self();
    }
}
