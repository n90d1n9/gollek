package tech.kayys.gollek.ml;

import static tech.kayys.gollek.ml.GollekTrainingOptionValidators.normalizeBestModelMonitorMetric;

import tech.kayys.gollek.ml.optim.CosineAnnealingLR;
import tech.kayys.gollek.ml.optim.CosineAnnealingWarmRestartsLR;
import tech.kayys.gollek.ml.optim.ExponentialLR;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.OneCycleLR;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.ReduceLROnPlateau;
import tech.kayys.gollek.ml.optim.SequentialLR;
import tech.kayys.gollek.ml.optim.StepLR;
import tech.kayys.gollek.ml.optim.WarmupCosineScheduler;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Fluent learning-rate scheduler configuration shared by training option builders.
 */
public abstract class GollekTrainingSchedulerBuilderSupport<B extends GollekTrainingSchedulerBuilderSupport<B>> {
    protected Gollek.DL.SchedulerFactory schedulerFactory;
    protected CanonicalTrainer.SchedulerStepUnit schedulerStepUnit =
            CanonicalTrainer.SchedulerStepUnit.BATCH;
    protected String schedulerMonitorMetric;

    @SuppressWarnings("unchecked")
    protected final B self() {
        return (B) this;
    }

    public B scheduler(Gollek.DL.SchedulerFactory schedulerFactory) {
        return scheduler(schedulerFactory, CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B scheduler(
            Gollek.DL.SchedulerFactory schedulerFactory,
            CanonicalTrainer.SchedulerStepUnit stepUnit) {
        this.schedulerFactory = Objects.requireNonNull(
                schedulerFactory, "schedulerFactory must not be null");
        this.schedulerStepUnit = stepUnit == null
                ? CanonicalTrainer.SchedulerStepUnit.BATCH
                : stepUnit;
        return self();
    }

    public B schedulerMonitorMetric(String metricName) {
        this.schedulerMonitorMetric = normalizeBestModelMonitorMetric(metricName);
        return self();
    }

    public B schedulerMonitorValidationLoss() {
        this.schedulerMonitorMetric = null;
        return self();
    }

    public B stepLr(int stepSize, float gamma) {
        return stepLrBatches(stepSize, gamma);
    }

    public B stepLrBatches(int stepSize, float gamma) {
        return scheduler(
                optimizer -> new StepLR(optimizer, Math.max(1, stepSize), gamma),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B stepLrEpochs(int stepSize, float gamma) {
        return scheduler(
                optimizer -> new StepLR(optimizer, Math.max(1, stepSize), gamma),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B cosineAnnealingLr(int tMax, float minLr) {
        return cosineAnnealingLrBatches(tMax, minLr);
    }

    public B cosineAnnealingLrBatches(int tMax, float minLr) {
        return scheduler(
                optimizer -> new CosineAnnealingLR(
                        optimizer,
                        Math.max(1, tMax),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B cosineAnnealingLrEpochs(int tMax, float minLr) {
        return scheduler(
                optimizer -> new CosineAnnealingLR(
                        optimizer,
                        Math.max(1, tMax),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B cosineAnnealingWarmRestartsLr(
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        return cosineAnnealingWarmRestartsLrBatches(firstCycleSteps, cycleMultiplier, minLr);
    }

    public B cosineAnnealingWarmRestartsLrBatches(
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        return scheduler(
                optimizer -> new CosineAnnealingWarmRestartsLR(
                        optimizer,
                        Math.max(1, firstCycleSteps),
                        Math.max(1, cycleMultiplier),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B cosineAnnealingWarmRestartsLrEpochs(
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        return scheduler(
                optimizer -> new CosineAnnealingWarmRestartsLR(
                        optimizer,
                        Math.max(1, firstCycleSteps),
                        Math.max(1, cycleMultiplier),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B exponentialLr(float gamma) {
        return exponentialLrBatches(gamma);
    }

    public B exponentialLrBatches(float gamma) {
        return scheduler(
                optimizer -> new ExponentialLR(optimizer, Math.max(1.0e-12f, gamma)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B exponentialLrEpochs(float gamma) {
        return scheduler(
                optimizer -> new ExponentialLR(optimizer, Math.max(1.0e-12f, gamma)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B sequentialLr(
            List<? extends Gollek.DL.SchedulerFactory> schedulerFactories,
            int... milestones) {
        return sequentialLrBatches(schedulerFactories, milestones);
    }

    public B sequentialLrBatches(
            List<? extends Gollek.DL.SchedulerFactory> schedulerFactories,
            int... milestones) {
        return sequentialLr(
                schedulerFactories,
                CanonicalTrainer.SchedulerStepUnit.BATCH,
                milestones);
    }

    public B sequentialLrEpochs(
            List<? extends Gollek.DL.SchedulerFactory> schedulerFactories,
            int... milestones) {
        return sequentialLr(
                schedulerFactories,
                CanonicalTrainer.SchedulerStepUnit.EPOCH,
                milestones);
    }

    public B warmupCosineLr(int warmupSteps, int totalSteps, float minLr) {
        return warmupCosineLrBatches(warmupSteps, totalSteps, minLr);
    }

    public B warmupCosineLr(int warmupSteps, int totalSteps, float maxLr, float minLr) {
        return warmupCosineLrBatches(warmupSteps, totalSteps, maxLr, minLr);
    }

    public B warmupCosineLrBatches(int warmupSteps, int totalSteps, float minLr) {
        return scheduler(
                optimizer -> new WarmupCosineScheduler(
                        optimizer,
                        Math.max(0, warmupSteps),
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, optimizer.learningRate()),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B warmupCosineLrBatches(int warmupSteps, int totalSteps, float maxLr, float minLr) {
        return scheduler(
                optimizer -> new WarmupCosineScheduler(
                        optimizer,
                        Math.max(0, warmupSteps),
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B warmupCosineLrEpochs(int warmupSteps, int totalSteps, float minLr) {
        return scheduler(
                optimizer -> new WarmupCosineScheduler(
                        optimizer,
                        Math.max(0, warmupSteps),
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, optimizer.learningRate()),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B warmupCosineLrEpochs(int warmupSteps, int totalSteps, float maxLr, float minLr) {
        return scheduler(
                optimizer -> new WarmupCosineScheduler(
                        optimizer,
                        Math.max(0, warmupSteps),
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B oneCycleLr(int totalSteps, float maxLr) {
        return oneCycleLrBatches(totalSteps, maxLr);
    }

    public B oneCycleLrBatches(int totalSteps, float maxLr) {
        return scheduler(
                optimizer -> new OneCycleLR(
                        optimizer,
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr)),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B oneCycleLrEpochs(int totalSteps, float maxLr) {
        return scheduler(
                optimizer -> new OneCycleLR(
                        optimizer,
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr)),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B oneCycleLr(
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor) {
        return oneCycleLrBatches(
                totalSteps,
                maxLr,
                pctStart,
                divFactor,
                finalDivFactor,
                OneCycleLR.AnnealStrategy.COSINE);
    }

    public B oneCycleLrBatches(
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor) {
        return oneCycleLrBatches(
                totalSteps,
                maxLr,
                pctStart,
                divFactor,
                finalDivFactor,
                OneCycleLR.AnnealStrategy.COSINE);
    }

    public B oneCycleLrBatches(
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor,
            OneCycleLR.AnnealStrategy annealStrategy) {
        return scheduler(
                optimizer -> new OneCycleLR(
                        optimizer,
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr),
                        pctStart,
                        Math.max(1.0e-12f, divFactor),
                        Math.max(1.0e-12f, finalDivFactor),
                        annealStrategy),
                CanonicalTrainer.SchedulerStepUnit.BATCH);
    }

    public B oneCycleLrEpochs(
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor) {
        return oneCycleLrEpochs(
                totalSteps,
                maxLr,
                pctStart,
                divFactor,
                finalDivFactor,
                OneCycleLR.AnnealStrategy.COSINE);
    }

    public B oneCycleLrEpochs(
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor,
            OneCycleLR.AnnealStrategy annealStrategy) {
        return scheduler(
                optimizer -> new OneCycleLR(
                        optimizer,
                        Math.max(1, totalSteps),
                        Math.max(1.0e-12f, maxLr),
                        pctStart,
                        Math.max(1.0e-12f, divFactor),
                        Math.max(1.0e-12f, finalDivFactor),
                        annealStrategy),
                CanonicalTrainer.SchedulerStepUnit.EPOCH);
    }

    public B reduceLrOnPlateauValidationLoss(float factor, int patience, float minLr) {
        return reduceLrOnPlateauValidationLoss(factor, patience, 0.0, 0, minLr);
    }

    public B reduceLrOnPlateauValidationLoss(
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        schedulerMonitorValidationLoss();
        return scheduler(
                optimizer -> new ReduceLROnPlateau(
                        optimizer,
                        ReduceLROnPlateau.Mode.MIN,
                        factor,
                        Math.max(0, patience),
                        Math.max(0.0, threshold),
                        Math.max(0, cooldown),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.VALIDATION);
    }

    public B reduceLrOnPlateauMetric(
            String metricName,
            ReduceLROnPlateau.Mode mode,
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        schedulerMonitorMetric(metricName);
        return scheduler(
                optimizer -> new ReduceLROnPlateau(
                        optimizer,
                        mode,
                        factor,
                        Math.max(0, patience),
                        Math.max(0.0, threshold),
                        Math.max(0, cooldown),
                        Math.max(0.0f, minLr)),
                CanonicalTrainer.SchedulerStepUnit.VALIDATION);
    }

    private B sequentialLr(
            List<? extends Gollek.DL.SchedulerFactory> schedulerFactories,
            CanonicalTrainer.SchedulerStepUnit stepUnit,
            int... milestones) {
        List<Gollek.DL.SchedulerFactory> factories = copySchedulerFactories(schedulerFactories);
        int[] milestoneCopy = milestones == null ? new int[0] : Arrays.copyOf(milestones, milestones.length);
        if (milestoneCopy.length != factories.size() - 1) {
            throw new IllegalArgumentException(
                    "milestones length must be schedulerFactories.size() - 1, got: "
                            + milestoneCopy.length + " for " + factories.size() + " scheduler factories");
        }
        return scheduler(
                optimizer -> {
                    List<LRScheduler> schedulers = new ArrayList<>(factories.size());
                    for (int index = 0; index < factories.size(); index++) {
                        LRScheduler child = factories.get(index).create(optimizer);
                        if (child == null) {
                            throw new IllegalArgumentException("scheduler factory " + index + " returned null");
                        }
                        schedulers.add(child);
                    }
                    return new SequentialLR(optimizer, schedulers, Arrays.copyOf(milestoneCopy, milestoneCopy.length));
                },
                stepUnit);
    }

    private static List<Gollek.DL.SchedulerFactory> copySchedulerFactories(
            List<? extends Gollek.DL.SchedulerFactory> schedulerFactories) {
        Objects.requireNonNull(schedulerFactories, "schedulerFactories must not be null");
        if (schedulerFactories.isEmpty()) {
            throw new IllegalArgumentException("schedulerFactories must contain at least one scheduler factory");
        }
        return List.copyOf(schedulerFactories);
    }
}
