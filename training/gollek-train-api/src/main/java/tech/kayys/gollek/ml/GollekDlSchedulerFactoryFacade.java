package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.optim.CosineAnnealingWarmRestartsLR;
import tech.kayys.gollek.ml.optim.CosineAnnealingLR;
import tech.kayys.gollek.ml.optim.ExponentialLR;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.OneCycleLR;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.ReduceLROnPlateau;
import tech.kayys.gollek.ml.optim.SequentialLR;
import tech.kayys.gollek.ml.optim.StepLR;
import tech.kayys.gollek.ml.optim.WarmupCosineScheduler;

import java.util.List;

/**
 * Learning-rate scheduler construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlSchedulerFactoryFacade extends GollekDlOptimizerFactoryFacade {
    protected GollekDlSchedulerFactoryFacade() {
    }

    public static StepLR stepScheduler(
            Optimizer optimizer,
            int stepSize,
            float gamma) {
        return new StepLR(optimizer, stepSize, gamma);
    }

    public static CosineAnnealingLR cosineAnnealingScheduler(
            Optimizer optimizer,
            int tMax,
            float minLr) {
        return new CosineAnnealingLR(optimizer, tMax, minLr);
    }

    public static ExponentialLR exponentialScheduler(
            Optimizer optimizer,
            float gamma) {
        return new ExponentialLR(optimizer, gamma);
    }

    public static WarmupCosineScheduler warmupCosineScheduler(
            Optimizer optimizer,
            int warmupSteps,
            int totalSteps,
            float maxLr,
            float minLr) {
        return new WarmupCosineScheduler(optimizer, warmupSteps, totalSteps, maxLr, minLr);
    }

    public static CosineAnnealingWarmRestartsLR cosineAnnealingWarmRestartsScheduler(
            Optimizer optimizer,
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        return new CosineAnnealingWarmRestartsLR(optimizer, firstCycleSteps, cycleMultiplier, minLr);
    }

    public static OneCycleLR oneCycleScheduler(
            Optimizer optimizer,
            int totalSteps,
            float maxLr) {
        return new OneCycleLR(optimizer, totalSteps, maxLr);
    }

    public static OneCycleLR oneCycleScheduler(
            Optimizer optimizer,
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor,
            OneCycleLR.AnnealStrategy annealStrategy) {
        return new OneCycleLR(
                optimizer,
                totalSteps,
                maxLr,
                pctStart,
                divFactor,
                finalDivFactor,
                annealStrategy);
    }

    public static ReduceLROnPlateau reduceLrOnPlateauScheduler(
            Optimizer optimizer,
            ReduceLROnPlateau.Mode mode,
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        return new ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, cooldown, minLr);
    }

    public static SequentialLR sequentialScheduler(
            Optimizer optimizer,
            List<? extends LRScheduler> schedulers,
            int... milestones) {
        return new SequentialLR(optimizer, List.copyOf(schedulers), milestones);
    }
}
