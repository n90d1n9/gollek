package tech.kayys.gollek.train.diffusion.api;

import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Listener hooks for Java-side DiffusionOPD runs.
 */
public interface DiffusionOpdListener {

    default void onTrainingStart(DiffusionOpdSession session) {
    }

    default void onRoundStart(DiffusionOpdSession session, int round) {
    }

    default void onTaskStart(DiffusionOpdSession session, int round, DiffusionTask task) {
    }

    default void onStep(
            DiffusionOpdSession session,
            int round,
            DiffusionTask task,
            int timestep,
            String teacherKey,
            double stepLoss) {
    }

    default void onRoundEnd(
            DiffusionOpdSession session,
            int round,
            double meanLoss,
            int optimizationSteps) {
    }

    default void onTrainingEnd(DiffusionOpdSession session, TrainingSummary summary) {
    }
}
