package tech.kayys.gollek.train.diffusion.api;

import java.util.Map;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Runtime observer hook for diffusion OPD integrations that need to collect
 * diagnostics during the training loop and surface them through the standard
 * summary/history artifacts.
 */
public interface DiffusionOpdRuntimeObserver {

    default void onConditioningResolved(
            DiffusionOpdSession session,
            int round,
            DiffusionTask task,
            DiffusionPromptSample sample,
            Tensor conditioning) {
    }

    default void onStep(
            DiffusionOpdSession session,
            int round,
            DiffusionTask task,
            int timestep,
            String teacherKey,
            double stepLoss) {
    }

    default Map<String, Object> summaryMetadata() {
        return Map.of();
    }

    default Map<String, Object> roundHistoryMetadata() {
        return Map.of();
    }
}
