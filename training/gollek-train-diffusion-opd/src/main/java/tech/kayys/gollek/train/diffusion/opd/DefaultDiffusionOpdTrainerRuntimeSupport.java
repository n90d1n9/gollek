package tech.kayys.gollek.train.diffusion.opd;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBinding;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Owns trainer listener/observer coordination and runtime metadata shaping.
 *
 * <p>This helper covers the runtime-facing notification boundary after rollout
 * decisions are made, while persistence and adaptive weighting stay in their
 * sibling trainer helpers.
 */
final class DefaultDiffusionOpdTrainerRuntimeSupport {

    private DefaultDiffusionOpdTrainerRuntimeSupport() {
    }

    static void notifyTrainingStart(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdListener> listeners) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTrainingStart(session);
        }
    }

    static void notifyRoundStart(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdListener> listeners,
            int round) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onRoundStart(session, round);
        }
    }

    static void notifyTaskStart(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdListener> listeners,
            int round,
            DiffusionTask task) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTaskStart(session, round, task);
        }
    }

    static void notifyStep(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdRuntimeObserver> runtimeObservers,
            List<DiffusionOpdListener> listeners,
            int round,
            DiffusionTask task,
            int timestep,
            String teacherKey,
            double stepLoss) {
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            observer.onStep(session, round, task, timestep, teacherKey, stepLoss);
        }
        for (DiffusionOpdListener listener : listeners) {
            listener.onStep(session, round, task, timestep, teacherKey, stepLoss);
        }
    }

    static void notifyConditioningResolved(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdRuntimeObserver> runtimeObservers,
            int round,
            DiffusionTask task,
            DiffusionPromptSample sample,
            Tensor conditioning) {
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            observer.onConditioningResolved(session, round, task, sample, conditioning);
        }
    }

    static void notifyRoundEnd(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdListener> listeners,
            int round,
            double meanLoss,
            int optimizationSteps) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onRoundEnd(session, round, meanLoss, optimizationSteps);
        }
    }

    static void notifyTrainingEnd(
            DefaultDiffusionOpdTrainer session,
            List<DiffusionOpdListener> listeners,
            TrainingSummary summary) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTrainingEnd(session, summary);
        }
    }

    static Map<String, List<Map<String, Object>>> describeTeacherBindings(List<DiffusionTask> tasks) {
        Map<String, List<Map<String, Object>>> bindingsByTask = new LinkedHashMap<>();
        for (DiffusionTask task : tasks) {
            List<Map<String, Object>> bindings = new ArrayList<>();
            for (DiffusionTeacherBinding binding : task.teacherBindings()) {
                bindings.add(Map.of(
                        "teacherKey", binding.teacherKey(),
                        "stageName", binding.stageName(),
                        "startStepInclusive", binding.startStepInclusive(),
                        "endStepExclusive", binding.endStepExclusive(),
                        "lossWeight", binding.lossWeight()));
            }
            bindingsByTask.put(task.id(), List.copyOf(bindings));
        }
        return Map.copyOf(bindingsByTask);
    }

    static Map<String, Object> collectRuntimeSummaryMetadata(
            List<DiffusionOpdRuntimeObserver> runtimeObservers) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            metadata.putAll(observer.summaryMetadata());
        }
        return Map.copyOf(metadata);
    }

    static Map<String, Object> collectRuntimeRoundHistoryMetadata(
            List<DiffusionOpdRuntimeObserver> runtimeObservers) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            metadata.putAll(observer.roundHistoryMetadata());
        }
        return Map.copyOf(metadata);
    }
}
