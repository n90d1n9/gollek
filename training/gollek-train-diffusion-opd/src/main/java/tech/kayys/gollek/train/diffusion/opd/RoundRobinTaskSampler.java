package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;

final class RoundRobinTaskSampler {
    private final List<DiffusionTask> tasks;
    private int nextIndex;

    RoundRobinTaskSampler(List<DiffusionTask> tasks) {
        this.tasks = List.copyOf(Objects.requireNonNull(tasks, "tasks must not be null"));
        if (this.tasks.isEmpty()) {
            throw new IllegalArgumentException("tasks must contain at least one entry");
        }
    }

    DiffusionTask next() {
        DiffusionTask task = tasks.get(nextIndex);
        nextIndex = (nextIndex + 1) % tasks.size();
        return task;
    }
}
