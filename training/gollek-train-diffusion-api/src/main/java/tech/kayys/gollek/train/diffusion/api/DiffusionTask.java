package tech.kayys.gollek.train.diffusion.api;

import java.util.List;
import java.util.Objects;

/**
 * A task-specific teacher and prompt pool used during multi-task OPD.
 */
public record DiffusionTask(
        String id,
        String name,
        String rewardFamily,
        String teacherModelId,
        List<DiffusionTeacherBinding> teacherBindings,
        List<DiffusionPromptSample> promptSamples) {

    public DiffusionTask {
        id = Objects.requireNonNull(id, "id must not be null");
        name = Objects.requireNonNull(name, "name must not be null");
        rewardFamily = rewardFamily == null ? "unknown" : rewardFamily;
        teacherModelId = teacherModelId == null ? "" : teacherModelId;
        teacherBindings = teacherBindings == null ? List.of() : List.copyOf(teacherBindings);
        promptSamples = List.copyOf(Objects.requireNonNull(promptSamples, "promptSamples must not be null"));
    }

    public DiffusionTask(
            String id,
            String name,
            String rewardFamily,
            String teacherModelId,
            List<DiffusionPromptSample> promptSamples) {
        this(id, name, rewardFamily, teacherModelId, List.of(), promptSamples);
    }
}
