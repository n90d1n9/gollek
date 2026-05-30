package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.train.diffusion.api.DiffusionDenoiser;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBinding;

/**
 * Resolves the active teacher binding for a task and rollout step so the main
 * trainer loop can stay focused on optimization flow instead of binding rules.
 */
final class StageAwareTeacherSelector {

    ResolvedTeacher resolve(
            DiffusionTask task,
            int stepIndex,
            Map<String, DiffusionDenoiser> teachers) {
        Objects.requireNonNull(task, "task must not be null");
        Objects.requireNonNull(teachers, "teachers must not be null");

        List<DiffusionTeacherBinding> bindings = task.teacherBindings();
        if (bindings == null || bindings.isEmpty()) {
            DiffusionDenoiser teacher = teachers.get(task.id());
            if (teacher == null) {
                throw new IllegalStateException("Missing default teacher for task: " + task.id());
            }
            return new ResolvedTeacher(task.id(), "default", 1.0d, teacher);
        }

        for (DiffusionTeacherBinding binding : bindings) {
            if (binding.matchesStep(stepIndex)) {
                DiffusionDenoiser teacher = teachers.get(binding.teacherKey());
                if (teacher == null) {
                    throw new IllegalStateException("Missing stage-aware teacher: " + binding.teacherKey());
                }
                return new ResolvedTeacher(
                        binding.teacherKey(),
                        binding.stageName(),
                        binding.lossWeight(),
                        teacher);
            }
        }

        throw new IllegalStateException(
                "No teacher binding matched step " + stepIndex + " for task " + task.id());
    }

    /**
     * Carries the resolved teacher plus the stage metadata that downstream trainer helpers reuse
     * for weighting, diagnostics, and round-history rows.
     */
    record ResolvedTeacher(String teacherKey, String stageName, double lossWeight, DiffusionDenoiser teacher) {
    }
}
