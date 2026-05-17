package tech.kayys.gollek.train.diffusion.api;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * High-level configuration for a DiffusionOPD run.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public record DiffusionOpdConfig(
        List<DiffusionTask> tasks,
        DiffusionSamplerType samplerType,
        int batchSize,
        int gradientAccumulationSteps,
        int maxRounds,
        long seed,
        Path checkpointDir) {

    public DiffusionOpdConfig {
        tasks = List.copyOf(Objects.requireNonNull(tasks, "tasks must not be null"));
        samplerType = Objects.requireNonNull(samplerType, "samplerType must not be null");
        batchSize = Math.max(1, batchSize);
        gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
        maxRounds = Math.max(1, maxRounds);
    }
}
