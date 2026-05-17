package tech.kayys.gollek.trainer.api;

import java.nio.file.Path;

/**
 * Canonical training runtime configuration shared across trainer modules.
 */
public record TrainerConfig(
        int epochs,
        double gradientClip,
        boolean mixedPrecision,
        Path checkpointDir) {
}
