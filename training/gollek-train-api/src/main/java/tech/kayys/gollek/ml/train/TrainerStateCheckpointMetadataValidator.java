package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.function.BiConsumer;

/**
 * Artifact-level compatibility validation for state-dictionary checkpoint metadata.
 */
final class TrainerStateCheckpointMetadataValidator {
    private TrainerStateCheckpointMetadataValidator() {
    }

    static void requireOptimizer(
            Map<String, Object> state,
            String optimizerClass,
            String optimizerParameterSignature,
            BiConsumer<String, String> mismatchRecorder) {
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "optimizer",
                state,
                "trainerOptimizerClass",
                optimizerClass,
                "optimizer class",
                mismatchRecorder);
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "optimizer",
                state,
                "trainerOptimizerParameterSignature",
                optimizerParameterSignature,
                "optimizer parameter signature",
                mismatchRecorder);
    }

    static void requireScheduler(
            Map<String, Object> state,
            String schedulerClass,
            String optimizerParameterSignature,
            String schedulerStepUnit,
            BiConsumer<String, String> mismatchRecorder) {
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "scheduler",
                state,
                "trainerSchedulerClass",
                schedulerClass,
                "scheduler class",
                mismatchRecorder);
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "scheduler",
                state,
                "trainerSchedulerOptimizerParameterSignature",
                optimizerParameterSignature,
                "scheduler optimizer parameter signature",
                mismatchRecorder);
        TrainerCheckpointMetadataGuards.requireSchedulerStepUnit(state, schedulerStepUnit);
    }

    static void requireGradScaler(
            Map<String, Object> state,
            String gradScalerClass,
            BiConsumer<String, String> mismatchRecorder) {
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "gradScaler",
                state,
                "trainerGradScalerClass",
                gradScalerClass,
                "GradScaler class",
                mismatchRecorder);
        TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(state);
    }
}
