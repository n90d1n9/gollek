package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerCheckpointMetadataGuardsTest {

    @Test
    void metadataMatchIgnoresMissingAndEqualValues() {
        List<String> mismatches = new ArrayList<>();

        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "optimizer",
                Map.of(),
                "trainerOptimizerClass",
                "ExpectedOptimizer",
                "optimizer class",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));
        TrainerCheckpointMetadataGuards.requireMetadataMatch(
                "optimizer",
                Map.of("trainerOptimizerClass", "ExpectedOptimizer"),
                "trainerOptimizerClass",
                "ExpectedOptimizer",
                "optimizer class",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));

        assertEquals(List.of(), mismatches);
    }

    @Test
    void metadataMatchRecordsMismatchBeforeThrowing() {
        List<String> mismatches = new ArrayList<>();

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerCheckpointMetadataGuards.requireMetadataMatch(
                        "optimizer",
                        Map.of("trainerOptimizerClass", "LoadedOptimizer"),
                        "trainerOptimizerClass",
                        "ExpectedOptimizer",
                        "optimizer class",
                        (artifact, reason) -> mismatches.add(artifact + ": " + reason)));

        assertEquals(
                "Checkpoint metadata mismatch: optimizer class mismatch "
                        + "(expected ExpectedOptimizer, got LoadedOptimizer)",
                error.getMessage());
        assertEquals(
                List.of("optimizer: optimizer class mismatch (expected ExpectedOptimizer, got LoadedOptimizer)"),
                mismatches);
    }

    @Test
    void schedulerStepUnitAcceptsMissingAndMatchingValues() {
        TrainerCheckpointMetadataGuards.requireSchedulerStepUnit(Map.of(), "EPOCH");
        TrainerCheckpointMetadataGuards.requireSchedulerStepUnit(
                Map.of("trainerSchedulerStepUnit", "BATCH"),
                "BATCH");
    }

    @Test
    void schedulerStepUnitRejectsMismatch() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerCheckpointMetadataGuards.requireSchedulerStepUnit(
                        Map.of("trainerSchedulerStepUnit", "BATCH"),
                        "EPOCH"));

        assertEquals(
                "Invalid scheduler checkpoint payload: step unit mismatch (expected EPOCH, got BATCH)",
                error.getMessage());
    }

    @Test
    void mixedPrecisionAcceptsMissingBooleanTrueAndStringTrue() {
        TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(Map.of());
        TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(
                Map.of("trainerMixedPrecisionEnabled", Boolean.TRUE));
        TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(
                Map.of("trainerMixedPrecisionEnabled", "TRUE"));
    }

    @Test
    void mixedPrecisionRejectsFalseValues() {
        IllegalArgumentException booleanError = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(
                        Map.of("trainerMixedPrecisionEnabled", Boolean.FALSE)));
        IllegalArgumentException stringError = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerCheckpointMetadataGuards.requireMixedPrecisionEnabled(
                        Map.of("trainerMixedPrecisionEnabled", "false")));

        assertEquals(
                "Invalid GradScaler checkpoint payload: mixed precision was not enabled",
                booleanError.getMessage());
        assertEquals(booleanError.getMessage(), stringError.getMessage());
    }
}
