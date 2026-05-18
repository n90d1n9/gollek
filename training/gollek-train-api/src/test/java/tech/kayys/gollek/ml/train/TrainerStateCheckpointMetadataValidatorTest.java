package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerStateCheckpointMetadataValidatorTest {

    @Test
    void optimizerMetadataAcceptsMissingAndMatchingValues() {
        List<String> mismatches = new ArrayList<>();

        TrainerStateCheckpointMetadataValidator.requireOptimizer(
                Map.of(),
                "SGD",
                "weight:[1]:1;",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));
        TrainerStateCheckpointMetadataValidator.requireOptimizer(
                Map.of(
                        "trainerOptimizerClass", "SGD",
                        "trainerOptimizerParameterSignature", "weight:[1]:1;"),
                "SGD",
                "weight:[1]:1;",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));

        assertEquals(List.of(), mismatches);
    }

    @Test
    void optimizerMetadataRecordsParameterSignatureMismatch() {
        List<String> mismatches = new ArrayList<>();

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerStateCheckpointMetadataValidator.requireOptimizer(
                        Map.of("trainerOptimizerParameterSignature", "bias:[1]:1;"),
                        "SGD",
                        "weight:[1]:1;",
                        (artifact, reason) -> mismatches.add(artifact + ": " + reason)));

        assertEquals(
                "Checkpoint metadata mismatch: optimizer parameter signature mismatch "
                        + "(expected weight:[1]:1;, got bias:[1]:1;)",
                error.getMessage());
        assertEquals(
                List.of("optimizer: optimizer parameter signature mismatch "
                        + "(expected weight:[1]:1;, got bias:[1]:1;)"),
                mismatches);
    }

    @Test
    void schedulerMetadataValidatesClassSignatureAndStepUnit() {
        List<String> mismatches = new ArrayList<>();

        TrainerStateCheckpointMetadataValidator.requireScheduler(
                Map.of(
                        "trainerSchedulerClass", "StepLR",
                        "trainerSchedulerOptimizerParameterSignature", "weight:[1]:1;",
                        "trainerSchedulerStepUnit", "EPOCH"),
                "StepLR",
                "weight:[1]:1;",
                "EPOCH",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));

        assertEquals(List.of(), mismatches);
    }

    @Test
    void schedulerMetadataRejectsStepUnitMismatch() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerStateCheckpointMetadataValidator.requireScheduler(
                        Map.of("trainerSchedulerStepUnit", "BATCH"),
                        "StepLR",
                        "weight:[1]:1;",
                        "EPOCH",
                        (artifact, reason) -> {
                        }));

        assertEquals(
                "Invalid scheduler checkpoint payload: step unit mismatch (expected EPOCH, got BATCH)",
                error.getMessage());
    }

    @Test
    void gradScalerMetadataValidatesClassAndMixedPrecisionFlag() {
        List<String> mismatches = new ArrayList<>();

        TrainerStateCheckpointMetadataValidator.requireGradScaler(
                Map.of(
                        "trainerGradScalerClass", "GradScaler",
                        "trainerMixedPrecisionEnabled", true),
                "GradScaler",
                (artifact, reason) -> mismatches.add(artifact + ": " + reason));

        assertEquals(List.of(), mismatches);
    }

    @Test
    void gradScalerMetadataRejectsDisabledMixedPrecisionFlag() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerStateCheckpointMetadataValidator.requireGradScaler(
                        Map.of("trainerMixedPrecisionEnabled", false),
                        "GradScaler",
                        (artifact, reason) -> {
                        }));

        assertEquals(
                "Invalid GradScaler checkpoint payload: mixed precision was not enabled",
                error.getMessage());
    }
}
