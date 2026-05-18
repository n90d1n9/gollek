package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader.Batch;

class TrainerBatchGuardsTest {

    @Test
    void toBatchRejectsMissingInputAndRequestsGradientDiscardForTraining() {
        RecordingFailures failures = new RecordingFailures();
        Batch batch = new Batch(null, GradTensor.of(new float[] {1f}, 1));

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> TrainerBatchGuards.toBatch(batch, "train", failures));

        assertEquals("train batch input tensor must not be null", error.getMessage());
        assertEquals("train", failures.phase);
        assertEquals("missing-input", failures.reason);
        assertTrue(failures.optimizerStepSkipped);
        assertTrue(failures.discarded);
    }

    @Test
    void finiteBatchGuardRejectsSampleCountMismatch() {
        RecordingFailures failures = new RecordingFailures();
        Batch batch = new Batch(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f}, 1, 1));

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> TrainerBatchGuards.requireFiniteBatchTensors(batch, "train", failures));

        assertTrue(error.getMessage().contains("train batch sample count mismatch"));
        assertEquals("sample-count-mismatch", failures.reason);
        assertTrue(failures.discarded);
    }

    @Test
    void lossGuardRejectsNonScalarValidationLossWithoutDiscard() {
        RecordingFailures failures = new RecordingFailures();
        GradTensor loss = GradTensor.of(new float[] {1f, 2f}, 2, 1);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> TrainerBatchGuards.requireUsableLoss(loss, "validation", failures));

        assertTrue(error.getMessage().contains("validation loss tensor must contain exactly one value"));
        assertEquals("[2, 1]", failures.shape);
        assertEquals(2L, failures.elements);
        assertFalse(failures.optimizerStepSkipped);
        assertFalse(failures.discarded);
    }

    @Test
    void tensorGuardRejectsNonFiniteValidationPredictionWithoutDiscard() {
        RecordingFailures failures = new RecordingFailures();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerBatchGuards.requireFiniteTensor(
                        GradTensor.of(new float[] {Float.NaN}, 1),
                        "validation",
                        "prediction",
                        false,
                        failures));

        assertTrue(error.getMessage().contains("validation prediction must be finite"));
        assertEquals("prediction", failures.kind);
        assertFalse(failures.optimizerStepSkipped);
        assertFalse(failures.discarded);
    }

    private static final class RecordingFailures implements TrainerBatchGuards.FailureRecorder {
        private String phase;
        private String reason;
        private String kind;
        private String shape;
        private long elements;
        private boolean optimizerStepSkipped;
        private boolean discarded;

        @Override
        public String invalidBatch(
                String phase,
                String reason,
                String message,
                boolean optimizerStepSkipped) {
            this.phase = phase;
            this.reason = reason;
            this.optimizerStepSkipped = optimizerStepSkipped;
            return message;
        }

        @Override
        public String invalidLossShape(
                String phase,
                String shape,
                long elements,
                boolean optimizerStepSkipped) {
            this.phase = phase;
            this.shape = shape;
            this.elements = elements;
            this.optimizerStepSkipped = optimizerStepSkipped;
            return phase + " loss tensor must contain exactly one value before backward, got shape "
                    + shape + " with " + elements + " values";
        }

        @Override
        public String nonFinite(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped) {
            this.phase = phase;
            this.kind = kind;
            this.optimizerStepSkipped = optimizerStepSkipped;
            return phase + " " + label + " must be finite, got " + value;
        }

        @Override
        public void discardPendingGradients() {
            discarded = true;
        }
    }
}
