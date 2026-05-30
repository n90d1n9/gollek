package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

class TrainerTensorDiagnosticsTest {

    @Test
    void tensorsSummarizeCountsL2NormAndMaxAbs() {
        TensorDiagnostics diagnostics = TrainerTensorDiagnostics.tensors(List.of(
                GradTensor.of(new float[] {3f, 0f, -4f}, 3),
                GradTensor.of(new float[] {12f}, 1)));

        assertEquals(2, diagnostics.tensorCount());
        assertEquals(4L, diagnostics.valueCount());
        assertEquals(1L, diagnostics.zeroCount());
        assertEquals(3L, diagnostics.nonZeroCount());
        assertEquals(4L, diagnostics.finiteCount());
        assertEquals(0L, diagnostics.nonFiniteCount());
        assertEquals(0L, diagnostics.nanCount());
        assertEquals(0L, diagnostics.positiveInfinityCount());
        assertEquals(0L, diagnostics.negativeInfinityCount());
        assertEquals(0.0, diagnostics.nonFiniteFraction(), 1e-9);
        assertEquals(0.25, diagnostics.zeroFraction(), 1e-9);
        assertEquals(19.0, diagnostics.sumAbs(), 1e-6);
        assertEquals(4.75, diagnostics.meanAbs(), 1e-6);
        assertEquals(6.5, diagnostics.rms(), 1e-6);
        assertEquals(13.0, diagnostics.l2Norm(), 1e-6);
        assertEquals(12.0, diagnostics.maxAbs(), 1e-6);
    }

    @Test
    void parametersSummarizeParameterData() {
        List<Parameter> parameters = List.of(
                new Parameter(GradTensor.of(new float[] {1f, -2f}, 2)),
                new Parameter(GradTensor.of(new float[] {2f}, 1)));

        TensorDiagnostics diagnostics = TrainerTensorDiagnostics.parameters(parameters);

        assertEquals(2, diagnostics.tensorCount());
        assertEquals(3L, diagnostics.valueCount());
        assertEquals(0L, diagnostics.zeroCount());
        assertEquals(0.0, diagnostics.zeroFraction(), 1e-9);
        assertEquals(5.0, diagnostics.sumAbs(), 1e-6);
        assertEquals(5.0 / 3.0, diagnostics.meanAbs(), 1e-6);
        assertEquals(Math.sqrt(3.0), diagnostics.rms(), 1e-6);
        assertEquals(3.0, diagnostics.l2Norm(), 1e-6);
        assertEquals(2.0, diagnostics.maxAbs(), 1e-6);
    }

    @Test
    void parameterUpdatesSummarizeExactDeltasFromSnapshot() {
        List<Parameter> parameters = List.of(
                new Parameter(GradTensor.of(new float[] {1f, -2f}, 2)),
                new Parameter(GradTensor.of(new float[] {3f}, 1)));
        List<float[]> beforeStep = TrainerTensorDiagnostics.snapshotParameters(parameters);
        parameters.get(0).data().data()[0] = 1.5f;
        parameters.get(0).data().data()[1] = -2f;
        parameters.get(1).data().data()[0] = 1f;

        TensorDiagnostics diagnostics = TrainerTensorDiagnostics.parameterUpdates(parameters, beforeStep);

        assertEquals(2, diagnostics.tensorCount());
        assertEquals(3L, diagnostics.valueCount());
        assertEquals(1L, diagnostics.zeroCount());
        assertEquals(2.5, diagnostics.sumAbs(), 1e-6);
        assertEquals(Math.sqrt(4.25), diagnostics.l2Norm(), 1e-6);
        assertEquals(2.0, diagnostics.maxAbs(), 1e-6);
        assertEquals(2.5 / 3.0, diagnostics.meanAbs(), 1e-6);
        assertEquals(Math.sqrt(4.25) / Math.sqrt(3.0), diagnostics.rms(), 1e-6);
    }

    @Test
    void parameterUpdatesRequireMatchingSnapshotShape() {
        List<Parameter> parameters = List.of(new Parameter(GradTensor.of(new float[] {1f, 2f}, 2)));

        assertThrows(IllegalArgumentException.class,
                () -> TrainerTensorDiagnostics.parameterUpdates(parameters, List.of()));
        assertThrows(IllegalStateException.class,
                () -> TrainerTensorDiagnostics.parameterUpdates(parameters, List.of(new float[] {1f})));
    }

    @Test
    void gradientsSkipParametersWithoutGradients() {
        Parameter withGradient = new Parameter(GradTensor.of(new float[] {1f, 2f}, 2));
        withGradient.data().backward(GradTensor.of(new float[] {6f, 8f}, 2));
        Parameter withoutGradient = new Parameter(GradTensor.of(new float[] {99f}, 1));

        TensorDiagnostics diagnostics = TrainerTensorDiagnostics.gradients(List.of(withGradient, withoutGradient));

        assertEquals(1, diagnostics.tensorCount());
        assertEquals(2L, diagnostics.valueCount());
        assertEquals(0L, diagnostics.zeroCount());
        assertEquals(10.0, diagnostics.l2Norm(), 1e-6);
        assertEquals(8.0, diagnostics.maxAbs(), 1e-6);
    }

    @Test
    void scaleGradientsScalesExistingGradientsAndSkipsMissingOnes() {
        Parameter withGradient = new Parameter(GradTensor.of(new float[] {1f, 2f}, 2));
        withGradient.data().backward(GradTensor.of(new float[] {4f, -6f}, 2));
        Parameter withoutGradient = new Parameter(GradTensor.of(new float[] {99f}, 1));

        TrainerTensorDiagnostics.scaleGradients(List.of(withGradient, withoutGradient), 0.25f);

        assertEquals(1.0f, withGradient.grad().data()[0], 1e-6f);
        assertEquals(-1.5f, withGradient.grad().data()[1], 1e-6f);
        assertNull(withoutGradient.grad());
    }

    @Test
    void tensorsPreserveNonFiniteDiagnosticsForTrainerGuard() {
        TensorDiagnostics diagnostics = TrainerTensorDiagnostics.tensors(List.of(
                GradTensor.of(new float[] {Float.POSITIVE_INFINITY, Float.NaN, Float.NEGATIVE_INFINITY, 0f}, 4)));

        assertEquals(1, diagnostics.tensorCount());
        assertEquals(4L, diagnostics.valueCount());
        assertEquals(1L, diagnostics.zeroCount());
        assertEquals(1L, diagnostics.finiteCount());
        assertEquals(3L, diagnostics.nonFiniteCount());
        assertEquals(1L, diagnostics.nanCount());
        assertEquals(1L, diagnostics.positiveInfinityCount());
        assertEquals(1L, diagnostics.negativeInfinityCount());
        assertEquals(0.75, diagnostics.nonFiniteFraction(), 1e-9);
        assertTrue(Double.isNaN(diagnostics.representativeNonFiniteValue()));
        assertTrue(Double.isNaN(diagnostics.l2Norm()));
        assertTrue(Double.isNaN(diagnostics.maxAbs()));
    }

    @Test
    void requireFiniteAcceptsFiniteDiagnostics() {
        RecordingFailures failures = new RecordingFailures();

        TrainerTensorDiagnostics.requireFinite(
                new TensorDiagnostics(1, 2L, 0L, 4.0, 3.0, 2.0, 0L, 0L, 0L),
                "train",
                "gradient",
                true,
                failures);

        assertFalse(failures.recorded);
    }

    @Test
    void requireFiniteRecordsAndThrowsForNonFiniteDiagnostics() {
        RecordingFailures failures = new RecordingFailures();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerTensorDiagnostics.requireFinite(
                        new TensorDiagnostics(1, 2L, 0L, 4.0, Double.NaN, 2.0, 1L, 0L, 0L),
                        "train",
                        "gradient",
                        true,
                        failures));

        assertEquals("train gradient must be finite, got NaN", error.getMessage());
        assertTrue(failures.recorded);
        assertEquals("train", failures.phase);
        assertEquals("gradient", failures.kind);
        assertTrue(failures.optimizerStepSkipped);
        assertFalse(failures.discarded);
    }

    private static final class RecordingFailures implements TrainerBatchGuards.FailureRecorder {
        private boolean recorded;
        private String phase;
        private String kind;
        private boolean optimizerStepSkipped;
        private boolean discarded;

        @Override
        public String invalidBatch(
                String phase,
                String reason,
                String message,
                boolean optimizerStepSkipped) {
            throw new AssertionError("not used");
        }

        @Override
        public String invalidLossShape(
                String phase,
                String shape,
                long elements,
                boolean optimizerStepSkipped) {
            throw new AssertionError("not used");
        }

        @Override
        public String nonFinite(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped) {
            this.recorded = true;
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
