package tech.kayys.gollek.ml.train;

import java.util.Arrays;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Batch, tensor, and loss validation helpers for {@link CanonicalTrainer}.
 */
final class TrainerBatchGuards {
    private TrainerBatchGuards() {
    }

    interface FailureRecorder {
        String invalidBatch(String phase, String reason, String message, boolean optimizerStepSkipped);

        String invalidLossShape(String phase, String shape, long elements, boolean optimizerStepSkipped);

        String nonFinite(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped);

        default String nonFiniteTensor(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped,
                long totalValueCount,
                long nanCount,
                long positiveInfinityCount,
                long negativeInfinityCount) {
            return nonFinite(phase, kind, value, label, optimizerStepSkipped);
        }

        void discardPendingGradients();
    }

    static Batch toBatch(Object rawBatch, String phase, FailureRecorder failures) {
        if (rawBatch instanceof Batch batch) {
            requirePresentBatchTensors(batch, phase, failures);
            return batch;
        }
        throw new IllegalArgumentException(
                "Expected DataLoader.Batch for " + phase + " step but got: "
                        + (rawBatch == null ? "null" : rawBatch.getClass().getName()));
    }

    static double requireUsableLoss(GradTensor lossTensor, String phase, FailureRecorder failures) {
        requireSingleValueLoss(lossTensor, phase, failures);
        return requireFiniteLoss(lossTensor.item(), phase, failures);
    }

    static void requireFiniteBatchTensors(Batch batch, String phase, FailureRecorder failures) {
        boolean optimizerStepSkipped = "train".equals(phase);
        requireValidBatchStructure(batch, phase, optimizerStepSkipped, failures);
        requireFiniteTensor(batch.inputs(), phase, "input", optimizerStepSkipped, failures);
        requireFiniteTensor(batch.labels(), phase, "label", optimizerStepSkipped, failures);
    }

    static void requireFiniteTensor(
            GradTensor tensor,
            String phase,
            String kind,
            boolean optimizerStepSkipped,
            FailureRecorder failures) {
        float[] values = tensor.data();
        long nanCount = 0L;
        long positiveInfinityCount = 0L;
        long negativeInfinityCount = 0L;
        double firstNonFiniteValue = Double.NaN;
        for (float value : values) {
            if (!Float.isFinite(value)) {
                if (Double.isNaN(firstNonFiniteValue) && nanCount == 0L
                        && positiveInfinityCount == 0L && negativeInfinityCount == 0L) {
                    firstNonFiniteValue = value;
                }
                if (Float.isNaN(value)) {
                    nanCount++;
                } else if (value == Float.POSITIVE_INFINITY) {
                    positiveInfinityCount++;
                } else {
                    negativeInfinityCount++;
                }
            }
        }
        if (nanCount == 0L && positiveInfinityCount == 0L && negativeInfinityCount == 0L) {
            return;
        }
        String message = failures.nonFiniteTensor(
                phase,
                kind,
                firstNonFiniteValue,
                kind,
                optimizerStepSkipped,
                values.length,
                nanCount,
                positiveInfinityCount,
                negativeInfinityCount);
        discardGradientsIfNeeded(optimizerStepSkipped, failures);
        throw new IllegalArgumentException(message);
    }

    static String shapeString(GradTensor tensor) {
        return Arrays.toString(tensor.shape());
    }

    static long sampleCount(Batch batch) {
        long[] shape = batch.inputs().shape();
        return shape.length == 0 ? 1L : Math.max(0L, shape[0]);
    }

    private static void requireSingleValueLoss(
            GradTensor lossTensor,
            String phase,
            FailureRecorder failures) {
        if (lossTensor.numel() == 1L) {
            return;
        }
        boolean optimizerStepSkipped = "train".equals(phase);
        String shape = shapeString(lossTensor);
        long elements = lossTensor.numel();
        String message = failures.invalidLossShape(phase, shape, elements, optimizerStepSkipped);
        discardGradientsIfNeeded(optimizerStepSkipped, failures);
        throw new IllegalArgumentException(message);
    }

    private static double requireFiniteLoss(
            double value,
            String phase,
            FailureRecorder failures) {
        if (!Double.isFinite(value)) {
            boolean optimizerStepSkipped = "train".equals(phase);
            String message = failures.nonFinite(phase, "loss", value, "loss", optimizerStepSkipped);
            discardGradientsIfNeeded(optimizerStepSkipped, failures);
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    private static void requirePresentBatchTensors(Batch batch, String phase, FailureRecorder failures) {
        boolean optimizerStepSkipped = "train".equals(phase);
        if (batch.inputs() == null) {
            failInvalidBatch(
                    phase,
                    "missing-input",
                    phase + " batch input tensor must not be null",
                    optimizerStepSkipped,
                    failures);
        }
        if (batch.labels() == null) {
            failInvalidBatch(
                    phase,
                    "missing-label",
                    phase + " batch label tensor must not be null",
                    optimizerStepSkipped,
                    failures);
        }
    }

    private static void requireValidBatchStructure(
            Batch batch,
            String phase,
            boolean optimizerStepSkipped,
            FailureRecorder failures) {
        GradTensor inputs = batch.inputs();
        GradTensor labels = batch.labels();
        long inputElements = inputs.numel();
        if (inputElements <= 0L) {
            failInvalidBatch(
                    phase,
                    "empty-input",
                    phase + " batch input tensor must contain at least one value, got shape "
                            + shapeString(inputs),
                    optimizerStepSkipped,
                    failures);
        }
        long labelElements = labels.numel();
        if (labelElements <= 0L) {
            failInvalidBatch(
                    phase,
                    "empty-label",
                    phase + " batch label tensor must contain at least one value, got shape "
                            + shapeString(labels),
                    optimizerStepSkipped,
                    failures);
        }
        long inputSamples = leadingDimension(inputs);
        long labelSamples = leadingDimension(labels);
        if (inputSamples != labelSamples) {
            failInvalidBatch(
                    phase,
                    "sample-count-mismatch",
                    phase + " batch sample count mismatch: inputs shape " + shapeString(inputs)
                            + " has " + inputSamples + " samples but labels shape "
                            + shapeString(labels) + " has " + labelSamples + " samples",
                    optimizerStepSkipped,
                    failures);
        }
    }

    private static long leadingDimension(GradTensor tensor) {
        long[] shape = tensor.shape();
        return shape.length == 0 ? tensor.numel() : shape[0];
    }

    private static void failInvalidBatch(
            String phase,
            String reason,
            String message,
            boolean optimizerStepSkipped,
            FailureRecorder failures) {
        String recordedMessage = failures.invalidBatch(phase, reason, message, optimizerStepSkipped);
        discardGradientsIfNeeded(optimizerStepSkipped, failures);
        throw new IllegalArgumentException(recordedMessage);
    }

    private static void discardGradientsIfNeeded(boolean optimizerStepSkipped, FailureRecorder failures) {
        if (optimizerStepSkipped) {
            failures.discardPendingGradients();
        }
    }
}
