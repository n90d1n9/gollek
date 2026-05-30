package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.List;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

/**
 * Tensor norm/count diagnostics used by the canonical trainer.
 */
final class TrainerTensorDiagnostics {
    private TrainerTensorDiagnostics() {
    }

    static TensorDiagnostics gradients(List<Parameter> parameters) {
        List<GradTensor> gradients = new ArrayList<>();
        for (Parameter parameter : parameters) {
            GradTensor gradient = parameter.grad();
            if (gradient != null) {
                gradients.add(gradient);
            }
        }
        return tensors(gradients);
    }

    static TensorDiagnostics parameters(List<Parameter> parameters) {
        List<GradTensor> tensors = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            tensors.add(parameter.data());
        }
        return tensors(tensors);
    }

    static List<float[]> snapshotParameters(List<Parameter> parameters) {
        List<float[]> snapshot = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            snapshot.add(parameter.data().data().clone());
        }
        return snapshot;
    }

    static TensorDiagnostics parameterUpdates(List<Parameter> parameters, List<float[]> beforeStep) {
        if (beforeStep == null || beforeStep.size() != parameters.size()) {
            throw new IllegalArgumentException("parameter snapshot must match parameter count");
        }
        double sumSquares = 0.0;
        double sumAbs = 0.0;
        double maxAbs = 0.0;
        int tensorCount = 0;
        long valueCount = 0;
        long zeroCount = 0;
        long nanCount = 0;
        long positiveInfinityCount = 0;
        long negativeInfinityCount = 0;
        for (int parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++) {
            float[] before = beforeStep.get(parameterIndex);
            float[] after = parameters.get(parameterIndex).data().data();
            if (before.length != after.length) {
                throw new IllegalStateException("parameter snapshot shape mismatch for parameter " + parameterIndex);
            }
            tensorCount++;
            valueCount += after.length;
            for (int i = 0; i < after.length; i++) {
                float delta = after[i] - before[i];
                if (delta == 0.0f) {
                    zeroCount++;
                }
                double absolute = Math.abs(delta);
                if (Float.isNaN(delta)) {
                    nanCount++;
                } else if (delta == Float.POSITIVE_INFINITY) {
                    positiveInfinityCount++;
                } else if (delta == Float.NEGATIVE_INFINITY) {
                    negativeInfinityCount++;
                }
                sumAbs += absolute;
                sumSquares += absolute * absolute;
                maxAbs = Math.max(maxAbs, absolute);
            }
        }
        return new TensorDiagnostics(
                tensorCount,
                valueCount,
                zeroCount,
                sumAbs,
                Math.sqrt(sumSquares),
                maxAbs,
                nanCount,
                positiveInfinityCount,
                negativeInfinityCount);
    }

    static void scaleGradients(List<Parameter> parameters, float scale) {
        if (scale == 1.0f) {
            return;
        }
        for (Parameter parameter : parameters) {
            GradTensor gradient = parameter.grad();
            if (gradient == null) {
                continue;
            }
            float[] values = gradient.data();
            for (int i = 0; i < values.length; i++) {
                values[i] *= scale;
            }
        }
    }

    static void requireFinite(
            TensorDiagnostics diagnostics,
            String phase,
            String kind,
            boolean optimizerStepSkipped,
            TrainerBatchGuards.FailureRecorder failures) {
        if (Double.isFinite(diagnostics.l2Norm()) && Double.isFinite(diagnostics.maxAbs())) {
            return;
        }
        double value = diagnostics.representativeNonFiniteValue();
        throw new IllegalArgumentException(failures.nonFiniteTensor(
                phase,
                kind,
                value,
                kind,
                optimizerStepSkipped,
                diagnostics.valueCount(),
                diagnostics.nanCount(),
                diagnostics.positiveInfinityCount(),
                diagnostics.negativeInfinityCount()));
    }

    static TensorDiagnostics tensors(Iterable<GradTensor> tensors) {
        double sumSquares = 0.0;
        double sumAbs = 0.0;
        double maxAbs = 0.0;
        int tensorCount = 0;
        long valueCount = 0;
        long zeroCount = 0;
        long nanCount = 0;
        long positiveInfinityCount = 0;
        long negativeInfinityCount = 0;
        for (GradTensor tensor : tensors) {
            if (tensor == null) {
                continue;
            }
            tensorCount++;
            float[] values = tensor.data();
            valueCount += values.length;
            for (float value : values) {
                if (value == 0.0f) {
                    zeroCount++;
                }
                double absolute = Math.abs(value);
                if (Float.isNaN(value)) {
                    nanCount++;
                } else if (value == Float.POSITIVE_INFINITY) {
                    positiveInfinityCount++;
                } else if (value == Float.NEGATIVE_INFINITY) {
                    negativeInfinityCount++;
                }
                sumAbs += absolute;
                sumSquares += absolute * absolute;
                maxAbs = Math.max(maxAbs, absolute);
            }
        }
        return new TensorDiagnostics(
                tensorCount,
                valueCount,
                zeroCount,
                sumAbs,
                Math.sqrt(sumSquares),
                maxAbs,
                nanCount,
                positiveInfinityCount,
                negativeInfinityCount);
    }
}

record TensorDiagnostics(
        int tensorCount,
        long valueCount,
        long zeroCount,
        double sumAbs,
        double l2Norm,
        double maxAbs,
        long nanCount,
        long positiveInfinityCount,
        long negativeInfinityCount) {
    static TensorDiagnostics empty() {
        return new TensorDiagnostics(0, 0L, 0L, 0.0, 0.0, 0.0, 0L, 0L, 0L);
    }

    long finiteCount() {
        return Math.max(0L, valueCount - nonFiniteCount());
    }

    long nonFiniteCount() {
        return nanCount + positiveInfinityCount + negativeInfinityCount;
    }

    long nonZeroCount() {
        return valueCount - zeroCount;
    }

    double meanAbs() {
        if (valueCount == 0L) {
            return 0.0;
        }
        return sumAbs / valueCount;
    }

    double rms() {
        if (valueCount == 0L) {
            return 0.0;
        }
        return l2Norm / Math.sqrt(valueCount);
    }

    double zeroFraction() {
        if (valueCount == 0L) {
            return 0.0;
        }
        return zeroCount / (double) valueCount;
    }

    double nonFiniteFraction() {
        if (valueCount == 0L) {
            return 0.0;
        }
        return nonFiniteCount() / (double) valueCount;
    }

    double representativeNonFiniteValue() {
        if (nanCount > 0L) {
            return Double.NaN;
        }
        if (positiveInfinityCount > 0L) {
            return Double.POSITIVE_INFINITY;
        }
        if (negativeInfinityCount > 0L) {
            return Double.NEGATIVE_INFINITY;
        }
        if (!Double.isFinite(l2Norm)) {
            return l2Norm;
        }
        return maxAbs;
    }
}
