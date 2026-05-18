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
        double value = Double.isFinite(diagnostics.l2Norm()) ? diagnostics.maxAbs() : diagnostics.l2Norm();
        throw new IllegalArgumentException(failures.nonFinite(
                phase,
                kind,
                value,
                kind,
                optimizerStepSkipped));
    }

    static TensorDiagnostics tensors(Iterable<GradTensor> tensors) {
        double sumSquares = 0.0;
        double maxAbs = 0.0;
        int tensorCount = 0;
        long valueCount = 0;
        for (GradTensor tensor : tensors) {
            if (tensor == null) {
                continue;
            }
            tensorCount++;
            float[] values = tensor.data();
            valueCount += values.length;
            for (float value : values) {
                double absolute = Math.abs(value);
                sumSquares += absolute * absolute;
                maxAbs = Math.max(maxAbs, absolute);
            }
        }
        return new TensorDiagnostics(tensorCount, valueCount, Math.sqrt(sumSquares), maxAbs);
    }
}

record TensorDiagnostics(int tensorCount, long valueCount, double l2Norm, double maxAbs) {
}
