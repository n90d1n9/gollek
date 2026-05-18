package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;
import java.util.Objects;

final class OptimizerValidation {

    private OptimizerValidation() {
    }

    static List<Parameter> requireParameters(List<Parameter> parameters) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        for (int i = 0; i < parameters.size(); i++) {
            Parameter parameter = parameters.get(i);
            if (parameter == null) {
                throw new IllegalArgumentException("parameters[" + i + "] must not be null");
            }
            if (parameter.data() == null) {
                throw new IllegalArgumentException("parameters[" + i + "] data must not be null");
            }
        }
        return parameters;
    }

    static float learningRate(float value) {
        return finiteNonNegative(value, "learningRate");
    }

    static float weightDecay(float value) {
        return finiteNonNegative(value, "weightDecay");
    }

    static float momentum(float value) {
        return finiteNonNegative(value, "momentum");
    }

    static float epsilon(float value) {
        return finitePositive(value, "epsilon");
    }

    static float beta(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f || value >= 1.0f) {
            throw new IllegalArgumentException(name + " must be finite and in [0, 1), got: " + value);
        }
        return value;
    }

    static float finitePositive(float value, String name) {
        if (!Float.isFinite(value) || value <= 0.0f) {
            throw new IllegalArgumentException(name + " must be finite and positive, got: " + value);
        }
        return value;
    }

    static float finiteNonNegative(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f) {
            throw new IllegalArgumentException(name + " must be finite and non-negative, got: " + value);
        }
        return value;
    }

    static void requireStepInputs(List<Parameter> parameters, String optimizerName) {
        requireParameters(parameters);
        for (int parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++) {
            Parameter parameter = parameters.get(parameterIndex);
            GradTensor grad = parameter.grad();
            if (grad == null) {
                continue;
            }
            float[] data = parameter.data().data();
            float[] gradData = grad.data();
            if (gradData.length != data.length) {
                throw new IllegalStateException(
                        optimizerName + " gradient shape mismatch for parameter " + parameterIndex
                                + " (expected " + data.length + " values, got " + gradData.length + ")");
            }
            requireFiniteValues(data, optimizerName + " parameter " + parameterIndex);
            requireFiniteValues(gradData, optimizerName + " gradient " + parameterIndex);
        }
    }

    static void requireFiniteValues(float[] values, String name) {
        Objects.requireNonNull(values, name + " values must not be null");
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                throw new IllegalStateException(name + " must contain only finite values, got "
                        + values[i] + " at index " + i);
            }
        }
    }
}
