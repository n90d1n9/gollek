/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * Stochastic Gradient Descent optimizer with optional momentum.
 *
 * <p>Standardized implementation with Builder pattern.</p>
 */
public class SGD implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float momentum;
    private final float weightDecay;
    private final boolean nesterov;

    private final Map<Parameter, float[]> velocity = new HashMap<>();

    private SGD(Builder builder) {
        this.parameters = builder.parameters;
        this.learningRate = builder.lr;
        this.momentum = builder.momentum;
        this.weightDecay = builder.weightDecay;
        this.nesterov = builder.nesterov;

        for (Parameter param : parameters) {
            velocity.put(param, new float[(int) param.data().numel()]);
        }
    }

    public static Builder builder(List<Parameter> parameters, float lr) {
        return new Builder(parameters, lr);
    }

    @Override
    public void step() {
        for (Parameter param : parameters) {
            if (param.grad() == null) continue;

            float[] data = param.data().data();
            float[] grad = param.grad().data();
            float[] v = velocity.get(param);

            for (int i = 0; i < data.length; i++) {
                float g = grad[i];
                if (weightDecay != 0) {
                    g += weightDecay * data[i];
                }

                if (momentum != 0) {
                    v[i] = momentum * v[i] + g;
                    if (nesterov) {
                        g = g + momentum * v[i];
                    } else {
                        g = v[i];
                    }
                }

                data[i] -= learningRate * g;
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (Parameter param : parameters) param.zeroGrad();
    }

    @Override public float learningRate() { return learningRate; }
    @Override public void setLearningRate(float lr) { this.learningRate = lr; }
    @Override public List<Parameter> parameters() { return parameters; }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("optimizer", "SGD");
        state.put("learningRate", learningRate);
        state.put("momentum", momentum);
        state.put("weightDecay", weightDecay);
        state.put("nesterov", nesterov);
        state.put("velocity", exportVelocity());
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"SGD".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected SGD but got " + name);
        }

        requireFloatMatch(state.get("momentum"), momentum, "momentum");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        requireBooleanMatch(state.get("nesterov"), nesterov, "nesterov");
        this.learningRate = readFloat(state.get("learningRate"), learningRate);
        Object velocityState = state.get("velocity");
        if (velocityState != null) {
            restoreVelocity(velocityState);
        }
    }

    private List<float[]> exportVelocity() {
        List<float[]> exported = new ArrayList<>(parameters.size());
        for (Parameter param : parameters) {
            float[] source = velocity.get(param);
            if (source == null) {
                exported.add(new float[(int) param.data().numel()]);
            } else {
                exported.add(source.clone());
            }
        }
        return exported;
    }

    private void restoreVelocity(Object rawVelocityState) {
        if (!(rawVelocityState instanceof List<?> serializedVelocity)) {
            throw new IllegalArgumentException("Invalid SGD checkpoint payload: velocity must be a List");
        }
        if (serializedVelocity.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid SGD checkpoint payload: velocity size mismatch (expected "
                            + parameters.size() + ", got " + serializedVelocity.size() + ")");
        }

        for (int i = 0; i < parameters.size(); i++) {
            float[] source = coerceFloatArray(serializedVelocity.get(i), "velocity", i);
            Parameter param = parameters.get(i);
            float[] target = velocity.get(param);
            if (target == null) {
                target = new float[(int) param.data().numel()];
                velocity.put(param, target);
            }
            if (source.length != target.length) {
                throw new IllegalArgumentException(
                        "Invalid SGD checkpoint payload: velocity[" + i + "] length mismatch (expected "
                                + target.length + ", got " + source.length + ")");
            }
            System.arraycopy(source, 0, target, 0, source.length);
        }
    }

    private static float readFloat(Object value, float fallback) {
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof String text) {
            try {
                return Float.parseFloat(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static void requireFloatMatch(Object value, float expected, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected);
        if (Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid SGD checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static void requireBooleanMatch(Object value, boolean expected, String fieldName) {
        if (value == null) {
            return;
        }
        boolean loaded;
        if (value instanceof Boolean bool) {
            loaded = bool;
        } else if (value instanceof String text) {
            loaded = Boolean.parseBoolean(text);
        } else {
            throw new IllegalArgumentException(
                    "Invalid SGD checkpoint payload: " + fieldName + " must be boolean");
        }
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid SGD checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static float[] coerceFloatArray(Object value, String field, int index) {
        if (value instanceof float[] array) {
            return array;
        }
        if (value instanceof List<?> numbers) {
            float[] converted = new float[numbers.size()];
            for (int i = 0; i < numbers.size(); i++) {
                Object element = numbers.get(i);
                if (!(element instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "Invalid SGD checkpoint payload: " + field + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid SGD checkpoint payload: " + field + "[" + index + "] must be float[]");
    }

    @Override
    public String toString() {
        return String.format("SGD(lr=%.4f, momentum=%.2f, weightDecay=%.4f, nesterov=%b)",
                learningRate, momentum, weightDecay, nesterov);
    }

    public static class Builder {
        private final List<Parameter> parameters;
        private final float lr;
        private float momentum = 0.0f;
        private float weightDecay = 0.0f;
        private boolean nesterov = false;

        private Builder(List<Parameter> parameters, float lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        public Builder momentum(float m) { this.momentum = m; return this; }
        public Builder weightDecay(float wd) { this.weightDecay = wd; return this; }
        public Builder nesterov(boolean enabled) { this.nesterov = enabled; return this; }

        public SGD build() { return new SGD(this); }
    }
}
