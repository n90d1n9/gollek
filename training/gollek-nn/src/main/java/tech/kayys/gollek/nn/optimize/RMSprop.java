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
 * RMSprop Optimizer (Root Mean Square Propagation).
 */
public class RMSprop implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float alpha;
    private final float epsilon;
    private final float weightDecay;
    private final float momentum;

    private final Map<Parameter, float[]> squareAvg = new HashMap<>();
    private final Map<Parameter, float[]> velocity = new HashMap<>();

    private RMSprop(Builder builder) {
        this.parameters = builder.parameters;
        this.learningRate = builder.lr;
        this.alpha = builder.alpha;
        this.epsilon = builder.eps;
        this.weightDecay = builder.weightDecay;
        this.momentum = builder.momentum;

        for (Parameter param : parameters) {
            int size = (int) param.data().numel();
            squareAvg.put(param, new float[size]);
            if (momentum > 0) {
                velocity.put(param, new float[size]);
            }
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
            float[] avg = squareAvg.get(param);

            for (int i = 0; i < data.length; i++) {
                float g = grad[i];
                if (weightDecay != 0) g += weightDecay * data[i];

                avg[i] = alpha * avg[i] + (1 - alpha) * g * g;
                float denom = (float) Math.sqrt(avg[i] + epsilon);

                if (momentum > 0) {
                    float[] v = velocity.get(param);
                    v[i] = momentum * v[i] + g / denom;
                    data[i] -= learningRate * v[i];
                } else {
                    data[i] -= learningRate * g / denom;
                }
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
        state.put("optimizer", "RMSprop");
        state.put("learningRate", learningRate);
        state.put("alpha", alpha);
        state.put("epsilon", epsilon);
        state.put("weightDecay", weightDecay);
        state.put("momentum", momentum);
        state.put("squareAvg", exportSlot(squareAvg));
        if (momentum > 0) {
            state.put("velocity", exportSlot(velocity));
        }
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"RMSprop".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected RMSprop but got " + name);
        }

        requireFloatMatch(state.get("alpha"), alpha, "alpha");
        requireFloatMatch(state.get("epsilon"), epsilon, "epsilon");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        requireFloatMatch(state.get("momentum"), momentum, "momentum");
        this.learningRate = readFloat(state.get("learningRate"), learningRate);

        Object squareAvgState = state.get("squareAvg");
        if (squareAvgState == null) {
            throw new IllegalArgumentException("Invalid RMSprop checkpoint payload: missing key 'squareAvg'");
        }
        restoreSlot(squareAvg, squareAvgState, "squareAvg");

        if (momentum > 0) {
            Object velocityState = state.get("velocity");
            if (velocityState == null) {
                throw new IllegalArgumentException(
                        "Invalid RMSprop checkpoint payload: missing key 'velocity' for momentum optimizer");
            }
            restoreSlot(velocity, velocityState, "velocity");
        }
    }

    private List<float[]> exportSlot(Map<Parameter, float[]> slotMap) {
        List<float[]> exported = new ArrayList<>(parameters.size());
        for (Parameter param : parameters) {
            float[] source = slotMap.get(param);
            if (source == null) {
                exported.add(new float[(int) param.data().numel()]);
            } else {
                exported.add(source.clone());
            }
        }
        return exported;
    }

    private void restoreSlot(Map<Parameter, float[]> slotMap, Object rawState, String slotName) {
        if (!(rawState instanceof List<?> serializedSlot)) {
            throw new IllegalArgumentException(
                    "Invalid RMSprop checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid RMSprop checkpoint payload: " + slotName + " size mismatch (expected "
                            + parameters.size() + ", got " + serializedSlot.size() + ")");
        }

        for (int i = 0; i < parameters.size(); i++) {
            float[] source = coerceFloatArray(serializedSlot.get(i), slotName, i);
            Parameter param = parameters.get(i);
            float[] target = slotMap.get(param);
            if (target == null) {
                target = new float[(int) param.data().numel()];
                slotMap.put(param, target);
            }
            if (source.length != target.length) {
                throw new IllegalArgumentException(
                        "Invalid RMSprop checkpoint payload: " + slotName + "[" + i
                                + "] length mismatch (expected " + target.length
                                + ", got " + source.length + ")");
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
                    "Invalid RMSprop checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static float[] coerceFloatArray(Object value, String slotName, int index) {
        if (value instanceof float[] array) {
            return array;
        }
        if (value instanceof List<?> numbers) {
            float[] converted = new float[numbers.size()];
            for (int i = 0; i < numbers.size(); i++) {
                Object element = numbers.get(i);
                if (!(element instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "Invalid RMSprop checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid RMSprop checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }

    public static class Builder {
        private final List<Parameter> parameters;
        private final float lr;
        private float alpha = 0.99f;
        private float eps = 1e-8f;
        private float weightDecay = 0.0f;
        private float momentum = 0.0f;

        private Builder(List<Parameter> parameters, float lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        public Builder alpha(float a) { this.alpha = a; return this; }
        public Builder eps(float e) { this.eps = e; return this; }
        public Builder weightDecay(float wd) { this.weightDecay = wd; return this; }
        public Builder momentum(float m) { this.momentum = m; return this; }

        public RMSprop build() { return new RMSprop(this); }
    }
}
