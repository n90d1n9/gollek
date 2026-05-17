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
 * Adam Optimizer (Adaptive Moment Estimation).
 *
 * <p>Improved version with Builder pattern, weight decay, and AMSGrad support.</p>
 *
 * @author Gollek Team
 */
public class Adam implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final boolean amsgrad;
    
    // Per-parameter state
    private final Map<Parameter, float[]> m = new HashMap<>(); // first moment
    private final Map<Parameter, float[]> v = new HashMap<>(); // second moment
    private final Map<Parameter, float[]> maxV = new HashMap<>(); // amsgrad
    
    private int step = 0;

    private Adam(Builder builder) {
        this.parameters = builder.parameters;
        this.learningRate = builder.lr;
        this.beta1 = builder.beta1;
        this.beta2 = builder.beta2;
        this.epsilon = builder.eps;
        this.weightDecay = builder.weightDecay;
        this.amsgrad = builder.amsgrad;

        for (Parameter param : parameters) {
            int size = (int) param.data().numel();
            m.put(param, new float[size]);
            v.put(param, new float[size]);
            if (amsgrad) {
                maxV.put(param, new float[size]);
            }
        }
    }

    public static Builder builder(List<Parameter> parameters, float lr) {
        return new Builder(parameters, lr);
    }

    @Override
    public void step() {
        step++;
        float bc1 = (float) (1.0 - Math.pow(beta1, step));
        float bc2 = (float) (1.0 - Math.pow(beta2, step));

        for (Parameter param : parameters) {
            if (param.grad() == null) continue;

            float[] data = param.data().data();
            float[] grad = param.grad().data();
            float[] m_t = m.get(param);
            float[] v_t = v.get(param);

            // Traditional weight decay (L2 penalty)
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    grad[i] += weightDecay * data[i];
                }
            }

            for (int i = 0; i < data.length; i++) {
                m_t[i] = beta1 * m_t[i] + (1 - beta1) * grad[i];
                v_t[i] = beta2 * v_t[i] + (1 - beta2) * grad[i] * grad[i];

                float mHat = m_t[i] / bc1;
                float vHat = v_t[i] / bc2;

                if (amsgrad) {
                    float[] mv = maxV.get(param);
                    mv[i] = Math.max(mv[i], vHat);
                    data[i] -= learningRate * mHat / ((float) Math.sqrt(mv[i]) + epsilon);
                } else {
                    data[i] -= learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
                }
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (Parameter param : parameters) param.zeroGrad();
    }

    @Override
    public float learningRate() { return learningRate; }

    @Override
    public void setLearningRate(float lr) {
        if (lr < 0) throw new IllegalArgumentException("LR must be non-negative");
        this.learningRate = lr;
    }

    @Override
    public List<Parameter> parameters() { return parameters; }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("optimizer", "Adam");
        state.put("step", step);
        state.put("learningRate", learningRate);
        state.put("beta1", beta1);
        state.put("beta2", beta2);
        state.put("epsilon", epsilon);
        state.put("weightDecay", weightDecay);
        state.put("amsgrad", amsgrad);
        state.put("m", exportSlot(m));
        state.put("v", exportSlot(v));
        if (amsgrad) {
            state.put("maxV", exportSlot(maxV));
        }
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"Adam".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected Adam but got " + name);
        }

        requireFloatMatch(state.get("beta1"), beta1, "beta1");
        requireFloatMatch(state.get("beta2"), beta2, "beta2");
        requireFloatMatch(state.get("epsilon"), epsilon, "epsilon");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        requireBooleanMatch(state.get("amsgrad"), amsgrad, "amsgrad");
        this.step = Math.max(0, readInt(state.get("step"), step));
        this.learningRate = readFloat(state.get("learningRate"), learningRate);

        Object mState = state.get("m");
        if (mState == null) {
            throw new IllegalArgumentException("Invalid Adam checkpoint payload: missing key 'm'");
        }
        restoreSlot(m, mState, "m");

        Object vState = state.get("v");
        if (vState == null) {
            throw new IllegalArgumentException("Invalid Adam checkpoint payload: missing key 'v'");
        }
        restoreSlot(v, vState, "v");

        if (amsgrad) {
            Object maxVState = state.get("maxV");
            if (maxVState == null) {
                throw new IllegalArgumentException(
                        "Invalid Adam checkpoint payload: missing key 'maxV' for AMSGrad optimizer");
            }
            restoreSlot(maxV, maxVState, "maxV");
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
                    "Invalid Adam checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid Adam checkpoint payload: " + slotName + " size mismatch (expected "
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
                        "Invalid Adam checkpoint payload: " + slotName + "[" + i
                                + "] length mismatch (expected " + target.length
                                + ", got " + source.length + ")");
            }
            System.arraycopy(source, 0, target, 0, source.length);
        }
    }

    private static int readInt(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
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
                    "Invalid Adam checkpoint payload: " + fieldName
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
                    "Invalid Adam checkpoint payload: " + fieldName + " must be boolean");
        }
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid Adam checkpoint payload: " + fieldName
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
                            "Invalid Adam checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid Adam checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }

    /**
     * Builder for Adam optimizer.
     */
    public static class Builder {
        private final List<Parameter> parameters;
        private final float lr;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float eps = 1e-8f;
        private float weightDecay = 0.0f;
        private boolean amsgrad = false;

        private Builder(List<Parameter> parameters, float lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        public Builder betas(float b1, float b2) { this.beta1 = b1; this.beta2 = b2; return this; }
        public Builder eps(float e) { this.eps = e; return this; }
        public Builder weightDecay(float wd) { this.weightDecay = wd; return this; }
        public Builder amsgrad(boolean enabled) { this.amsgrad = enabled; return this; }

        public Adam build() { return new Adam(this); }
    }
}
