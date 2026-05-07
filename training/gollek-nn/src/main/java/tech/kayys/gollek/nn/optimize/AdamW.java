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

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * AdamW Optimizer (Adam with decoupled weight decay).
 *
 * <p>Standardized implementation with Builder pattern and AMSGrad support.</p>
 */
public class AdamW implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final boolean amsgrad;

    private final Map<Parameter, float[]> m = new HashMap<>(); // first moment
    private final Map<Parameter, float[]> v = new HashMap<>(); // second moment
    private final Map<Parameter, float[]> maxV = new HashMap<>(); // amsgrad

    private int step = 0;

    private AdamW(Builder builder) {
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

            // Decoupled weight decay
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    data[i] *= (1.0f - learningRate * weightDecay);
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
    public void setLearningRate(float lr) { this.learningRate = lr; }

    @Override
    public List<Parameter> parameters() { return parameters; }

    @Override
    public String toString() {
        return String.format("AdamW(lr=%.4f, betas=(%.2f, %.3f), weightDecay=%.4f, amsgrad=%b)",
                learningRate, beta1, beta2, weightDecay, amsgrad);
    }

    public static class Builder {
        private final List<Parameter> parameters;
        private final float lr;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float eps = 1e-8f;
        private float weightDecay = 0.01f;
        private boolean amsgrad = false;

        private Builder(List<Parameter> parameters, float lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        public Builder betas(float b1, float b2) { this.beta1 = b1; this.beta2 = b2; return this; }
        public Builder eps(float e) { this.eps = e; return this; }
        public Builder weightDecay(float wd) { this.weightDecay = wd; return this; }
        public Builder amsgrad(boolean enabled) { this.amsgrad = enabled; return this; }

        public AdamW build() { return new AdamW(this); }
    }
}
