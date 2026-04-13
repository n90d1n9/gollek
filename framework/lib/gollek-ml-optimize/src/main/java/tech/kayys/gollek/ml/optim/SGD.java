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
