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
