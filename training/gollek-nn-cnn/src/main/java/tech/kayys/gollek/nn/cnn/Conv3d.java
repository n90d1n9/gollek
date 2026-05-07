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

package tech.kayys.gollek.ml.cnn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.backend.NNBackendRegistry;
import tech.kayys.gollek.ml.nn.backend.NNBackendProvider;

/**
 * 3D Convolutional Layer.
 */
public class Conv3d extends NNModule {

    private final int inChannels, outChannels, kernelSize, stride, padding;
    private final Parameter weight;
    private final Parameter bias;

    public Conv3d(int inChannels, int outChannels, int kernelSize, int stride, int padding, boolean useBias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;

        float limit = (float) Math.sqrt(6.0 / (inChannels * kernelSize * kernelSize * kernelSize + outChannels));
        this.weight = registerParameter("weight", 
            GradTensor.of(randomUniform(outChannels * inChannels * kernelSize * kernelSize * kernelSize, -limit, limit),
                          outChannels, inChannels, kernelSize, kernelSize, kernelSize));

        this.bias = useBias ? registerParameter("bias", GradTensor.zeros(outChannels)) : null;
    }

    public Conv3d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        NNBackendProvider backend = NNBackendRegistry.getDefault();
        return backend.conv3d(input, weight.data(), bias != null ? bias.data() : null, stride, padding);
    }

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n]; java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    @Override
    public String toString() {
        return String.format("Conv3d(%d, %d, kernel=%d, stride=%d, padding=%d)",
                inChannels, outChannels, kernelSize, stride, padding);
    }
}
