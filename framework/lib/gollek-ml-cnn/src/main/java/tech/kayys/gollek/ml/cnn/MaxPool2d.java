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
import tech.kayys.gollek.ml.nn.backend.NNBackendRegistry;
import tech.kayys.gollek.ml.nn.backend.NNBackendProvider;

/**
 * 2D Max Pooling layer.
 */
public class MaxPool2d extends NNModule {

    private final int kernelSize, stride, padding;

    public MaxPool2d(int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    public MaxPool2d(int kernelSize, int stride) {
        this(kernelSize, stride, 0);
    }

    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        NNBackendProvider backend = NNBackendRegistry.getDefault();
        return backend.maxPool2d(input, kernelSize, stride, padding);
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel=%d, stride=%d, padding=%d)", kernelSize, stride, padding);
    }
}
