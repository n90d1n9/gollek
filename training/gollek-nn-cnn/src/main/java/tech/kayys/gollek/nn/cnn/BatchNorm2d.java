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
import java.util.Arrays;

/**
 * 2D Batch Normalization layer.
 */
public class BatchNorm2d extends NNModule {

    private final int numFeatures;
    private final float eps, momentum;
    private final boolean affine, trackRunningStats;

    private final Parameter weight, bias;
    private final float[] runningMean, runningVar;

    public BatchNorm2d(int numFeatures, float eps, float momentum, boolean affine, boolean trackRunningStats) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine) {
            this.weight = registerParameter("weight", GradTensor.ones(numFeatures));
            this.bias = registerParameter("bias", GradTensor.zeros(numFeatures));
        } else {
            this.weight = this.bias = null;
        }

        if (trackRunningStats) {
            this.runningMean = new float[numFeatures];
            this.runningVar = new float[numFeatures];
            Arrays.fill(this.runningVar, 1.0f);
        } else {
            this.runningMean = this.runningVar = null;
        }
    }

    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 1e-5f, 0.1f, true, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        NNBackendProvider backend = NNBackendRegistry.getDefault();
        
        GradTensor rm = trackRunningStats ? GradTensor.of(runningMean, numFeatures) : null;
        GradTensor rv = trackRunningStats ? GradTensor.of(runningVar, numFeatures) : null;
        
        GradTensor out = backend.batchNorm(input, 
                                           affine ? weight.data() : null, 
                                           affine ? bias.data() : null,
                                           rm, rv, isTraining(), momentum, eps);
        
        // Update local running stats from returned tensors if training
        if (isTraining() && trackRunningStats) {
            System.arraycopy(rm.data(), 0, runningMean, 0, numFeatures);
            System.arraycopy(rv.data(), 0, runningVar, 0, numFeatures);
        }
        
        return out;
    }

    @Override
    public String toString() {
        return String.format("BatchNorm2d(%d, eps=%.1e, momentum=%.2f, affine=%b)",
                numFeatures, eps, momentum, affine);
    }
}
