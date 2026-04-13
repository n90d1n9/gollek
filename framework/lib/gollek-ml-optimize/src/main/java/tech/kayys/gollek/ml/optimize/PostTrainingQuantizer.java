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

package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Post-training INT8 quantization (Consolidated in gollek-ml-nn).
 */
public final class PostTrainingQuantizer {

    public record QuantizedTensor(byte[] data, float scale, float zeroPoint, long[] shape) {
        public int numel() {
            return data.length;
        }
    }

    public QuantizedTensor quantize(GradTensor t) {
        float[] data = t.data();
        float max = VectorOps.max(data);
        float min = -VectorOps.max(negate(data));

        float scale = (max - min) / 255f;
        if (scale == 0f)
            scale = 1e-8f;
        float zeroPoint = -min / scale;

        byte[] q = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            int v = Math.round(data[i] / scale + zeroPoint);
            q[i] = (byte) Math.max(0, Math.min(255, v));
        }
        return new QuantizedTensor(q, scale, zeroPoint, t.shape().clone());
    }

    public GradTensor dequantize(QuantizedTensor q) {
        float[] data = new float[q.data().length];
        for (int i = 0; i < data.length; i++) {
            data[i] = ((q.data()[i] & 0xFF) - q.zeroPoint()) * q.scale();
        }
        return GradTensor.of(data, q.shape());
    }

    public Map<String, QuantizedTensor> quantizeModel(Map<String, GradTensor> stateDict) {
        Map<String, QuantizedTensor> result = new LinkedHashMap<>();
        stateDict.forEach((name, tensor) -> result.put(name, quantize(tensor)));
        return result;
    }

    public float compressionRatio(Map<String, GradTensor> original,
            Map<String, QuantizedTensor> quantized) {
        return 4.0f;
    }

    private static float[] negate(float[] a) {
        float[] neg = new float[a.length];
        VectorOps.mulScalar(a, -1f, neg);
        return neg;
    }
}
