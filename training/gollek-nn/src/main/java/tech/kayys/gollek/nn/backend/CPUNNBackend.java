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

package tech.kayys.gollek.ml.nn.backend;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;
import java.util.Locale;

/**
 * CPU-based neural network operations backend.
 */
public class CPUNNBackend implements NNBackendProvider {

    private static final Logger LOG = LoggerFactory.getLogger(CPUNNBackend.class);
    private static final String ID = "cpu";
    private static final int PRIORITY = 100;

    static {
        NNBackendRegistry.register(new CPUNNBackend());
    }

    @Override public String getBackendId() { return ID; }
    @Override public boolean isAvailable() { return true; }
    @Override public int getPriority() { return PRIORITY; }

    @Override
    public GradTensor conv2d(GradTensor input, GradTensor weight, GradTensor bias, int stride, int padding) {
        long[] s = input.shape();
        int N = (int) s[0], Cin = (int) s[1], H = (int) s[2], W = (int) s[3];
        long[] ws = weight.shape();
        int Cout = (int) ws[0], kH = (int) ws[2], kW = (int) ws[3];

        int Hout = (H + 2 * padding - kH) / stride + 1;
        int Wout = (W + 2 * padding - kW) / stride + 1;

        // im2col: [N, C_in*kH*kW, Hout*Wout]
        int colRows = Cin * kH * kW;
        int colCols = Hout * Wout;
        float[] col = im2col(input.data(), N, Cin, H, W, kH, kW, stride, padding, Hout, Wout, colRows, colCols);

        // GEMM: [Cout, colRows] x [colRows, colCols] = [Cout, colCols] per sample
        float[] wFlat = weight.data();
        float[] outData = new float[N * Cout * Hout * Wout];
        for (int n = 0; n < N; n++) {
            float[] colN = slice(col, n * colRows * colCols, colRows * colCols);
            float[] gemm = tech.kayys.gollek.ml.autograd.VectorOps.matmul(wFlat, colN, Cout, colRows, colCols);
            System.arraycopy(gemm, 0, outData, n * Cout * Hout * Wout, gemm.length);
        }

        // Add bias
        if (bias != null) {
            float[] bData = bias.data();
            for (int n = 0; n < N; n++)
                for (int c = 0; c < Cout; c++) {
                    int base = n * Cout * Hout * Wout + c * Hout * Wout;
                    for (int i = base; i < base + Hout * Wout; i++) outData[i] += bData[c];
                }
        }

        GradTensor result = GradTensor.of(outData, N, Cout, Hout, Wout);
        // Note: Backend doesn't handle autograd directly here, 
        // the layer will set the gradFn if needed using the backend's metadata if exposed.
        // For now, we'll let the layer handle autograd registration to stay consistent with Gollek's design.
        return result;
    }

    private float[] im2col(float[] input, int N, int Cin, int H, int W, int kH, int kW,
                           int stride, int padding, int Hout, int Wout, int colRows, int colCols) {
        float[] col = new float[N * colRows * colCols];
        for (int n = 0; n < N; n++) {
            int colBase = n * colRows * colCols;
            int row = 0;
            for (int c = 0; c < Cin; c++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++, row++) {
                        int colIdx = colBase + row * colCols;
                        for (int oh = 0; oh < Hout; oh++) {
                            int ih = oh * stride - padding + kh;
                            for (int ow = 0; ow < Wout; ow++) {
                                int iw = ow * stride - padding + kw;
                                float val = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        ? input[n * Cin * H * W + c * H * W + ih * W + iw]
                                        : 0f;
                                col[colIdx + oh * Wout + ow] = val;
                             }
                        }
                    }
                }
            }
        }
        return col;
    }

    private static float[] slice(float[] src, int offset, int len) {
        float[] out = new float[len];
        System.arraycopy(src, offset, out, 0, len);
        return out;
    }

    @Override
    public GradTensor conv1d(GradTensor input, GradTensor weight, GradTensor bias, int stride, int padding) {
        long[] s = input.shape();
        int N = (int) s[0], Cin = (int) s[1], L = (int) s[2];
        long[] ws = weight.shape();
        int Cout = (int) ws[0], k = (int) ws[2];

        int Lout = (L + 2 * padding - k) / stride + 1;

        // im2col for 1D: [N, Cin*k, Lout]
        int colRows = Cin * k;
        float[] col = im2col1d(input.data(), N, Cin, L, k, stride, padding, Lout, colRows);

        float[] wFlat = weight.data();
        float[] outData = new float[N * Cout * Lout];
        for (int n = 0; n < N; n++) {
            float[] colN = slice(col, n * colRows * Lout, colRows * Lout);
            float[] gemm = tech.kayys.gollek.ml.autograd.VectorOps.matmul(wFlat, colN, Cout, colRows, Lout);
            System.arraycopy(gemm, 0, outData, n * Cout * Lout, gemm.length);
        }

        if (bias != null) {
            float[] bData = bias.data();
            for (int n = 0; n < N; n++)
                for (int c = 0; c < Cout; c++) {
                    int base = n * Cout * Lout + c * Lout;
                    for (int i = base; i < base + Lout; i++) outData[i] += bData[c];
                }
        }
        return GradTensor.of(outData, N, Cout, Lout);
    }

    private float[] im2col1d(float[] data, int N, int Cin, int L, int kSize, int stride, int padding, int Lout, int colRows) {
        float[] col = new float[N * colRows * Lout];
        for (int n = 0; n < N; n++) {
            int colBase = n * colRows * Lout;
            int row = 0;
            for (int c = 0; c < Cin; c++) {
                for (int k = 0; k < kSize; k++, row++) {
                    for (int ol = 0; ol < Lout; ol++) {
                        int il = ol * stride - padding + k;
                        col[colBase + row * Lout + ol] =
                            (il >= 0 && il < L) ? data[n * Cin * L + c * L + il] : 0f;
                    }
                }
            }
        }
        return col;
    }

    @Override
    public GradTensor conv3d(GradTensor input, GradTensor weight, GradTensor bias, int stride, int padding) {
        long[] s = input.shape();
        int N = (int) s[0], Cin = (int) s[1], Din = (int) s[2], Hin = (int) s[3], Win = (int) s[4];
        long[] ws = weight.shape();
        int Cout = (int) ws[0], kD = (int) ws[2], kH = (int) ws[3], kW = (int) ws[4];

        int Dout = (Din + 2 * padding - kD) / stride + 1;
        int Hout = (Hin + 2 * padding - kH) / stride + 1;
        int Wout = (Win + 2 * padding - kW) / stride + 1;

        float[] xd = input.data(), wd = weight.data();
        float[] out = new float[N * Cout * Dout * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int co = 0; co < Cout; co++)
                for (int d = 0; d < Dout; d++)
                    for (int h = 0; h < Hout; h++)
                        for (int w = 0; w < Wout; w++) {
                            float sum = 0;
                            for (int ci = 0; ci < Cin; ci++)
                                for (int kd = 0; kd < kD; kd++)
                                    for (int kh = 0; kh < kH; kh++)
                                        for (int kw = 0; kw < kW; kw++) {
                                            int id = d * stride - padding + kd;
                                            int ih = h * stride - padding + kh;
                                            int iw = w * stride - padding + kw;
                                            if (id >= 0 && id < Din && ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                                                sum += xd[n * Cin * Din * Hin * Win + ci * Din * Hin * Win + id * Hin * Win + ih * Win + iw] *
                                                       wd[co * Cin * kD * kH * kW + ci * kD * kH * kW + kd * kH * kW + kh * kW + kw];
                                            }
                                        }
                            out[n * Cout * Dout * Hout * Wout + co * Dout * Hout * Wout + d * Hout * Wout + h * Wout + w] = sum;
                        }

        if (bias != null) {
            float[] bd = bias.data();
            for (int n = 0; n < N; n++)
                for (int co = 0; co < Cout; co++) {
                    int base = n * Cout * Dout * Hout * Wout + co * Dout * Hout * Wout;
                    for (int i = base; i < base + Dout * Hout * Wout; i++) out[i] += bd[co];
                }
        }
        return GradTensor.of(out, N, Cout, Dout, Hout, Wout);
    }

    @Override
    public GradTensor convTranspose2d(GradTensor input, GradTensor weight, GradTensor bias, int stride, int padding, int outputPadding) {
        long[] s = input.shape();
        int N = (int)s[0], Cin = (int)s[1], H = (int)s[2], W = (int)s[3];
        long[] ws = weight.shape();
        // weight shape for transpose: [Cin, Cout, kH, kW]
        int Cout = (int)ws[1], kH = (int)ws[2], kW = (int)ws[3];

        int Hout = (H - 1) * stride - 2 * padding + kH + outputPadding;
        int Wout = (W - 1) * stride - 2 * padding + kW + outputPadding;

        float[] xd = input.data(), wd = weight.data();
        float[] out = new float[N * Cout * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int ci = 0; ci < Cin; ci++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        float xVal = xd[n*Cin*H*W + ci*H*W + h*W + w];
                        for (int co = 0; co < Cout; co++)
                            for (int kh = 0; kh < kH; kh++)
                                for (int kw = 0; kw < kW; kw++) {
                                    int oh = h * stride - padding + kh;
                                    int ow = w * stride - padding + kw;
                                    if (oh < 0 || oh >= Hout || ow < 0 || ow >= Wout) continue;
                                    out[n*Cout*Hout*Wout + co*Hout*Wout + oh*Wout + ow]
                                        += xVal * wd[ci*Cout*kH*kW + co*kH*kW + kh*kW + kw];
                                }
                    }

        if (bias != null) {
            float[] bd = bias.data();
            for (int n = 0; n < N; n++)
                for (int co = 0; co < Cout; co++) {
                    int base = n*Cout*Hout*Wout + co*Hout*Wout;
                    for (int i = base; i < base + Hout*Wout; i++) out[i] += bd[co];
                }
        }
        return GradTensor.of(out, N, Cout, Hout, Wout);
    }

    @Override
    public GradTensor depthwiseConv2d(GradTensor input, GradTensor weight, GradTensor bias, int stride, int padding) {
        long[] inputShape = input.shape();
        long[] weightShape = weight.shape();
        requireRank("input", inputShape, 4);
        requireRank("weight", weightShape, 4);
        requirePositiveStrideAndPadding(stride, padding);

        int N = (int) inputShape[0], C = (int) inputShape[1], H = (int) inputShape[2], W = (int) inputShape[3];
        int weightChannels = (int) weightShape[0], channelMultiplier = (int) weightShape[1];
        int kH = (int) weightShape[2], kW = (int) weightShape[3];
        if (weightChannels != C) {
            throw new IllegalArgumentException(
                    "depthwise weight first dimension must match input channels, got "
                            + Arrays.toString(weightShape) + " for input " + Arrays.toString(inputShape));
        }
        if (channelMultiplier <= 0 || kH <= 0 || kW <= 0) {
            throw new IllegalArgumentException("depthwise weight must have positive multiplier/kernel dimensions, got "
                    + Arrays.toString(weightShape));
        }
        int Cout = C * channelMultiplier;
        if (bias != null && (bias.shape().length != 1 || bias.shape()[0] != Cout)) {
            throw new IllegalArgumentException(
                    "depthwise bias must be [channels * multiplier], got " + Arrays.toString(bias.shape()));
        }

        int Hout = outputSize(H, kH, stride, padding);
        int Wout = outputSize(W, kW, stride, padding);
        float[] inputData = input.data();
        float[] weightData = weight.data();
        float[] biasData = bias != null ? bias.data() : null;
        float[] output = new float[N * Cout * Hout * Wout];

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int m = 0; m < channelMultiplier; m++) {
                    int co = c * channelMultiplier + m;
                    for (int oh = 0; oh < Hout; oh++) {
                        for (int ow = 0; ow < Wout; ow++) {
                            float sum = 0.0f;
                            for (int kh = 0; kh < kH; kh++) {
                                int ih = oh * stride - padding + kh;
                                if (ih < 0 || ih >= H) {
                                    continue;
                                }
                                for (int kw = 0; kw < kW; kw++) {
                                    int iw = ow * stride - padding + kw;
                                    if (iw < 0 || iw >= W) {
                                        continue;
                                    }
                                    int inputIndex = n * C * H * W + c * H * W + ih * W + iw;
                                    int weightIndex = c * channelMultiplier * kH * kW + m * kH * kW + kh * kW + kw;
                                    sum += inputData[inputIndex] * weightData[weightIndex];
                                }
                            }
                            if (biasData != null) {
                                sum += biasData[co];
                            }
                            output[n * Cout * Hout * Wout + co * Hout * Wout + oh * Wout + ow] = sum;
                        }
                    }
                }
            }
        }
        return GradTensor.of(output, N, Cout, Hout, Wout);
    }

    @Override
    public GradTensor linear(GradTensor input, GradTensor weight, GradTensor bias) {
        // input: [N, inFeatures], weight: [outFeatures, inFeatures]
        long[] is = input.shape();
        long[] ws = weight.shape();
        int N = (int) is[0];
        int inF = (int) is[1];
        int outF = (int) ws[0];

        float[] id = input.data(), wd = weight.data();
        float[] od = tech.kayys.gollek.ml.autograd.VectorOps.matmul(id, transpose(wd, outF, inF), N, inF, outF);

        if (bias != null) {
            float[] bd = bias.data();
            for (int n = 0; n < N; n++)
                for (int f = 0; f < outF; f++) od[n * outF + f] += bd[f];
        }

        return GradTensor.of(od, N, outF);
    }

    private static float[] transpose(float[] m, int rows, int cols) {
        float[] t = new float[rows * cols];
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) t[c * rows + r] = m[r * cols + c];
        return t;
    }

    @Override
    public GradTensor maxPool2d(GradTensor input, int kernelSize, int stride, int padding) {
        long[] s = input.shape();
        int N = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
        int Hout = (H + 2 * padding - kernelSize) / stride + 1;
        int Wout = (W + 2 * padding - kernelSize) / stride + 1;

        float[] id = input.data();
        float[] od = new float[N * C * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < Hout; h++)
                    for (int w = 0; w < Wout; w++) {
                        float max = -Float.MAX_VALUE;
                        for (int kh = 0; kh < kernelSize; kh++)
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = h * stride - padding + kh;
                                int iw = w * stride - padding + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    max = Math.max(max, id[n * C * H * W + c * H * W + ih * W + iw]);
                                }
                            }
                        od[n * C * Hout * Wout + c * Hout * Wout + h * Wout + w] = max;
                    }
        return GradTensor.of(od, N, C, Hout, Wout);
    }

    @Override
    public GradTensor avgPool2d(GradTensor input, int kernelSize, int stride, int padding) {
        long[] s = input.shape();
        int N = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
        int Hout = (H + 2 * padding - kernelSize) / stride + 1;
        int Wout = (W + 2 * padding - kernelSize) / stride + 1;

        float[] id = input.data();
        float[] od = new float[N * C * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < Hout; h++)
                    for (int w = 0; w < Wout; w++) {
                        float sum = 0;
                        int count = 0;
                        for (int kh = 0; kh < kernelSize; kh++)
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = h * stride - padding + kh;
                                int iw = w * stride - padding + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += id[n * C * H * W + c * H * W + ih * W + iw];
                                    count++;
                                }
                            }
                        od[n * C * Hout * Wout + c * Hout * Wout + h * Wout + w] = sum / (kernelSize * kernelSize);
                    }
        return GradTensor.of(od, N, C, Hout, Wout);
    }

    @Override
    public GradTensor adaptiveAvgPool2d(GradTensor input, int[] outputSize) {
        long[] s = input.shape();
        int N = (int) s[0], C = (int) s[1], Hin = (int) s[2], Win = (int) s[3];
        int Hout = outputSize[0], Wout = outputSize[1];

        float[] id = input.data();
        float[] od = new float[N * C * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < Hout; oh++) {
                    int hStart = (int) Math.floor((double) oh * Hin / Hout);
                    int hEnd = (int) Math.ceil((double) (oh + 1) * Hin / Hout);
                    for (int ow = 0; ow < Wout; ow++) {
                        int wStart = (int) Math.floor((double) ow * Win / Wout);
                        int wEnd = (int) Math.ceil((double) (ow + 1) * Win / Wout);

                        float sum = 0;
                        int count = 0;
                        for (int ih = hStart; ih < hEnd; ih++)
                            for (int iw = wStart; iw < wEnd; iw++) {
                                sum += id[n * C * Hin * Win + c * Hin * Win + ih * Win + iw];
                                count++;
                            }
                        od[n * C * Hout * Wout + c * Hout * Wout + oh * Wout + ow] = sum / count;
                    }
                }
        return GradTensor.of(od, N, C, Hout, Wout);
    }

    @Override
    public GradTensor batchNorm(GradTensor input, GradTensor weight, GradTensor bias,
                                 GradTensor runningMean, GradTensor runningVar,
                                 boolean training, float momentum, float eps) {
        long[] s = input.shape();
        int N = (int) s[0], C = (int) s[1];
        int HW = (int) (input.numel() / (N * C));

        float[] id = input.data();
        float[] od = new float[id.length];
        float[] wd = weight != null ? weight.data() : null;
        float[] bd = bias != null ? bias.data() : null;

        float[] mean = new float[C];
        float[] var = new float[C];

        if (training) {
            for (int c = 0; c < C; c++) {
                float sum = 0;
                for (int n = 0; n < N; n++) {
                    for (int i = 0; i < HW; i++) sum += id[n * C * HW + c * HW + i];
                }
                mean[c] = sum / (N * HW);

                float varSum = 0;
                for (int n = 0; n < N; n++) {
                    for (int i = 0; i < HW; i++) {
                        float diff = id[n * C * HW + c * HW + i] - mean[c];
                        varSum += diff * diff;
                    }
                }
                var[c] = varSum / (N * HW);

                if (runningMean != null) {
                    float[] rm = runningMean.data();
                    float[] rv = runningVar.data();
                    rm[c] = (1 - momentum) * rm[c] + momentum * mean[c];
                    rv[c] = (1 - momentum) * rv[c] + momentum * var[c];
                }
            }
        } else {
            System.arraycopy(runningMean.data(), 0, mean, 0, C);
            System.arraycopy(runningVar.data(), 0, var, 0, C);
        }

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++) {
                float invStd = (float) (1.0 / Math.sqrt(var[c] + eps));
                float w = wd != null ? wd[c] : 1.0f;
                float b = bd != null ? bd[c] : 0.0f;
                for (int i = 0; i < HW; i++) {
                    int idx = n * C * HW + c * HW + i;
                    od[idx] = (id[idx] - mean[c]) * invStd * w + b;
                }
            }
        return GradTensor.of(od, s);
    }

    @Override
    public GradTensor layerNorm(GradTensor input, int[] normalizedShape, GradTensor weight, GradTensor bias, float eps) {
        long[] s = input.shape();
        float[] id = input.data();
        float[] od = new float[id.length];
        float[] wd = weight != null ? weight.data() : null;
        float[] bd = bias != null ? bias.data() : null;

        int normSize = 1;
        for (int dim : normalizedShape) normSize *= dim;
        int numNorms = (int) (input.numel() / normSize);

        for (int i = 0; i < numNorms; i++) {
            int base = i * normSize;
            float sum = 0;
            for (int j = 0; j < normSize; j++) sum += id[base + j];
            float mean = sum / normSize;

            float varSum = 0;
            for (int j = 0; j < normSize; j++) {
                float diff = id[base + j] - mean;
                varSum += diff * diff;
            }
            float var = varSum / normSize;
            float invStd = (float) (1.0 / Math.sqrt(var + eps));

            for (int j = 0; j < normSize; j++) {
                float w = wd != null ? wd[j] : 1.0f;
                float b = bd != null ? bd[j] : 0.0f;
                od[base + j] = (id[base + j] - mean) * invStd * w + b;
            }
        }
        return GradTensor.of(od, s);
    }

    @Override
    public GradTensor resize(GradTensor input, int height, int width, String mode) {
        long[] shape = input.shape();
        requireRank("input", shape, 4);
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("resize height and width must be positive, got " + height + "x" + width);
        }

        int N = (int) shape[0], C = (int) shape[1], Hin = (int) shape[2], Win = (int) shape[3];
        float[] inputData = input.data();
        float[] output = new float[N * C * height * width];
        String normalizedMode = mode == null ? "bilinear" : mode.toLowerCase(Locale.ROOT);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < height; oh++) {
                    for (int ow = 0; ow < width; ow++) {
                        float value = switch (normalizedMode) {
                            case "nearest" -> sampleNearest(inputData, N, C, Hin, Win, n, c, oh, ow, height, width);
                            case "bilinear", "linear" ->
                                    sampleBilinear(inputData, N, C, Hin, Win, n, c, oh, ow, height, width);
                            case "bicubic", "cubic" ->
                                    sampleBicubic(inputData, N, C, Hin, Win, n, c, oh, ow, height, width);
                            default -> throw new IllegalArgumentException(
                                    "unsupported resize mode '" + mode + "'; expected nearest, bilinear, or bicubic");
                        };
                        output[n * C * height * width + c * height * width + oh * width + ow] = value;
                    }
                }
            }
        }
        return GradTensor.of(output, N, C, height, width);
    }

    @Override
    public GradTensor crop(GradTensor input, int top, int left, int height, int width) {
        long[] shape = input.shape();
        requireRank("input", shape, 4);
        int N = (int) shape[0], C = (int) shape[1], H = (int) shape[2], W = (int) shape[3];
        if (top < 0 || left < 0 || height <= 0 || width <= 0 || top + height > H || left + width > W) {
            throw new IllegalArgumentException(
                    "crop must fit inside input [N,C,H,W]=" + Arrays.toString(shape)
                            + ", got top=" + top + ", left=" + left + ", height=" + height + ", width=" + width);
        }

        float[] inputData = input.data();
        float[] output = new float[N * C * height * width];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < height; h++) {
                    int inputBase = n * C * H * W + c * H * W + (top + h) * W + left;
                    int outputBase = n * C * height * width + c * height * width + h * width;
                    System.arraycopy(inputData, inputBase, output, outputBase, width);
                }
            }
        }
        return GradTensor.of(output, N, C, height, width);
    }

    @Override
    public GradTensor normalize(GradTensor input, float[] mean, float[] std) {
        long[] shape = input.shape();
        requireRank("input", shape, 4);
        int N = (int) shape[0], C = (int) shape[1], H = (int) shape[2], W = (int) shape[3];
        if (mean == null || std == null || mean.length != C || std.length != C) {
            throw new IllegalArgumentException(
                    "mean and std must be non-null arrays with one value per channel; channels=" + C);
        }
        for (int c = 0; c < C; c++) {
            if (!Float.isFinite(mean[c]) || !Float.isFinite(std[c]) || std[c] == 0.0f) {
                throw new IllegalArgumentException(
                        "mean/std must be finite and std must be non-zero at channel " + c);
            }
        }

        float[] inputData = input.data();
        float[] output = new float[inputData.length];
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float channelMean = mean[c];
                float channelStd = std[c];
                int base = n * C * H * W + c * H * W;
                for (int i = 0; i < H * W; i++) {
                    output[base + i] = (inputData[base + i] - channelMean) / channelStd;
                }
            }
        }
        return GradTensor.of(output, shape);
    }

    @Override
    public GradTensor attention(GradTensor query, GradTensor key, GradTensor value, GradTensor mask, float scale) {
        long[] queryShape = query.shape();
        long[] keyShape = key.shape();
        long[] valueShape = value.shape();
        requireAttentionShapes(queryShape, keyShape, valueShape);
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("attention scale must be finite, got: " + scale);
        }

        int rank = queryShape.length;
        int L = (int) queryShape[rank - 2];
        int D = (int) queryShape[rank - 1];
        int S = (int) keyShape[rank - 2];
        int Dv = (int) valueShape[rank - 1];
        int groups = leadingProduct(queryShape, rank - 2);
        long[] outputShape = queryShape.clone();
        outputShape[rank - 1] = Dv;

        float[] queryData = query.data();
        float[] keyData = key.data();
        float[] valueData = value.data();
        float[] maskData = mask != null ? mask.data() : null;
        long[] maskShape = mask != null ? mask.shape() : null;
        float[] output = new float[groups * L * Dv];
        float[] scores = new float[S];

        for (int group = 0; group < groups; group++) {
            int queryBase = group * L * D;
            int keyBase = group * S * D;
            int valueBase = group * S * Dv;
            int outputBase = group * L * Dv;
            for (int l = 0; l < L; l++) {
                float maxScore = -Float.MAX_VALUE;
                for (int s = 0; s < S; s++) {
                    float score = 0.0f;
                    for (int d = 0; d < D; d++) {
                        score += queryData[queryBase + l * D + d] * keyData[keyBase + s * D + d];
                    }
                    score *= scale;
                    if (maskData != null && !maskAllows(maskData, maskShape, queryShape, group, groups, l, s, S)) {
                        score = -1.0e9f;
                    }
                    scores[s] = score;
                    maxScore = Math.max(maxScore, score);
                }

                float sumExp = 0.0f;
                for (int s = 0; s < S; s++) {
                    scores[s] = (float) Math.exp(scores[s] - maxScore);
                    sumExp += scores[s];
                }

                for (int dv = 0; dv < Dv; dv++) {
                    float weighted = 0.0f;
                    for (int s = 0; s < S; s++) {
                        weighted += (scores[s] / sumExp) * valueData[valueBase + s * Dv + dv];
                    }
                    output[outputBase + l * Dv + dv] = weighted;
                }
            }
        }
        return GradTensor.of(output, outputShape);
    }

    @Override
    public GradTensor multiHeadAttention(GradTensor query, GradTensor key, GradTensor value, int numHeads, GradTensor mask) {
        long[] queryShape = query.shape();
        long[] keyShape = key.shape();
        long[] valueShape = value.shape();
        requireRank("query", queryShape, 3);
        requireRank("key", keyShape, 3);
        requireRank("value", valueShape, 3);
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive, got: " + numHeads);
        }
        if (queryShape[0] != keyShape[0] || queryShape[0] != valueShape[0]) {
            throw new IllegalArgumentException("query, key, and value batch sizes must match");
        }
        if (queryShape[2] != keyShape[2] || keyShape[2] != valueShape[2]) {
            throw new IllegalArgumentException("query, key, and value embedding dimensions must match");
        }
        int batch = (int) queryShape[0];
        int L = (int) queryShape[1];
        int S = (int) keyShape[1];
        int embedDim = (int) queryShape[2];
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedding dimension " + embedDim
                    + " must be divisible by numHeads " + numHeads);
        }
        int headDim = embedDim / numHeads;

        float[] qHeads = splitHeads(query.data(), batch, L, embedDim, numHeads, headDim);
        float[] kHeads = splitHeads(key.data(), batch, S, embedDim, numHeads, headDim);
        float[] vHeads = splitHeads(value.data(), batch, S, embedDim, numHeads, headDim);

        GradTensor attended = attention(
                GradTensor.of(qHeads, batch, numHeads, L, headDim),
                GradTensor.of(kHeads, batch, numHeads, S, headDim),
                GradTensor.of(vHeads, batch, numHeads, S, headDim),
                mask,
                (float) (1.0 / Math.sqrt(headDim)));
        float[] merged = mergeHeads(attended.data(), batch, L, embedDim, numHeads, headDim);
        return GradTensor.of(merged, batch, L, embedDim);
    }

    private static void requireRank(String name, long[] shape, int rank) {
        if (shape.length != rank) {
            throw new IllegalArgumentException(name + " must be rank " + rank + ", got " + Arrays.toString(shape));
        }
    }

    private static void requirePositiveStrideAndPadding(int stride, int padding) {
        if (stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("stride must be positive and padding must be non-negative");
        }
    }

    private static int outputSize(int input, int kernel, int stride, int padding) {
        int output = (input + 2 * padding - kernel) / stride + 1;
        if (output <= 0) {
            throw new IllegalArgumentException(
                    "computed output size must be positive; input=" + input
                            + ", kernel=" + kernel + ", stride=" + stride + ", padding=" + padding);
        }
        return output;
    }

    private static float sampleNearest(float[] input, int N, int C, int H, int W,
                                       int n, int c, int oh, int ow, int outH, int outW) {
        int ih = clamp((int) Math.floor((oh + 0.5f) * H / outH), 0, H - 1);
        int iw = clamp((int) Math.floor((ow + 0.5f) * W / outW), 0, W - 1);
        return input[n * C * H * W + c * H * W + ih * W + iw];
    }

    private static float sampleBilinear(float[] input, int N, int C, int H, int W,
                                        int n, int c, int oh, int ow, int outH, int outW) {
        float y = sourceCoordinate(oh, H, outH);
        float x = sourceCoordinate(ow, W, outW);
        int y0 = clamp((int) Math.floor(y), 0, H - 1);
        int x0 = clamp((int) Math.floor(x), 0, W - 1);
        int y1 = clamp(y0 + 1, 0, H - 1);
        int x1 = clamp(x0 + 1, 0, W - 1);
        float wy = y - (float) Math.floor(y);
        float wx = x - (float) Math.floor(x);

        float v00 = pixel(input, C, H, W, n, c, y0, x0);
        float v01 = pixel(input, C, H, W, n, c, y0, x1);
        float v10 = pixel(input, C, H, W, n, c, y1, x0);
        float v11 = pixel(input, C, H, W, n, c, y1, x1);
        float top = v00 * (1.0f - wx) + v01 * wx;
        float bottom = v10 * (1.0f - wx) + v11 * wx;
        return top * (1.0f - wy) + bottom * wy;
    }

    private static float sampleBicubic(float[] input, int N, int C, int H, int W,
                                       int n, int c, int oh, int ow, int outH, int outW) {
        float y = sourceCoordinate(oh, H, outH);
        float x = sourceCoordinate(ow, W, outW);
        int yBase = (int) Math.floor(y);
        int xBase = (int) Math.floor(x);
        float sum = 0.0f;
        for (int dy = -1; dy <= 2; dy++) {
            int iy = clamp(yBase + dy, 0, H - 1);
            float wy = cubicWeight(y - (yBase + dy));
            for (int dx = -1; dx <= 2; dx++) {
                int ix = clamp(xBase + dx, 0, W - 1);
                float wx = cubicWeight(x - (xBase + dx));
                sum += pixel(input, C, H, W, n, c, iy, ix) * wy * wx;
            }
        }
        return sum;
    }

    private static float sourceCoordinate(int outIndex, int inSize, int outSize) {
        return Math.max(0.0f, (outIndex + 0.5f) * inSize / outSize - 0.5f);
    }

    private static float cubicWeight(float distance) {
        float x = Math.abs(distance);
        float a = -0.75f;
        if (x <= 1.0f) {
            return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
        }
        return 0.0f;
    }

    private static float pixel(float[] input, int C, int H, int W, int n, int c, int h, int w) {
        return input[n * C * H * W + c * H * W + h * W + w];
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void requireAttentionShapes(long[] queryShape, long[] keyShape, long[] valueShape) {
        if (queryShape.length < 3 || keyShape.length != queryShape.length || valueShape.length != queryShape.length) {
            throw new IllegalArgumentException(
                    "query, key, and value must have the same rank >= 3, got "
                            + Arrays.toString(queryShape) + ", "
                            + Arrays.toString(keyShape) + ", "
                            + Arrays.toString(valueShape));
        }
        int rank = queryShape.length;
        for (int i = 0; i < rank - 2; i++) {
            if (queryShape[i] != keyShape[i] || queryShape[i] != valueShape[i]) {
                throw new IllegalArgumentException("attention leading dimensions must match");
            }
        }
        if (queryShape[rank - 1] != keyShape[rank - 1]) {
            throw new IllegalArgumentException("query and key feature dimensions must match");
        }
        if (keyShape[rank - 2] != valueShape[rank - 2]) {
            throw new IllegalArgumentException("key and value sequence lengths must match");
        }
        if (queryShape[rank - 2] <= 0 || keyShape[rank - 2] <= 0
                || queryShape[rank - 1] <= 0 || valueShape[rank - 1] <= 0) {
            throw new IllegalArgumentException("attention sequence and feature dimensions must be positive");
        }
    }

    private static int leadingProduct(long[] shape, int endExclusive) {
        long product = 1L;
        for (int i = 0; i < endExclusive; i++) {
            product *= shape[i];
        }
        if (product > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor is too large for CPU attention backend");
        }
        return (int) product;
    }

    private static boolean maskAllows(float[] maskData, long[] maskShape, long[] queryShape,
                                      int group, int groups, int queryIndex, int keyIndex, int expectedKeyLength) {
        int rank = queryShape.length;
        int L = (int) queryShape[rank - 2];
        int S = (int) maskShape[maskShape.length - 1];
        if (maskShape.length == 2) {
            requireMaskTail(maskShape, L, expectedKeyLength);
            return maskData[queryIndex * S + keyIndex] > 0.0f;
        }
        if (maskShape.length == 3 && maskShape[0] == groups) {
            requireMaskTail(maskShape, L, expectedKeyLength);
            return maskData[group * L * S + queryIndex * S + keyIndex] > 0.0f;
        }
        if (rank == 4 && maskShape.length == 3 && maskShape[0] == queryShape[0]) {
            requireMaskTail(maskShape, L, expectedKeyLength);
            int heads = (int) queryShape[1];
            int batch = group / heads;
            return maskData[batch * L * S + queryIndex * S + keyIndex] > 0.0f;
        }
        if (maskShape.length == rank) {
            for (int i = 0; i < rank - 2; i++) {
                if (maskShape[i] != queryShape[i]) {
                    throw new IllegalArgumentException(
                            "attention mask leading dimensions must match or be broadcastable, got "
                                    + Arrays.toString(maskShape));
                }
            }
            requireMaskTail(maskShape, L, expectedKeyLength);
            return maskData[group * L * S + queryIndex * S + keyIndex] > 0.0f;
        }
        throw new IllegalArgumentException(
                "unsupported attention mask shape " + Arrays.toString(maskShape)
                        + " for query shape " + Arrays.toString(queryShape));
    }

    private static void requireMaskTail(long[] maskShape, int expectedQueryLength, int expectedKeyLength) {
        if (maskShape[maskShape.length - 2] != expectedQueryLength
                || maskShape[maskShape.length - 1] != expectedKeyLength) {
            throw new IllegalArgumentException("attention mask query/key lengths do not match");
        }
    }

    private static float[] splitHeads(float[] input, int batch, int sequence, int embedDim, int numHeads, int headDim) {
        float[] output = new float[batch * numHeads * sequence * headDim];
        for (int n = 0; n < batch; n++) {
            for (int seq = 0; seq < sequence; seq++) {
                for (int head = 0; head < numHeads; head++) {
                    for (int d = 0; d < headDim; d++) {
                        int inputIndex = n * sequence * embedDim + seq * embedDim + head * headDim + d;
                        int outputIndex = n * numHeads * sequence * headDim
                                + head * sequence * headDim + seq * headDim + d;
                        output[outputIndex] = input[inputIndex];
                    }
                }
            }
        }
        return output;
    }

    private static float[] mergeHeads(float[] input, int batch, int sequence, int embedDim, int numHeads, int headDim) {
        float[] output = new float[batch * sequence * embedDim];
        for (int n = 0; n < batch; n++) {
            for (int seq = 0; seq < sequence; seq++) {
                for (int head = 0; head < numHeads; head++) {
                    for (int d = 0; d < headDim; d++) {
                        int inputIndex = n * numHeads * sequence * headDim
                                + head * sequence * headDim + seq * headDim + d;
                        int outputIndex = n * sequence * embedDim + seq * embedDim + head * headDim + d;
                        output[outputIndex] = input[inputIndex];
                    }
                }
            }
        }
        return output;
    }

    @Override public void setKernelFusionEnabled(boolean enable) {}
    @Override public boolean isKernelFusionEnabled() { return false; }
    @Override public void setMemoryStrategy(String strategy) {}
    @Override public String getMemoryStrategy() { return "malloc"; }
    @Override public long getMemoryUsage() { return 0; }
    @Override public void clearMemory() {}
}
