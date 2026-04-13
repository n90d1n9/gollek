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
            float[] gemm = tech.kayys.gollek.ml.tensor.VectorOps.matmul(wFlat, colN, Cout, colRows, colCols);
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
            float[] gemm = tech.kayys.gollek.ml.tensor.VectorOps.matmul(wFlat, colN, Cout, colRows, Lout);
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
        return null;
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
        float[] od = tech.kayys.gollek.ml.tensor.VectorOps.matmul(id, transpose(wd, outF, inF), N, inF, outF);

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
        return null;
    }

    @Override
    public GradTensor crop(GradTensor input, int top, int left, int height, int width) {
        return null;
    }

    @Override
    public GradTensor normalize(GradTensor input, float[] mean, float[] std) {
        return null;
    }

    @Override
    public GradTensor attention(GradTensor query, GradTensor key, GradTensor value, GradTensor mask, float scale) {
        return null;
    }

    @Override
    public GradTensor multiHeadAttention(GradTensor query, GradTensor key, GradTensor value, int numHeads, GradTensor mask) {
        return null;
    }

    @Override public void setKernelFusionEnabled(boolean enable) {}
    @Override public boolean isKernelFusionEnabled() { return false; }
    @Override public void setMemoryStrategy(String strategy) {}
    @Override public String getMemoryStrategy() { return "malloc"; }
    @Override public long getMemoryUsage() { return 0; }
    @Override public void clearMemory() {}
}
