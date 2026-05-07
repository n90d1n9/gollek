package tech.kayys.gollek.ml.cnn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * 2D Upsampling layer using Bilinear Interpolation.
 * <p>
 * Scales the spatial dimensions (H, W) of the input tensor by a scale factor.
 */
public class Upsample2d extends NNModule {

    private final float scaleFactor;

    public Upsample2d(float scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        if (shape.length != 4) {
            throw new IllegalArgumentException("Upsample2d expects 4D input [N, C, H, W]");
        }
        int N = (int) shape[0];
        int C = (int) shape[1];
        int H = (int) shape[2];
        int W = (int) shape[3];

        int newH = (int) (H * scaleFactor);
        int newW = (int) (W * scaleFactor);

        float[] inData = input.data();
        float[] outData = new float[N * C * newH * newW];

        // Bilinear interpolation loop
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int baseIn = (n * C + c) * H * W;
                int baseOut = (n * C + c) * newH * newW;

                for (int oh = 0; oh < newH; oh++) {
                    float h = (oh + 0.5f) / scaleFactor - 0.5f;
                    int h0 = (int) Math.floor(h);
                    int h1 = Math.min(H - 1, h0 + 1);
                    float hW = h - h0;
                    h0 = Math.max(0, h0);

                    for (int ow = 0; ow < newW; ow++) {
                        float w = (ow + 0.5f) / scaleFactor - 0.5f;
                        int w0 = (int) Math.floor(w);
                        int w1 = Math.min(W - 1, w0 + 1);
                        float wW = w - w0;
                        w0 = Math.max(0, w0);

                        float v00 = inData[baseIn + h0 * W + w0];
                        float v01 = inData[baseIn + h0 * W + w1];
                        float v10 = inData[baseIn + h1 * W + w0];
                        float v11 = inData[baseIn + h1 * W + w1];

                        float val = v00 * (1 - hW) * (1 - wW) +
                                    v01 * (1 - hW) * wW +
                                    v10 * hW * (1 - wW) +
                                    v11 * hW * wW;

                        outData[baseOut + oh * newW + ow] = val;
                    }
                }
            }
        }

        return GradTensor.of(outData, N, C, newH, newW);
    }

    @Override
    public String toString() {
        return String.format("Upsample2d(scale=%.2f)", scaleFactor);
    }
}
