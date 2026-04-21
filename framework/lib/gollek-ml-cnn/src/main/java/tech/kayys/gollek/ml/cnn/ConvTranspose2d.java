package tech.kayys.gollek.ml.cnn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.backend.NNBackendProvider;
import tech.kayys.gollek.ml.nn.backend.NNBackendRegistry;
import tech.kayys.gollek.ml.autograd.VectorOps;

/**
 * Transposed 2D Convolution (deconvolution) — upsamples spatial dimensions.
 *
 * <p>
 * Equivalent to {@code torch.nn.ConvTranspose2d}. Used in decoder networks,
 * GANs, and semantic segmentation (U-Net decoder path).
 *
 * <p>
 * Output size: {@code H_out = (H_in - 1) * stride - 2*padding + kernel}
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var deconv = new ConvTranspose2d(64, 32, 4, stride = 2, padding = 1); // 2× upsample
 * GradTensor out = deconv.forward(x); // [N,64,H,W] → [N,32,2H,2W]
 * }</pre>
 */
public class ConvTranspose2d extends NNModule {

    private final int inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW;
    private final Parameter weight; // [C_in, C_out, kH, kW]
    private final Parameter bias;

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0);
    }

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, true);
    }

    public ConvTranspose2d(int inC, int outC, int kH, int kW, int sH, int sW, int pH, int pW, boolean useBias) {
        this.inChannels = inC;
        this.outChannels = outC;
        this.kernelH = kH;
        this.kernelW = kW;
        this.strideH = sH;
        this.strideW = sW;
        this.padH = pH;
        this.padW = pW;

        float bound = (float) Math.sqrt(2.0 / (inC * kH * kW));
        this.weight = registerParameter("weight",
                GradTensor.of(randomUniform(inC * outC * kH * kW, -bound, bound), inC, outC, kH, kW));
        this.bias = useBias
                ? registerParameter("bias", GradTensor.of(new float[outC], outC))
                : null;
    }

    /**
     * Forward pass via col2im (transpose of im2col).
     *
     * @param x input {@code [N, C_in, H, W]}
     * @return output {@code [N, C_out, H_out, W_out]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        NNBackendProvider backend = NNBackendRegistry.getDefault();
        // Assuming outputPadding=0 for now as it wasn't in the original ConvTranspose2d
        return backend.convTranspose2d(x, weight.data(), bias != null ? bias.data() : null, strideH, padH, 0);
    }

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++)
            d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    public int getInChannels() {
        return inChannels;
    }

    public int getOutChannels() {
        return outChannels;
    }

    @Override
    public String toString() {
        return String.format("ConvTranspose2d(%d→%d, kernel=(%d,%d), stride=(%d,%d))",
                inChannels, outChannels, kernelH, kernelW, strideH, strideW);
    }
}
