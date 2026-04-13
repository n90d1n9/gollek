package tech.kayys.gollek.ml.nn.layer;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;

/**
 * Group Normalization layer.
 * <p>
 * GroupNorm divides the channels into groups and computes within each group
 * the mean and variance for normalization.
 * <p>
 * Input: [N, C, H, W]
 * Output: Same as input.
 */
public class GroupNorm extends NNModule {

    private final int numGroups;
    private final int numChannels;
    private final Parameter weight;
    private final Parameter bias;
    private final float eps;

    public GroupNorm(int numGroups, int numChannels) {
        this(numGroups, numChannels, 1e-5f);
    }

    public GroupNorm(int numGroups, int numChannels, float eps) {
        if (numChannels % numGroups != 0) {
            throw new IllegalArgumentException("numChannels must be divisible by numGroups");
        }
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;

        this.weight = registerParameter("weight", GradTensor.ones(numChannels));
        this.bias = registerParameter("bias", GradTensor.zeros(numChannels));
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        if (shape.length != 4) {
            throw new IllegalArgumentException("GroupNorm expects 4D input [N, C, H, W]");
        }
        int N = (int) shape[0];
        int C = (int) shape[1];
        int H = (int) shape[2];
        int W = (int) shape[3];

        // Reshape to [N, numGroups, C // numGroups, H * W]
        int channelsPerGroup = C / numGroups;
        GradTensor x = input.reshape(N, numGroups, channelsPerGroup * H * W);

        // Compute mean and variance along the last dimension
        GradTensor mean = x.mean(-1).unsqueeze(-1);
        GradTensor var = x.sub(mean).pow(2f).mean(-1).unsqueeze(-1);

        // Normalize
        GradTensor normalized = x.sub(mean).div(var.add(eps).sqrt());

        // Reshape back and apply affine transform
        GradTensor out = normalized.reshape(N, C, H, W);
        
        // Broadcast weight and bias: [1, C, 1, 1]
        GradTensor w = weight.data().reshape(1, C, 1, 1);
        GradTensor b = bias.data().reshape(1, C, 1, 1);

        return out.mul(w).add(b);
    }

    @Override
    public String toString() {
        return String.format("GroupNorm(groups=%d, channels=%d)", numGroups, numChannels);
    }
}
