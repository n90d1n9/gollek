package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Group Normalization — normalizes over groups of channels, independent of
 * batch size.
 *
 * <p>
 * Equivalent to {@code torch.nn.GroupNorm}. Unlike BatchNorm, GroupNorm works
 * well with small batch sizes (even batch=1), making it preferred for
 * detection,
 * segmentation, and video models.
 *
 * <p>
 * Formula:
 * 
 * <pre>
 *   Split C channels into G groups of C/G channels each
 *   For each group: normalize over [C/G, H, W] dimensions
 *   y = γ·x̂ + β
 * </pre>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var gn = new GroupNorm(numGroups = 32, numChannels = 256);
 * GradTensor out = gn.forward(x); // [N, 256, H, W] → [N, 256, H, W]
 * }</pre>
 */
public class GroupNorm extends NNModule {

    private final int numGroups;
    private final int numChannels;
    private final float eps;
    private final Parameter gamma; // [C]
    private final Parameter beta; // [C]

    /**
     * Creates a GroupNorm layer.
     *
     * @param numGroups   number of groups to divide channels into (must divide
     *                    numChannels)
     * @param numChannels total number of channels C
     * @throws IllegalArgumentException if numChannels is not divisible by numGroups
     */
    public GroupNorm(int numGroups, int numChannels) {
        this(numGroups, numChannels, 1e-5f);
    }

    /**
     * Creates a GroupNorm layer with custom epsilon.
     *
     * @param numGroups   number of groups
     * @param numChannels total channels
     * @param eps         numerical stability constant
     */
    public GroupNorm(int numGroups, int numChannels, float eps) {
        if (numGroups <= 0) {
            throw new IllegalArgumentException("numGroups must be positive");
        }
        if (numChannels <= 0) {
            throw new IllegalArgumentException("numChannels must be positive");
        }
        if (numChannels % numGroups != 0)
            throw new IllegalArgumentException("numChannels must be divisible by numGroups");
        if (!Float.isFinite(eps) || eps <= 0f) {
            throw new IllegalArgumentException("eps must be finite and positive");
        }
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;
        float[] ones = new float[numChannels];
        java.util.Arrays.fill(ones, 1f);
        this.gamma = registerParameter("weight", GradTensor.of(ones, numChannels));
        this.beta = registerParameter("bias", GradTensor.of(new float[numChannels], numChannels));
    }

    /**
     * Forward pass.
     *
     * @param x input tensor {@code [N, C, H, W]}
     * @return normalized tensor {@code [N, C, H, W]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        if (s.length != 4) {
            throw new IllegalArgumentException("GroupNorm expects 4D input [N, C, H, W], got rank " + s.length);
        }
        int N = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
        if (C != numChannels) {
            throw new IllegalArgumentException("input channel count must be " + numChannels + ", got: " + C);
        }

        int cPerG = C / numGroups;
        GradTensor grouped = x.reshape(N, numGroups, cPerG * H * W);
        GradTensor mean = grouped.mean(-1).unsqueeze(-1);
        GradTensor centered = grouped.sub(mean);
        GradTensor variance = centered.pow(2f).mean(-1).unsqueeze(-1);
        GradTensor normalized = centered.div(variance.add(eps).sqrt()).reshape(N, C, H, W);

        GradTensor scale = gamma.data().reshape(1, C, 1, 1);
        GradTensor shift = beta.data().reshape(1, C, 1, 1);
        return normalized.mul(scale).add(shift);
    }

    @Override
    public String toString() {
        return "GroupNorm(groups=" + numGroups + ", channels=" + numChannels + ")";
    }
}
