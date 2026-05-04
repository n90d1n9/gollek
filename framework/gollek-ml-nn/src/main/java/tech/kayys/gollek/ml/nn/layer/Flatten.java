package tech.kayys.gollek.ml.nn.layer;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * Flattens a tensor to 1D.
 * <p>
 * Equivalent to {@code torch.nn.Flatten}.
 * Commonly used between convolutional and linear layers.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var flatten = new Flatten();
 * var output = flatten.forward(input);  // [N, C, H, W] → [N, C*H*W]
 * }</pre>
 */
public class Flatten extends NNModule {

    private final int startDim;
    private final int endDim;

    public Flatten() {
        this(1, -1);
    }

    public Flatten(int startDim, int endDim) {
        this.startDim = startDim;
        this.endDim = endDim;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] s = input.shape();
        int ndim = s.length;
        int start = startDim < 0 ? ndim + startDim : startDim;
        int end = endDim < 0 ? ndim + endDim : endDim;

        if (start < 0 || end >= ndim || start > end) {
            throw new IllegalArgumentException("Invalid flatten dimensions: [" + start + ", " + end + "] for " + ndim + "D tensor");
        }

        long[] outShape = new long[start + 1 + (ndim - 1 - end)];
        long flattened = 1;
        for (int i = 0; i < start; i++) outShape[i] = s[i];
        for (int i = start; i <= end; i++) flattened *= s[i];
        outShape[start] = flattened;
        for (int i = end + 1, j = start + 1; i < ndim; i++, j++) outShape[j] = s[i];

        return input.reshape(outShape);
    }

    @Override
    public String toString() {
        return "Flatten(startDim=" + startDim + ", endDim=" + endDim + ")";
    }
}
