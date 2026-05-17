package tech.kayys.gollek.ml.autograd;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Array-backed compatibility tensor for the ML/trainer Gradle migration.
 *
 * <p>This is intentionally CPU-only. It restores the legacy
 * {@code tech.kayys.gollek.ml.autograd.GradTensor} surface so the training-side
 * modules can build coherently in-repo while deeper runtime work continues.</p>
 */
public final class GradTensor {

    private static final Random RNG = new Random();

    private final float[] data;
    private final long[] shape;
    private boolean requiresGrad;
    private GradTensor grad;
    private Function gradFn;

    private GradTensor(float[] data, long[] shape) {
        this.data = Objects.requireNonNull(data, "data");
        this.shape = normalizeShape(shape);
        long expected = numel(this.shape);
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " does not match shape " + Arrays.toString(this.shape));
        }
    }

    public static GradTensor of(float[] data, long... shape) {
        return new GradTensor(data.clone(), shape.clone());
    }

    public static GradTensor of(float... data) {
        return new GradTensor(data.clone(), new long[]{data.length});
    }

    public static GradTensor zeros(long... shape) {
        return new GradTensor(new float[Math.toIntExact(numel(shape))], shape.clone());
    }

    public static GradTensor ones(long... shape) {
        float[] values = new float[Math.toIntExact(numel(shape))];
        Arrays.fill(values, 1f);
        return new GradTensor(values, shape.clone());
    }

    public static GradTensor full(float value, long... shape) {
        float[] values = new float[Math.toIntExact(numel(shape))];
        Arrays.fill(values, value);
        return new GradTensor(values, shape.clone());
    }

    public static GradTensor scalar(float value) {
        return new GradTensor(new float[]{value}, new long[0]);
    }

    public static GradTensor rand(long... shape) {
        float[] values = new float[Math.toIntExact(numel(shape))];
        for (int i = 0; i < values.length; i++) {
            values[i] = RNG.nextFloat();
        }
        return new GradTensor(values, shape.clone());
    }

    public static GradTensor randn(long... shape) {
        float[] values = new float[Math.toIntExact(numel(shape))];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) RNG.nextGaussian();
        }
        return new GradTensor(values, shape.clone());
    }

    public static GradTensor uniform(double lo, double hi, long... shape) {
        float[] values = new float[Math.toIntExact(numel(shape))];
        double range = hi - lo;
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (lo + RNG.nextDouble() * range);
        }
        return new GradTensor(values, shape.clone());
    }

    public static GradTensor arange(float start, float end, float step) {
        if (step == 0f) {
            throw new IllegalArgumentException("step must not be zero");
        }
        int size = Math.max(0, (int) Math.ceil((end - start) / step));
        float[] values = new float[size];
        float current = start;
        for (int i = 0; i < size; i++) {
            values[i] = current;
            current += step;
        }
        return new GradTensor(values, new long[]{size});
    }

    public static GradTensor eye(int n) {
        float[] values = new float[n * n];
        for (int i = 0; i < n; i++) {
            values[i * n + i] = 1f;
        }
        return new GradTensor(values, new long[]{n, n});
    }

    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        long[] outShape = broadcastShape(broadcastShape(condition.shape, x.shape), y.shape);
        float[] out = new float[Math.toIntExact(numel(outShape))];
        for (int flat = 0; flat < out.length; flat++) {
            long[] idx = unravel(flat, outShape);
            float c = condition.data[offsetFor(idx, condition.shape, outShape)];
            float xv = x.data[offsetFor(idx, x.shape, outShape)];
            float yv = y.data[offsetFor(idx, y.shape, outShape)];
            out[flat] = c != 0f ? xv : yv;
        }
        GradTensor result = new GradTensor(out, outShape).requiresGrad(x.requiresGrad || y.requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Where") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (x.requiresGrad()) {
                        float[] gradX = new float[x.data.length];
                        for (int flat = 0; flat < upstreamData.length; flat++) {
                            long[] idx = unravel(flat, outShape);
                            float c = condition.data[offsetFor(idx, condition.shape, outShape)];
                            if (c != 0f) {
                                gradX[offsetFor(idx, x.shape, outShape)] += upstreamData[flat];
                            }
                        }
                        x.backward(GradTensor.of(gradX, x.shape()));
                    }
                    if (y.requiresGrad()) {
                        float[] gradY = new float[y.data.length];
                        for (int flat = 0; flat < upstreamData.length; flat++) {
                            long[] idx = unravel(flat, outShape);
                            float c = condition.data[offsetFor(idx, condition.shape, outShape)];
                            if (c == 0f) {
                                gradY[offsetFor(idx, y.shape, outShape)] += upstreamData[flat];
                            }
                        }
                        y.backward(GradTensor.of(gradY, y.shape()));
                    }
                }
            });
        }
        return result;
    }

    public static GradTensor cat(GradTensor... tensors) {
        return cat(0, tensors);
    }

    public static GradTensor cat(int dim, GradTensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("tensors must not be empty");
        }
        long[] base = tensors[0].shape();
        int rank = base.length;
        if (rank == 0) {
            float[] out = new float[tensors.length];
            for (int i = 0; i < tensors.length; i++) {
                out[i] = tensors[i].item();
            }
            GradTensor result = new GradTensor(out, new long[]{tensors.length}).requiresGrad(anyRequiresGrad(tensors));
            if (result.requiresGrad() && !NoGrad.active()) {
                result.setGradFn(new Function.Context("CatScalar") {
                    @Override
                    public void backward(GradTensor upstream) {
                        float[] upstreamData = upstream.data();
                        for (int i = 0; i < tensors.length; i++) {
                            if (tensors[i].requiresGrad()) {
                                tensors[i].backward(GradTensor.of(new float[] {upstreamData[i]}, tensors[i].shape()));
                            }
                        }
                    }
                });
            }
            return result;
        }

        int resolvedDim = normalizeDim(dim, rank);
        long totalDim = 0;
        for (GradTensor tensor : tensors) {
            long[] currentShape = tensor.shape();
            if (currentShape.length != rank) {
                throw new IllegalArgumentException("all tensors must have same rank");
            }
            for (int i = 0; i < rank; i++) {
                if (i == resolvedDim) {
                    continue;
                }
                if (currentShape[i] != base[i]) {
                    throw new IllegalArgumentException("cat shape mismatch at dim " + i);
                }
            }
            totalDim += currentShape[resolvedDim];
        }

        long[] outShape = base.clone();
        outShape[resolvedDim] = totalDim;
        float[] out = new float[Math.toIntExact(numel(outShape))];

        int innerBlock = 1;
        for (int i = resolvedDim + 1; i < rank; i++) {
            innerBlock *= Math.toIntExact(base[i]);
        }
        int outerBlock = 1;
        for (int i = 0; i < resolvedDim; i++) {
            outerBlock *= Math.toIntExact(base[i]);
        }
        final int backwardInnerBlock = innerBlock;
        final int backwardOuterBlock = outerBlock;
        final int backwardResolvedDim = resolvedDim;

        int cursor = 0;
        for (int outer = 0; outer < outerBlock; outer++) {
            for (GradTensor tensor : tensors) {
                int copySize = Math.toIntExact(tensor.shape[resolvedDim]) * innerBlock;
                int srcOffset = outer * copySize;
                System.arraycopy(tensor.data, srcOffset, out, cursor, copySize);
                cursor += copySize;
            }
        }
        GradTensor result = new GradTensor(out, outShape).requiresGrad(anyRequiresGrad(tensors));
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Cat") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    float[][] grads = new float[tensors.length][];
                    for (int i = 0; i < tensors.length; i++) {
                        if (tensors[i].requiresGrad()) {
                            grads[i] = new float[tensors[i].data.length];
                        }
                    }

                    int cursor = 0;
                    for (int outer = 0; outer < backwardOuterBlock; outer++) {
                        for (int i = 0; i < tensors.length; i++) {
                            GradTensor tensor = tensors[i];
                            int copySize = Math.toIntExact(tensor.shape[backwardResolvedDim]) * backwardInnerBlock;
                            int srcOffset = outer * copySize;
                            if (grads[i] != null) {
                                System.arraycopy(upstreamData, cursor, grads[i], srcOffset, copySize);
                            }
                            cursor += copySize;
                        }
                    }

                    for (int i = 0; i < tensors.length; i++) {
                        if (grads[i] != null) {
                            tensors[i].backward(GradTensor.of(grads[i], tensors[i].shape()));
                        }
                    }
                }
            });
        }
        return result;
    }

    public static GradTensor stack(GradTensor... tensors) {
        return stack(0, tensors);
    }

    public static GradTensor stack(int dim, GradTensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("tensors must not be empty");
        }
        GradTensor[] expanded = new GradTensor[tensors.length];
        for (int i = 0; i < tensors.length; i++) {
            expanded[i] = tensors[i].unsqueeze(dim);
        }
        return cat(dim, expanded);
    }

    public static GradTensor fromImage(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("image must not be null");
        }
        int width = image.getWidth();
        int height = image.getHeight();
        float[] values = new float[3 * height * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int pixel = y * width + x;
                values[pixel] = r / 255f;
                values[height * width + pixel] = g / 255f;
                values[2 * height * width + pixel] = b / 255f;
            }
        }
        return new GradTensor(values, new long[]{3, height, width});
    }

    public float[] data() {
        return data;
    }

    public long[] shape() {
        return shape.clone();
    }

    public long numel() {
        return numel(shape);
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public GradTensor requiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad && !NoGrad.active();
        return this;
    }

    public GradTensor grad() {
        return grad;
    }

    public GradTensor detach() {
        return new GradTensor(data.clone(), shape.clone());
    }

    public void zeroGrad() {
        this.grad = null;
    }

    public void setGradFn(Function.Context gradFn) {
        this.gradFn = gradFn;
    }

    public void backward() {
        backward(full(1f, shape));
    }

    public void backward(GradTensor upstream) {
        if (!requiresGrad) {
            return;
        }
        accumulateGrad(upstream);
        if (gradFn != null && !NoGrad.active()) {
            gradFn.backward(upstream);
        }
    }

    public float item() {
        return data[0];
    }

    public float item(int flatIndex) {
        return data[flatIndex];
    }

    public GradTensor reshape(long... newShape) {
        if (numel(newShape) != numel()) {
            throw new IllegalArgumentException("reshape changes tensor size");
        }
        GradTensor result = new GradTensor(data.clone(), newShape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Reshape") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor.this.backward(GradTensor.of(upstream.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor unsqueeze(int dim) {
        int resolved = dim < 0 ? dim + shape.length + 1 : dim;
        if (resolved < 0 || resolved > shape.length) {
            throw new IllegalArgumentException("invalid unsqueeze dim: " + dim);
        }
        long[] outShape = new long[shape.length + 1];
        for (int i = 0, j = 0; i < outShape.length; i++) {
            if (i == resolved) {
                outShape[i] = 1;
            } else {
                outShape[i] = shape[j++];
            }
        }
        GradTensor result = new GradTensor(data.clone(), outShape).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Unsqueeze") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor.this.backward(GradTensor.of(upstream.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor squeeze() {
        long[] squeezed = Arrays.stream(shape).filter(dim -> dim != 1).toArray();
        GradTensor result = new GradTensor(data.clone(), squeezed).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Squeeze") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor.this.backward(GradTensor.of(upstream.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor flatten() {
        GradTensor result = new GradTensor(data.clone(), new long[]{numel()}).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Flatten") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor.this.backward(GradTensor.of(upstream.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor transpose() {
        if (shape.length < 2) {
            return reshape(shape.clone());
        }
        return transpose(shape.length - 2, shape.length - 1);
    }

    public GradTensor transpose(int dim0, int dim1) {
        int rank = shape.length;
        int d0 = normalizeDim(dim0, rank);
        int d1 = normalizeDim(dim1, rank);
        int[] order = new int[rank];
        for (int i = 0; i < rank; i++) {
            order[i] = i;
        }
        order[d0] = d1;
        order[d1] = d0;
        return permute(order);
    }

    public GradTensor permute(int... dims) {
        if (dims.length != shape.length) {
            throw new IllegalArgumentException("permute rank mismatch");
        }
        boolean[] seen = new boolean[dims.length];
        long[] outShape = new long[dims.length];
        int[] resolvedDims = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            int dim = normalizeDim(dims[i], shape.length);
            if (seen[dim]) {
                throw new IllegalArgumentException("duplicate permute dim: " + dim);
            }
            seen[dim] = true;
            resolvedDims[i] = dim;
            outShape[i] = shape[dim];
        }

        float[] out = new float[data.length];
        for (int flat = 0; flat < out.length; flat++) {
            long[] outIdx = unravel(flat, outShape);
            long[] inIdx = new long[shape.length];
            for (int i = 0; i < resolvedDims.length; i++) {
                inIdx[resolvedDims[i]] = outIdx[i];
            }
            out[flat] = data[ravel(inIdx, shape)];
        }
        GradTensor result = new GradTensor(out, outShape).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            int[] inverse = new int[resolvedDims.length];
            for (int i = 0; i < resolvedDims.length; i++) {
                inverse[resolvedDims[i]] = i;
            }
            result.setGradFn(new Function.Context("Permute") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor gradInput = upstream.permute(inverse);
                    GradTensor.this.backward(GradTensor.of(gradInput.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor add(GradTensor other) {
        return binaryOp(
                other,
                (left, right) -> left + right,
                (upstream, left, right) -> upstream,
                (upstream, left, right) -> upstream,
                "Add");
    }

    public GradTensor add(float scalar) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = data[i] + scalar;
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("AddScalar") {
                @Override
                public void backward(GradTensor upstream) {
                    GradTensor.this.backward(GradTensor.of(upstream.data(), GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor sub(GradTensor other) {
        return binaryOp(
                other,
                (left, right) -> left - right,
                (upstream, left, right) -> upstream,
                (upstream, left, right) -> -upstream,
                "Sub");
    }

    public GradTensor mul(GradTensor other) {
        return binaryOp(
                other,
                (left, right) -> left * right,
                (upstream, left, right) -> upstream * right,
                (upstream, left, right) -> upstream * left,
                "Mul");
    }

    public GradTensor mul(float scalar) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = data[i] * scalar;
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("MulScalar") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = upstream.data().clone();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] *= scalar;
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor div(GradTensor other) {
        return binaryOp(
                other,
                (left, right) -> left / right,
                (upstream, left, right) -> upstream / right,
                (upstream, left, right) -> -upstream * left / (right * right),
                "Div");
    }

    public GradTensor div(float scalar) {
        return mul(1f / scalar);
    }

    public GradTensor pow(float exponent) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.pow(data[i], exponent);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Pow") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] * exponent * (float) Math.pow(data[i], exponent - 1f);
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor sqrt() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.sqrt(data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Sqrt") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] * 0.5f / out[i];
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor abs() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.abs(data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Abs") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = data[i] > 0f ? upstreamData[i] : data[i] < 0f ? -upstreamData[i] : 0f;
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor relu() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(0f, data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("ReLU") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = data[i] > 0f ? upstreamData[i] : 0f;
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor sigmoid() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = 1f / (1f + (float) Math.exp(-data[i]));
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Sigmoid") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] * out[i] * (1f - out[i]);
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor tanh() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.tanh(data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Tanh") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] * (1f - out[i] * out[i]);
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor exp() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.exp(data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Exp") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] * out[i];
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor log() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.log(data[i]);
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Log") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = upstreamData[i] / data[i];
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor silu() {
        return mul(sigmoid());
    }

    public GradTensor softmax() {
        if (shape.length == 0) {
            return new GradTensor(new float[]{1f}, new long[0]).requiresGrad(requiresGrad);
        }
        int last = Math.toIntExact(shape[shape.length - 1]);
        int outer = data.length / last;
        float[] out = new float[data.length];
        for (int row = 0; row < outer; row++) {
            int base = row * last;
            float max = data[base];
            for (int i = 1; i < last; i++) {
                max = Math.max(max, data[base + i]);
            }
            float sum = 0f;
            for (int i = 0; i < last; i++) {
                float exp = (float) Math.exp(data[base + i] - max);
                out[base + i] = exp;
                sum += exp;
            }
            for (int i = 0; i < last; i++) {
                out[base + i] /= sum;
            }
        }
        GradTensor result = new GradTensor(out, shape.clone()).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Softmax") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    for (int row = 0; row < outer; row++) {
                        int base = row * last;
                        float dot = 0f;
                        for (int i = 0; i < last; i++) {
                            dot += upstreamData[base + i] * out[base + i];
                        }
                        for (int i = 0; i < last; i++) {
                            grad[base + i] = out[base + i] * (upstreamData[base + i] - dot);
                        }
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor mean() {
        GradTensor result = scalar(VectorOps.sum(data) / data.length).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Mean") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / data.length;
                    float[] grad = new float[data.length];
                    Arrays.fill(grad, scale);
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor mean(int dim) {
        return reduce(dim, true);
    }

    public GradTensor sum() {
        GradTensor result = scalar(VectorOps.sum(data)).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("Sum") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    Arrays.fill(grad, upstream.item());
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    public GradTensor sum(int dim) {
        return reduce(dim, false);
    }

    public GradTensor matmul(GradTensor other) {
        int aRank = shape.length;
        int bRank = other.shape.length;
        if (aRank < 2 || bRank < 2) {
            throw new IllegalArgumentException("matmul requires rank >= 2");
        }

        if (bRank == 2 && aRank >= 2) {
            return matmulRightMatrix(other);
        }
        if (aRank == 3 && bRank == 3) {
            return batchedMatmul(other);
        }
        throw new UnsupportedOperationException(
                "matmul compatibility currently supports [..,m,k]x[k,n] and [b,m,k]x[b,k,n]");
    }

    public GradTensor einsum(String equation, GradTensor other) {
        return TensorOps.einsum(equation, this, other);
    }

    @Override
    public String toString() {
        return "GradTensor(shape=" + Arrays.toString(shape) + ", requiresGrad=" + requiresGrad + ")";
    }

    private GradTensor matmulRightMatrix(GradTensor other) {
        int aRank = shape.length;
        int m = Math.toIntExact(shape[aRank - 2]);
        int k = Math.toIntExact(shape[aRank - 1]);
        int otherK = Math.toIntExact(other.shape[0]);
        int n = Math.toIntExact(other.shape[1]);
        if (k != otherK) {
            throw new IllegalArgumentException("matmul inner dimension mismatch");
        }

        long[] prefix = Arrays.copyOf(shape, aRank - 2);
        int batches = Math.toIntExact(numel(prefix));
        long[] outShape = Arrays.copyOf(shape, aRank);
        outShape[aRank - 1] = n;
        float[] out = AcceleratedOps.matmul(data, shape, other.data, other.shape);
        int aBlock = m * k;
        int outBlock = m * n;
        GradTensor result = new GradTensor(out, outShape).requiresGrad(requiresGrad || other.requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("MatMul") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (GradTensor.this.requiresGrad()) {
                        float[] gradLeft = new float[GradTensor.this.data.length];
                        for (int batch = 0; batch < batches; batch++) {
                            int leftOffset = batch * aBlock;
                            int upOffset = batch * outBlock;
                            for (int row = 0; row < m; row++) {
                                for (int inner = 0; inner < k; inner++) {
                                    float acc = 0f;
                                    for (int col = 0; col < n; col++) {
                                        acc += upstreamData[upOffset + row * n + col]
                                                * other.data[inner * n + col];
                                    }
                                    gradLeft[leftOffset + row * k + inner] += acc;
                                }
                            }
                        }
                        GradTensor.this.backward(GradTensor.of(gradLeft, GradTensor.this.shape()));
                    }
                    if (other.requiresGrad()) {
                        float[] gradRight = new float[other.data.length];
                        for (int batch = 0; batch < batches; batch++) {
                            int leftOffset = batch * aBlock;
                            int upOffset = batch * outBlock;
                            for (int inner = 0; inner < k; inner++) {
                                for (int col = 0; col < n; col++) {
                                    float acc = 0f;
                                    for (int row = 0; row < m; row++) {
                                        acc += GradTensor.this.data[leftOffset + row * k + inner]
                                                * upstreamData[upOffset + row * n + col];
                                    }
                                    gradRight[inner * n + col] += acc;
                                }
                            }
                        }
                        other.backward(GradTensor.of(gradRight, other.shape()));
                    }
                }
            });
        }
        return result;
    }

    private GradTensor batchedMatmul(GradTensor other) {
        int batch = Math.toIntExact(shape[0]);
        int m = Math.toIntExact(shape[1]);
        int k = Math.toIntExact(shape[2]);
        if (other.shape[0] != shape[0] || other.shape[1] != shape[2]) {
            throw new IllegalArgumentException("batched matmul shape mismatch");
        }
        int n = Math.toIntExact(other.shape[2]);
        float[] out = AcceleratedOps.matmul(data, shape, other.data, other.shape);
        GradTensor result = new GradTensor(out, new long[]{batch, m, n}).requiresGrad(requiresGrad || other.requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("BatchedMatMul") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (GradTensor.this.requiresGrad()) {
                        float[] gradLeft = new float[GradTensor.this.data.length];
                        for (int b = 0; b < batch; b++) {
                            int leftOffset = b * m * k;
                            int rightOffset = b * k * n;
                            int upOffset = b * m * n;
                            for (int row = 0; row < m; row++) {
                                for (int inner = 0; inner < k; inner++) {
                                    float acc = 0f;
                                    for (int col = 0; col < n; col++) {
                                        acc += upstreamData[upOffset + row * n + col]
                                                * other.data[rightOffset + inner * n + col];
                                    }
                                    gradLeft[leftOffset + row * k + inner] += acc;
                                }
                            }
                        }
                        GradTensor.this.backward(GradTensor.of(gradLeft, GradTensor.this.shape()));
                    }
                    if (other.requiresGrad()) {
                        float[] gradRight = new float[other.data.length];
                        for (int b = 0; b < batch; b++) {
                            int leftOffset = b * m * k;
                            int rightOffset = b * k * n;
                            int upOffset = b * m * n;
                            for (int inner = 0; inner < k; inner++) {
                                for (int col = 0; col < n; col++) {
                                    float acc = 0f;
                                    for (int row = 0; row < m; row++) {
                                        acc += GradTensor.this.data[leftOffset + row * k + inner]
                                                * upstreamData[upOffset + row * n + col];
                                    }
                                    gradRight[rightOffset + inner * n + col] += acc;
                                }
                            }
                        }
                        other.backward(GradTensor.of(gradRight, other.shape()));
                    }
                }
            });
        }
        return result;
    }

    private GradTensor reduce(int dim, boolean mean) {
        if (shape.length == 0) {
            return new GradTensor(data.clone(), new long[0]).requiresGrad(requiresGrad);
        }
        int resolved = normalizeDim(dim, shape.length);
        long[] outShape = new long[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != resolved) {
                outShape[j++] = shape[i];
            }
        }

        float[] out = new float[Math.toIntExact(numel(outShape))];
        for (int flat = 0; flat < data.length; flat++) {
            long[] idx = unravel(flat, shape);
            long[] outIdx = new long[outShape.length];
            for (int i = 0, j = 0; i < idx.length; i++) {
                if (i != resolved) {
                    outIdx[j++] = idx[i];
                }
            }
            int outFlat = outShape.length == 0 ? 0 : ravel(outIdx, outShape);
            out[outFlat] += data[flat];
        }

        if (mean) {
            float divisor = shape[resolved];
            for (int i = 0; i < out.length; i++) {
                out[i] /= divisor;
            }
        }
        GradTensor result = new GradTensor(out, outShape).requiresGrad(requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context(mean ? "MeanDim" : "SumDim") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] grad = new float[data.length];
                    float[] upstreamData = upstream.data();
                    float scale = mean ? 1f / shape[resolved] : 1f;
                    for (int flat = 0; flat < data.length; flat++) {
                        long[] idx = unravel(flat, shape);
                        long[] outIdx = new long[outShape.length];
                        for (int i = 0, j = 0; i < idx.length; i++) {
                            if (i != resolved) {
                                outIdx[j++] = idx[i];
                            }
                        }
                        int outFlat = outShape.length == 0 ? 0 : ravel(outIdx, outShape);
                        grad[flat] = upstreamData[outFlat] * scale;
                    }
                    GradTensor.this.backward(GradTensor.of(grad, GradTensor.this.shape()));
                }
            });
        }
        return result;
    }

    private GradTensor binaryOp(
            GradTensor other,
            BinaryOp op,
            BinaryGrad leftGrad,
            BinaryGrad rightGrad,
            String opName) {
        long[] outShape = broadcastShape(shape, other.shape);
        float[] out = new float[Math.toIntExact(numel(outShape))];
        for (int flat = 0; flat < out.length; flat++) {
            long[] idx = unravel(flat, outShape);
            float left = data[offsetFor(idx, shape, outShape)];
            float right = other.data[offsetFor(idx, other.shape, outShape)];
            out[flat] = op.apply(left, right);
        }
        GradTensor result = new GradTensor(out, outShape).requiresGrad(requiresGrad || other.requiresGrad);
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context(opName) {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (GradTensor.this.requiresGrad()) {
                        float[] gradLeft = new float[GradTensor.this.data.length];
                        for (int flat = 0; flat < upstreamData.length; flat++) {
                            long[] idx = unravel(flat, outShape);
                            int leftOffset = offsetFor(idx, GradTensor.this.shape, outShape);
                            int rightOffset = offsetFor(idx, other.shape, outShape);
                            gradLeft[leftOffset] += leftGrad.apply(
                                    upstreamData[flat],
                                    GradTensor.this.data[leftOffset],
                                    other.data[rightOffset]);
                        }
                        GradTensor.this.backward(GradTensor.of(gradLeft, GradTensor.this.shape()));
                    }
                    if (other.requiresGrad()) {
                        float[] gradRight = new float[other.data.length];
                        for (int flat = 0; flat < upstreamData.length; flat++) {
                            long[] idx = unravel(flat, outShape);
                            int leftOffset = offsetFor(idx, GradTensor.this.shape, outShape);
                            int rightOffset = offsetFor(idx, other.shape, outShape);
                            gradRight[rightOffset] += rightGrad.apply(
                                    upstreamData[flat],
                                    GradTensor.this.data[leftOffset],
                                    other.data[rightOffset]);
                        }
                        other.backward(GradTensor.of(gradRight, other.shape()));
                    }
                }
            });
        }
        return result;
    }

    private void accumulateGrad(GradTensor upstream) {
        if (grad == null) {
            grad = GradTensor.of(upstream.data, upstream.shape);
            return;
        }
        if (!Arrays.equals(grad.shape, upstream.shape)) {
            throw new IllegalArgumentException("gradient shape mismatch");
        }
        for (int i = 0; i < grad.data.length; i++) {
            grad.data[i] += upstream.data[i];
        }
    }

    private static long[] normalizeShape(long[] shape) {
        Objects.requireNonNull(shape, "shape");
        for (long dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("shape dimensions must be >= 0");
            }
        }
        return shape.clone();
    }

    private static long numel(long[] shape) {
        long result = 1;
        for (long dim : shape) {
            result *= dim;
        }
        return result;
    }

    private static int normalizeDim(int dim, int rank) {
        int resolved = dim < 0 ? dim + rank : dim;
        if (resolved < 0 || resolved >= rank) {
            throw new IllegalArgumentException("invalid dim " + dim + " for rank " + rank);
        }
        return resolved;
    }

    private static long[] broadcastShape(long[] left, long[] right) {
        int rank = Math.max(left.length, right.length);
        long[] out = new long[rank];
        for (int i = 0; i < rank; i++) {
            long l = dimFromEnd(left, i);
            long r = dimFromEnd(right, i);
            if (l != r && l != 1 && r != 1) {
                throw new IllegalArgumentException(
                        "cannot broadcast " + Arrays.toString(left) + " with " + Arrays.toString(right));
            }
            out[rank - 1 - i] = Math.max(l, r);
        }
        return out;
    }

    private static long dimFromEnd(long[] shape, int fromEnd) {
        int index = shape.length - 1 - fromEnd;
        return index >= 0 ? shape[index] : 1;
    }

    private static long[] unravel(int flat, long[] shape) {
        if (shape.length == 0) {
            return new long[0];
        }
        long[] idx = new long[shape.length];
        int remaining = flat;
        for (int i = shape.length - 1; i >= 0; i--) {
            int dim = Math.toIntExact(shape[i]);
            idx[i] = remaining % dim;
            remaining /= dim;
        }
        return idx;
    }

    private static int ravel(long[] idx, long[] shape) {
        int flat = 0;
        for (int i = 0; i < shape.length; i++) {
            flat *= Math.toIntExact(shape[i]);
            flat += Math.toIntExact(idx[i]);
        }
        return flat;
    }

    private static int offsetFor(long[] broadcastIdx, long[] srcShape, long[] outShape) {
        if (srcShape.length == 0) {
            return 0;
        }
        long[] srcIdx = new long[srcShape.length];
        int offset = outShape.length - srcShape.length;
        for (int i = 0; i < srcShape.length; i++) {
            srcIdx[i] = srcShape[i] == 1 ? 0 : broadcastIdx[i + offset];
        }
        return ravel(srcIdx, srcShape);
    }

    private static boolean anyRequiresGrad(GradTensor[] tensors) {
        for (GradTensor tensor : tensors) {
            if (tensor.requiresGrad()) {
                return true;
            }
        }
        return false;
    }

    @FunctionalInterface
    private interface BinaryOp {
        float apply(float left, float right);
    }

    @FunctionalInterface
    private interface BinaryGrad {
        float apply(float upstream, float left, float right);
    }
}
