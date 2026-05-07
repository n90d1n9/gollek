package tech.kayys.gollek.backend.metal;

import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.core.memory.Buffer;
import tech.kayys.gollek.core.memory.CpuBuffer;
import java.util.List;

public final class MetalBackend implements ComputeBackend {

    public MetalBackend() {
        NativeMetal.init();
    }

    private DefaultTensor asDefault(Tensor t) {
        if (t instanceof DefaultTensor dt) {
            return dt;
        }
        throw new IllegalArgumentException("MetalBackend only supports DefaultTensor");
    }

    private long byteSize(DType dtype) {
        return switch (dtype) {
            case F32, I32 -> 4;
            case F16, BF16 -> 2;
            case I8 -> 1;
            case Q4_K_M -> 0; // Not directly addressable
            case Q8_0 -> 1; // Simplified
        };
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        DefaultTensor da = asDefault(a);
        DefaultTensor db = asDefault(b);
        
        int M = (int) a.shape().dim(0);
        int K = (int) a.shape().dim(1);
        int N = (int) b.shape().dim(1);
        
        Shape shapeC = new Shape(M, N);
        long sizeBytes = shapeC.numel() * byteSize(a.dtype());
        
        CpuBuffer bufferC = new CpuBuffer(sizeBytes);
        NativeMetal.matmul(bufferC.segment(), da.buffer().segment(), db.buffer().segment(), M, K, N, 1.0f, 0.0f);
        
        return new DefaultTensor(shapeC, a.dtype(), a.device(), bufferC, this);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        DefaultTensor dQ = asDefault(Q);
        DefaultTensor dK = asDefault(K);
        DefaultTensor dV = asDefault(V);

        // Extract dimensions from Q: [Batch, Tokens, Heads, Dim]
        int B = (int) Q.shape().dim(0);
        int T = (int) Q.shape().dim(1);
        int H = (int) Q.shape().dim(2);
        int D = (int) Q.shape().dim(3);

        Shape shapeOut = Q.shape();
        long sizeBytes = shapeOut.numel() * byteSize(Q.dtype());
        CpuBuffer bufferOut = new CpuBuffer(sizeBytes);

        // These are typically provided by the model runtime, 
        // for now we use empty segments if not available
        // In a real scenario, these should be part of the attention request
        java.lang.foreign.MemorySegment empty = java.lang.foreign.MemorySegment.NULL;

        NativeMetal.attention(bufferOut.segment(), dQ.buffer().segment(), dK.buffer().segment(), dV.buffer().segment(),
                empty, empty, B, T, H, D, 16, 1024, (float)(1.0/Math.sqrt(D)), true, 0.0f);

        return new DefaultTensor(shapeOut, Q.dtype(), Q.device(), bufferOut, this);
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Metal 'add' kernel not yet linked");
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Metal 'sub' kernel not yet linked");
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        throw new UnsupportedOperationException("Metal 'mul scalar' kernel not yet linked");
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Metal 'mul tensor' kernel not yet linked");
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        throw new UnsupportedOperationException("Metal 'div scalar' kernel not yet linked");
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("Metal 'div tensor' kernel not yet linked");
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        throw new UnsupportedOperationException("Metal 'add scalar' kernel not yet linked");
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        DefaultTensor da = asDefault(a);
        return new DefaultTensor(new Shape(newShape), a.dtype(), a.device(), da.buffer(), this);
    }

    @Override
    public Tensor softmax(Tensor a) {
        throw new UnsupportedOperationException("Metal 'softmax' kernel not yet linked");
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        throw new UnsupportedOperationException("Metal 'slice' kernel not yet linked");
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        throw new UnsupportedOperationException("Metal 'split' kernel not yet linked");
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        throw new UnsupportedOperationException("Metal 'pow' kernel not yet linked");
    }

    @Override
    public Tensor mean(Tensor a) {
        throw new UnsupportedOperationException("Metal 'mean' kernel not yet linked");
    }

    @Override
    public Tensor abs(Tensor a) {
        throw new UnsupportedOperationException("Metal 'abs' kernel not yet linked");
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        throw new UnsupportedOperationException("Metal 'crossEntropy' kernel not yet linked");
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        throw new UnsupportedOperationException("Metal 'binaryCrossEntropy' kernel not yet linked");
    }

    @Override
    public Tensor cast(Tensor a, DType dtype) {
        throw new UnsupportedOperationException("Metal 'cast' kernel not yet linked");
    }

    @Override
    public Tensor to(Tensor a, DeviceType device) {
        throw new UnsupportedOperationException("Metal 'to' device transfer not yet linked");
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        Shape shape = a.shape();
        long sizeBytes = shape.numel() * byteSize(a.dtype());
        CpuBuffer buffer = new CpuBuffer(sizeBytes);
        // CpuBuffer is zeroed on allocation by default in Java FFM allocate
        return new DefaultTensor(shape, a.dtype(), a.device(), buffer, this);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        throw new UnsupportedOperationException("Metal 'sqrt' kernel not yet linked");
    }

    @Override
    public Tensor relu(Tensor a) {
        throw new UnsupportedOperationException("Metal 'relu' kernel not yet linked");
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        throw new UnsupportedOperationException("Metal 'sigmoid' kernel not yet linked");
    }

    @Override
    public Tensor tanh(Tensor a) {
        throw new UnsupportedOperationException("Metal 'tanh' kernel not yet linked");
    }

    @Override
    public Tensor log(Tensor a) {
        throw new UnsupportedOperationException("Metal 'log' kernel not yet linked");
    }

    @Override
    public Tensor exp(Tensor a) {
        throw new UnsupportedOperationException("Metal 'exp' kernel not yet linked");
    }

    @Override
    public Tensor silu(Tensor a) {
        throw new UnsupportedOperationException("Metal 'silu' kernel not yet linked");
    }

    @Override
    public Tensor flatten(Tensor a) {
        return reshape(a, a.numel());
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        long[] oldDims = a.shape().dims();
        long[] newDims = new long[oldDims.length + 1];
        System.arraycopy(oldDims, 0, newDims, 0, dim);
        newDims[dim] = 1;
        System.arraycopy(oldDims, dim, newDims, dim + 1, oldDims.length - dim);
        return reshape(a, newDims);
    }

    @Override
    public Tensor squeeze(Tensor a) {
        long[] squeezed = java.util.Arrays.stream(a.shape().dims())
                .filter(dim -> dim != 1)
                .toArray();
        return reshape(a, squeezed.length == 0 ? new long[] { 1 } : squeezed);
    }

    @Override
    public Tensor transpose(Tensor a) {
        throw new UnsupportedOperationException("Metal 'transpose' kernel not yet linked");
    }

    @Override
    public Tensor transpose(Tensor a, int dim0, int dim1) {
        throw new UnsupportedOperationException("Metal 'transpose(dim0, dim1)' kernel not yet linked");
    }

    @Override
    public long numel(Tensor a) {
        return a.numel();
    }
}
