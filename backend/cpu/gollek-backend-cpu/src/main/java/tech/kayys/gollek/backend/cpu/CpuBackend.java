package tech.kayys.gollek.backend.cpu;

import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.core.memory.*;
import java.lang.foreign.ValueLayout;

public final class CpuBackend implements ComputeBackend {
    @Override
    public Tensor add(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = a.buffer().segment();
        var sb = b.buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            float vb = sb.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va + vb);
        }
        return new DefaultTensor(
                shape,
                DType.FLOAT32,
                Device.CPU,
                outBuf,
                this);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = a.buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va * scalar);
        }
        return new DefaultTensor(
                shape,
                DType.FLOAT32,
                Device.CPU,
                outBuf,
                this);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("matmul not implemented yet");
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return new DefaultTensor(
                new Shape(newShape),
                a.dtype(),
                a.device(),
                a.buffer(), // zero-copy view
                this);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        int seqLen = (int) Q.shape().dim(0);
        int dim = (int) Q.shape().dim(1);

        // Use your FlashAttention implementation
        try {
            return FlashAttentionCpu.forward(Q, K, V,
                    Runtime.getRuntime().availableProcessors());
        } catch (Exception e) {
            // Fallback to naive implementation
            return naiveAttention(Q, K, V);
        }
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = a.buffer().segment();
        var sb = b.buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            float vb = sb.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va - vb);
        }
        return new DefaultTensor(shape, DType.FLOAT32, Device.CPU, outBuf, this);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return mul(a, 1.0f / scalar);
    }

    @Override
    public Tensor softmax(Tensor a) {
        throw new UnsupportedOperationException("softmax not implemented yet");
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        throw new UnsupportedOperationException("slice not implemented yet");
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        throw new UnsupportedOperationException("split not implemented yet");
    }

    private Tensor naiveAttention(Tensor Q, Tensor K, Tensor V) {
        // QK^T / sqrt(d)
        Tensor scores = matmul(Q, K.transpose());
        float scale = (float) (1.0 / Math.sqrt(Q.shape().dim(1)));
        scores = mul(scores, scale);

        // softmax
        Tensor probs = softmax(scores);

        // output
        return matmul(probs, V);
    }
}