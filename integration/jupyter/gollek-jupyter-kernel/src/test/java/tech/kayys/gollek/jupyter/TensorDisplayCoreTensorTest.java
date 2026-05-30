package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.memory.CpuBuffer;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DefaultTensor;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

import java.lang.foreign.ValueLayout;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorDisplayCoreTensorTest {

    @Test
    void renderSupportsCurrentCoreTensorApi() {
        DefaultTensor tensor = coreTensor2x2();

        DisplayData data = TensorDisplay.render(tensor, null);

        assertNotNull(data);
        assertNotNull(data.getData(MIMEType.TEXT_PLAIN));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("shape=[2, 2]"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("dtype=F32"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("size=4"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("min=0"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("max=3"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("sample=[0, 1, 2, 3]"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("size=4"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("sample=[0, 1, 2, 3]"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("data:image/png;base64,"));
    }

    private static DefaultTensor coreTensor2x2() {
        CpuBuffer buffer = new CpuBuffer(4L * Float.BYTES);
        buffer.segment().set(ValueLayout.JAVA_FLOAT, 0L, 0.0f);
        buffer.segment().set(ValueLayout.JAVA_FLOAT, 4L, 1.0f);
        buffer.segment().set(ValueLayout.JAVA_FLOAT, 8L, 2.0f);
        buffer.segment().set(ValueLayout.JAVA_FLOAT, 12L, 3.0f);
        return new DefaultTensor(new Shape(2, 2), DType.F32, DeviceType.CPU, buffer, NoopBackend.INSTANCE);
    }

    private enum NoopBackend implements ComputeBackend {
        INSTANCE;

        @Override
        public Tensor add(Tensor a, Tensor b) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor sub(Tensor a, Tensor b) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor mul(Tensor a, float scalar) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor div(Tensor a, float scalar) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor matmul(Tensor a, Tensor b) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<Tensor> split(Tensor a, int axis, int parts) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor attention(Tensor q, Tensor k, Tensor v) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor softmax(Tensor logits) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor slice(Tensor input, long[] offsets, long[] sizes) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor pow(Tensor input, float exponent) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor mean(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor abs(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor crossEntropy(Tensor logits, Tensor target) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor binaryCrossEntropy(Tensor logits, Tensor target) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor reshape(Tensor input, long... newShape) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor cast(Tensor input, DType toDType) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor to(Tensor input, DeviceType toDevice) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor mul(Tensor a, Tensor b) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor div(Tensor a, Tensor b) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor addScalar(Tensor input, float scalar) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor zerosLike(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor sqrt(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor relu(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor sigmoid(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor tanh(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor log(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor exp(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor silu(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor flatten(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor unsqueeze(Tensor input, int dim) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor squeeze(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor transpose(Tensor input) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Tensor transpose(Tensor input, int dim0, int dim1) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long numel(Tensor a) {
            throw new UnsupportedOperationException();
        }
    }
}
