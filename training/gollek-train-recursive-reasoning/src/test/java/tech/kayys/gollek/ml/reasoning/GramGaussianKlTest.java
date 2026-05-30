package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;

class GramGaussianKlTest {
    @Test
    void computesMeanDiagonalGaussianKlForScalarLatents() {
        GramLatentGaussian posterior = new GramLatentGaussian(new ScalarTensor(1.0f), new ScalarTensor(0.0f));
        GramLatentGaussian prior = new GramLatentGaussian(new ScalarTensor(0.0f), new ScalarTensor(0.0f));

        Tensor kl = GramGaussianKl.meanPosteriorToPrior(posterior, prior);

        assertEquals(0.5f, kl.item(), 1e-6f);
    }

    @Test
    void tensorObjectiveAppliesOptionalGramWeights() {
        GramObjectiveConfig config = GramObjectiveConfig.builder()
                .klBeta(0.2)
                .klBalance(0.5)
                .latentProcessRewardWeight(0.25)
                .adaptiveComputationWeight(0.125)
                .build();

        Tensor loss = GramTensorObjective.loss(
                new ScalarTensor(2.0f),
                new ScalarTensor(10.0f),
                new ScalarTensor(4.0f),
                new ScalarTensor(8.0f),
                config);

        assertEquals(6.0f, loss.item(), 1e-6f);
    }

    @Test
    void latentGaussianRequiresMatchingShapes() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new GramLatentGaussian(new ScalarTensor(0.0f, new Shape(1)), new ScalarTensor(0.0f, new Shape(2))));
    }

    private record ScalarTensor(float value, Shape shape) implements Tensor {
        private ScalarTensor(float value) {
            this(value, new Shape(1));
        }

        @Override
        public DeviceType device() {
            return DeviceType.CPU;
        }

        @Override
        public DType dtype() {
            return DType.F32;
        }

        @Override
        public ComputeBackend backend() {
            return null;
        }

        @Override
        public Tensor add(Tensor other) {
            return new ScalarTensor(value + other.item(), shape);
        }

        @Override
        public Tensor sub(Tensor other) {
            return new ScalarTensor(value - other.item(), shape);
        }

        @Override
        public Tensor mul(Tensor other) {
            return new ScalarTensor(value * other.item(), shape);
        }

        @Override
        public Tensor mul(float scalar) {
            return new ScalarTensor(value * scalar, shape);
        }

        @Override
        public Tensor div(float scalar) {
            return new ScalarTensor(value / scalar, shape);
        }

        @Override
        public Tensor matmul(Tensor other) {
            return mul(other);
        }

        @Override
        public Tensor reshape(long... newShape) {
            return new ScalarTensor(value, new Shape(newShape));
        }

        @Override
        public Tensor softmax() {
            return this;
        }

        @Override
        public Tensor slice(long[] offsets, long[] sizes) {
            return this;
        }

        @Override
        public Tensor pow(float exponent) {
            return new ScalarTensor((float) Math.pow(value, exponent), shape);
        }

        @Override
        public Tensor mean() {
            return this;
        }

        @Override
        public Tensor abs() {
            return new ScalarTensor(Math.abs(value), shape);
        }

        @Override
        public Tensor crossEntropy(Tensor target) {
            return sub(target).abs();
        }

        @Override
        public Tensor binaryCrossEntropy(Tensor target) {
            return crossEntropy(target);
        }

        @Override
        public Tensor div(Tensor other) {
            return new ScalarTensor(value / other.item(), shape);
        }

        @Override
        public Tensor add(float scalar) {
            return new ScalarTensor(value + scalar, shape);
        }

        @Override
        public Tensor zerosLike() {
            return new ScalarTensor(0.0f, shape);
        }

        @Override
        public Tensor sqrt() {
            return new ScalarTensor((float) Math.sqrt(value), shape);
        }

        @Override
        public Tensor cast(DType dtype) {
            return this;
        }

        @Override
        public Tensor to(DeviceType device) {
            return this;
        }

        @Override
        public float item() {
            return value;
        }

        @Override
        public void backward() {
        }

        @Override
        public Tensor grad() {
            return zerosLike();
        }

        @Override
        public void setGrad(Tensor grad) {
        }

        @Override
        public boolean requiresGrad() {
            return false;
        }

        @Override
        public void setRequiresGrad(boolean requiresGrad) {
        }

        @Override
        public Tensor relu() {
            return new ScalarTensor(Math.max(0.0f, value), shape);
        }

        @Override
        public Tensor sigmoid() {
            return new ScalarTensor((float) (1.0 / (1.0 + Math.exp(-value))), shape);
        }

        @Override
        public Tensor tanh() {
            return new ScalarTensor((float) Math.tanh(value), shape);
        }

        @Override
        public Tensor log() {
            return new ScalarTensor((float) Math.log(value), shape);
        }

        @Override
        public Tensor exp() {
            return new ScalarTensor((float) Math.exp(value), shape);
        }

        @Override
        public Tensor silu() {
            return mul(sigmoid());
        }

        @Override
        public Tensor flatten() {
            return this;
        }

        @Override
        public Tensor unsqueeze(int dim) {
            return this;
        }

        @Override
        public Tensor squeeze() {
            return this;
        }

        @Override
        public Tensor transpose() {
            return this;
        }

        @Override
        public Tensor transpose(int dim0, int dim1) {
            return this;
        }

        @Override
        public long numel() {
            return shape.numel();
        }
    }
}
