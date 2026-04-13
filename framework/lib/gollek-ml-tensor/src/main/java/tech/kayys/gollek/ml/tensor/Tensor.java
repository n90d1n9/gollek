package tech.kayys.gollek.ml.tensor;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.runtime.tensor.Device;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Unified Tensor abstraction with PyTorch-like API.
 *
 * <p>This is the primary tensor type in Gollek ML. It wraps {@link GradTensor}
 * to provide device placement, dtype control, operator overloading, and
 * convenient factory methods — mirroring {@code torch.Tensor} in PyTorch.
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Factory methods
 * Tensor x = Tensor.randn(2, 3, 224, 224).to(Device.CUDA);
 * Tensor y = Tensor.zeros(2, 1000);
 *
 * // Operator overloading
 * Tensor z = x.matmul(w).add(b).relu();
 *
 * // Autograd
 * x.requiresGrad(true);
 * z.sum().backward();
 * x.grad();  // ∂loss/∂x
 *
 * // Device placement
 * Tensor cpuTensor = z.to(Device.CPU);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class Tensor {

    private final GradTensor tensor;

    private Tensor(GradTensor tensor) {
        this.tensor = tensor;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Factory Methods (like torch.*)
    // ─────────────────────────────────────────────────────────────────────────

    /** Create a tensor from raw data with shape. */
    public static Tensor of(float[] data, long... shape) {
        return new Tensor(GradTensor.of(data, shape));
    }

    /** Create a 1D tensor from data. */
    public static Tensor of(float... data) {
        return new Tensor(GradTensor.of(data));
    }

    /** Create a tensor filled with zeros. */
    public static Tensor zeros(long... shape) {
        return new Tensor(GradTensor.zeros(shape));
    }

    /** Create a tensor filled with ones. */
    public static Tensor ones(long... shape) {
        return new Tensor(GradTensor.ones(shape));
    }

    /** Create a tensor filled with a scalar value. */
    public static Tensor full(float value, long... shape) {
        return new Tensor(GradTensor.full(value, shape));
    }

    /** Create a tensor with random normal values (mean=0, std=1). */
    public static Tensor randn(long... shape) {
        return new Tensor(GradTensor.randn(shape));
    }

    /** Create a tensor with random uniform values [0, 1). */
    public static Tensor rand(long... shape) {
        return new Tensor(GradTensor.rand(shape));
    }

    /** Create a tensor with random uniform values in [lo, hi). */
    public static Tensor uniform(double lo, double hi, long... shape) {
        return new Tensor(GradTensor.uniform(lo, hi, shape));
    }

    /** Create an identity matrix. */
    public static Tensor eye(int n) {
        return new Tensor(GradTensor.eye(n));
    }

    /** Create a tensor with values from a range. */
    public static Tensor arange(float start, float end, float step) {
        return new Tensor(GradTensor.arange(start, end, step));
    }

    /** Create a tensor with values 0..end-1. */
    public static Tensor arange(int end) {
        return new Tensor(GradTensor.arange(0, end, 1));
    }

    /** Create a scalar tensor. */
    public static Tensor scalar(float value) {
        return new Tensor(GradTensor.scalar(value));
    }

    /** Create a tensor from a 2D float array. */
    public static Tensor fromArray(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        return new Tensor(GradTensor.of(flat, rows, cols));
    }

    /** Create a tensor from a BufferedImage [C, H, W] in range [0, 1]. */
    public static Tensor fromImage(java.awt.image.BufferedImage img) {
        return new Tensor(GradTensor.fromImage(img));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device Placement
    // ─────────────────────────────────────────────────────────────────────────

    /** Move tensor to a device. */
    public Tensor to(Device device) {
        return new Tensor(tensor.to(device));
    }

    /** Get the device this tensor is on. */
    public Device device() {
        return tensor.device();
    }

    /** Check if tensor is on CUDA device. */
    public boolean isCuda() {
        return tensor.device() == Device.CUDA;
    }

    /** Check if tensor is on CPU. */
    public boolean isCpu() {
        return tensor.device() == Device.CPU;
    }

    /** Check if tensor is on Metal device. */
    public boolean isMetal() {
        return tensor.device() == Device.METAL;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dtype Control
    // ─────────────────────────────────────────────────────────────────────────

    /** Cast tensor to a different dtype. */
    public Tensor to(DType dtype) {
        return new Tensor(tensor.to(dtype));
    }

    /** Get the dtype. */
    public DType dtype() {
        return tensor.dtype();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Operations (Operator Overloading)
    // ─────────────────────────────────────────────────────────────────────────

    /** Element-wise addition. */
    public Tensor add(Tensor other) {
        return new Tensor(tensor.add(other.tensor));
    }

    /** Scalar addition. */
    public Tensor add(float scalar) {
        return new Tensor(tensor.add(scalar));
    }

    /** Element-wise subtraction. */
    public Tensor sub(Tensor other) {
        return new Tensor(tensor.sub(other.tensor));
    }

    /** Scalar subtraction. */
    public Tensor sub(float scalar) {
        return new Tensor(tensor.sub(scalar));
    }

    /** Element-wise multiplication (Hadamard product). */
    public Tensor mul(Tensor other) {
        return new Tensor(tensor.mul(other.tensor));
    }

    /** Scalar multiplication. */
    public Tensor mul(float scalar) {
        return new Tensor(tensor.mul(scalar));
    }

    /** Element-wise division. */
    public Tensor div(Tensor other) {
        return new Tensor(tensor.div(other.tensor));
    }

    /** Scalar division. */
    public Tensor div(float scalar) {
        return new Tensor(tensor.div(scalar));
    }

    /** Matrix multiplication. */
    public Tensor matmul(Tensor other) {
        return new Tensor(tensor.matmul(other.tensor));
    }

    /** ReLU activation. */
    public Tensor relu() {
        return new Tensor(tensor.relu());
    }

    /** Sigmoid activation. */
    public Tensor sigmoid() {
        return new Tensor(tensor.sigmoid());
    }

    /** Tanh activation. */
    public Tensor tanh() {
        return new Tensor(tensor.tanh());
    }

    /** SiLU / Swish activation. */
    public Tensor silu() {
        return new Tensor(tensor.silu());
    }

    /** Natural logarithm. */
    public Tensor log() {
        return new Tensor(tensor.log());
    }

    /** Exponential. */
    public Tensor exp() {
        return new Tensor(tensor.exp());
    }

    /** Softmax along the last dimension. */
    public Tensor softmax() {
        return new Tensor(tensor.softmax());
    }

    /** Log-softmax along the last dimension. */
    public Tensor logSoftmax() {
        return new Tensor(tensor.logSoftmax());
    }

    /** Sum all elements to a scalar. */
    public Tensor sum() {
        return new Tensor(tensor.sum());
    }

    /** Mean of all elements. */
    public Tensor mean() {
        return new Tensor(tensor.mean());
    }

    /** Power: x^p element-wise. */
    public Tensor pow(float p) {
        return new Tensor(tensor.pow(p));
    }

    /** Absolute value. */
    public Tensor abs() {
        return new Tensor(tensor.abs());
    }

    /** Square root. */
    public Tensor sqrt() {
        return new Tensor(tensor.sqrt());
    }

    /** Reciprocal square root. */
    public Tensor rsqrt() {
        return new Tensor(tensor.rsqrt());
    }

    /** Clamp values to [min, max]. */
    public Tensor clamp(float min, float max) {
        return new Tensor(tensor.clamp(min, max));
    }

    /** Negate: -x. */
    public Tensor neg() {
        return new Tensor(tensor.neg());
    }

    /** Transpose last two dimensions. */
    public Tensor transpose() {
        return new Tensor(tensor.transpose());
    }

    /** Reshape tensor. */
    public Tensor reshape(long... shape) {
        return new Tensor(tensor.reshape(shape));
    }

    /** Flatten to 1D. */
    public Tensor flatten() {
        return new Tensor(tensor.flatten());
    }

    /** Add a dimension of size 1. */
    public Tensor unsqueeze(int dim) {
        return new Tensor(tensor.unsqueeze(dim));
    }

    /** Remove dimensions of size 1. */
    public Tensor squeeze() {
        return new Tensor(tensor.squeeze());
    }

    /** View — alias to reshape. */
    public Tensor view(long... shape) {
        return reshape(shape);
    }

    /** Detach from computation graph. */
    public Tensor detach() {
        return new Tensor(tensor.detach());
    }

    /** Clone with full copy. */
    public Tensor clone_() {
        return new Tensor(tensor.clone_());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Autograd
    // ─────────────────────────────────────────────────────────────────────────

    /** Enable gradient computation. */
    public Tensor requiresGrad(boolean requires) {
        return new Tensor(tensor.requiresGrad(requires));
    }

    /** Check if gradients are required. */
    public boolean requiresGrad() {
        return tensor.requiresGrad();
    }

    /** Get gradient tensor. */
    public Tensor grad() {
        GradTensor g = tensor.grad();
        if (g == null) return null;
        return new Tensor(g);
    }

    /** Clear gradients. */
    public void zeroGrad() {
        tensor.zeroGrad();
    }

    /** Perform backward pass. */
    public void backward() {
        tensor.backward();
    }

    /** Perform backward pass with given upstream gradient. */
    public void backward(Tensor upstreamGrad) {
        tensor.backward(upstreamGrad.tensor);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /** Get the underlying GradTensor. */
    public GradTensor gradTensor() {
        return tensor;
    }

    /** Get raw data array (shared reference — do not mutate). */
    public float[] data() {
        return tensor.data();
    }

    /** Get tensor shape. */
    public long[] shape() {
        return tensor.shape();
    }

    /** Get number of dimensions. */
    public int ndim() {
        return tensor.ndim();
    }

    /** Get size of a dimension. */
    public long size(int dim) {
        return tensor.size(dim);
    }

    /** Get total number of elements. */
    public long numel() {
        return tensor.numel();
    }

    /** Get scalar value (tensor must have 1 element). */
    public float item() {
        return tensor.item();
    }

    /** Get element at flat index. */
    public float item(int index) {
        return tensor.item(index);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization
    // ─────────────────────────────────────────────────────────────────────────

    /** Save tensor to file. */
    public void save(Path path) throws IOException {
        try (var oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(path.toFile()))) {
            oos.writeObject(new SerializedForm(data(), shape()));
        }
    }

    /** Load tensor from file. */
    public static Tensor load(Path path) throws IOException, ClassNotFoundException {
        try (var ois = new java.io.ObjectInputStream(new java.io.FileInputStream(path.toFile()))) {
            SerializedForm form = (SerializedForm) ois.readObject();
            return of(form.data(), form.shape());
        }
    }

    private record SerializedForm(float[] data, long[] shape) implements java.io.Serializable {}

    // ─────────────────────────────────────────────────────────────────────────
    // Object Methods
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return String.format("Tensor(shape=%s, device=%s, dtype=%s, requires_grad=%b)",
                Arrays.toString(shape()), device(), dtype(), requiresGrad());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor t)) return false;
        return device().equals(t.device()) && tensor.equals(t.tensor);
    }

    @Override
    public int hashCode() {
        return 31 * tensor.hashCode() + device().hashCode();
    }
}
