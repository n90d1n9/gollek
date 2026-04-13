package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.layer.ReLU;
import tech.kayys.gollek.ml.nn.layer.Dropout;
import tech.kayys.gollek.ml.nn.layer.Sequential;
import tech.kayys.gollek.ml.nn.layer.LayerNorm;
import tech.kayys.gollek.ml.nn.layer.Embedding;
import tech.kayys.gollek.ml.transformer.MultiHeadAttention;
import tech.kayys.gollek.ml.cnn.Conv2d;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.L1Loss;
import tech.kayys.gollek.ml.tensor.Tensor;
import tech.kayys.gollek.ml.tensor.Device;
import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.ml.multimodal.VisionBuilder;
import tech.kayys.gollek.ml.multimodal.MultimodalBuilder;
import tech.kayys.gollek.ml.multimodal.VideoBuilder;
import tech.kayys.gollek.ml.multimodal.AudioBuilder;
import tech.kayys.gollek.sdk.api.GollekSdk;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch.
 *
 * <h3>Quick Start — Tensors &amp; Neural Networks</h3>
 * 
 * <pre>{@code
 * var x = Gollek.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
 * var w = Gollek.randn(2, 3).requiresGrad(true);
 * var y = x.matmul(w).relu().sum();
 * y.backward();
 *
 * var model = new Sequential(
 *         new Linear(784, 256),
 *         new ReLU(),
 *         new Linear(256, 10));
 * }</pre>
 *
 * <h3>Quick Start — Training</h3>
 * 
 * <pre>{@code
 * var model = new Sequential(
 *         new Linear(784, 128), new ReLU(), new Dropout(0.1), new Linear(128, 10));
 *
 * var optimizer = new Adam(model.parameters(), 1e-3);
 * var lossFn = new CrossEntropyLoss();
 *
 * for (int epoch = 0; epoch < 10; epoch++) {
 *     optimizer.zeroGrad();
 *     var pred = model.forward(batch);
 *     var loss = lossFn.compute(pred, target);
 *     loss.backward();
 *     optimizer.step();
 * }
 * }</pre>
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.1";

    private Gollek() {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation — delegates to GradTensor
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    public static GradTensor uniform(double lo, double hi, long... shape) {
        return GradTensor.uniform(lo, hi, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Operations (static)
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor cat(GradTensor... tensors) {
        return GradTensor.cat(tensors);
    }

    public static GradTensor cat(int dim, GradTensor... tensors) {
        return GradTensor.cat(dim, tensors);
    }

    public static GradTensor stack(GradTensor... tensors) {
        return GradTensor.stack(tensors);
    }

    public static GradTensor stack(int dim, GradTensor... tensors) {
        return GradTensor.stack(dim, tensors);
    }

    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        return GradTensor.where(condition, x, y);
    }

    public static GradTensor einsum(String equation, GradTensor a, GradTensor b) {
        return a.einsum(equation, b);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    private static GollekSdk _sdk;

    private static synchronized GollekSdk sdk() {
        if (_sdk == null) {
            _sdk = GollekSdk.builder().build();
        }
        return _sdk;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Facade
    // ══════════════════════════════════════════════════════════════════════

    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, sdk());
    }

    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, sdk());
    }

    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, sdk());
    }

    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, sdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    public static tech.kayys.gollek.runtime.tensor.Device defaultDevice() {
        if (isCudaAvailable())
            return tech.kayys.gollek.runtime.tensor.Device.CUDA;
        if (isMetalAvailable())
            return tech.kayys.gollek.runtime.tensor.Device.METAL;
        return tech.kayys.gollek.runtime.tensor.Device.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    public static void printInfo() {
        var device = defaultDevice();
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework                         ║");
        System.out.println("║          Version " + String.format("%-32s", VERSION) + "║");
        System.out.println("╠══════════════════════════════════════════════════════╣");
        System.out.println("║  Device:  " + String.format("%-32s", device) + "║");
        System.out.println("║  CUDA:    " + String.format("%-32s", isCudaAvailable()) + "║");
        System.out.println("║  Metal:   " + String.format("%-32s", isMetalAvailable()) + "║");
        System.out.println("║  Java:    " + String.format("%-32s", System.getProperty("java.version")) + "║");
        System.out.println("║  Vector:  "
                + String.format("%-32s", jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() + "-bit")
                + "║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
