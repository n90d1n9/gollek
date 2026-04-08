package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.hub.ModelHub;
import tech.kayys.gollek.ml.nlp.*;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.runtime.tensor.Device;
import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.sdk.api.GollekSdkBuilder;
import tech.kayys.gollek.sdk.multimodal.AudioBuilder;
import tech.kayys.gollek.sdk.multimodal.MultimodalBuilder;
import tech.kayys.gollek.sdk.multimodal.VideoBuilder;
import tech.kayys.gollek.sdk.multimodal.VisionBuilder;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, running NLP pipelines, performing multimodal inference,
 * and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch while extending it with
 * multimodal AI capabilities.
 *
 * <h3>Quick Start — Tensors &amp; Neural Networks</h3>
 * 
 * <pre>{@code
 * import tech.kayys.gollek.ml.Gollek;
 * import tech.kayys.gollek.ml.nn.*;
 * import tech.kayys.gollek.ml.autograd.GradTensor;
 *
 * // Create tensors
 * var x = Gollek.tensor(new float[]{1, 2, 3, 4}, 2, 2);
 * var w = Gollek.randn(2, 3).requiresGrad(true);
 * var y = x.matmul(w).relu().sum();
 * y.backward();
 *
 * // Build a neural network
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ReLU(),
 *     new Linear(256, 10)
 * );
 * System.out.println("Parameters: " + model.parameterCountFormatted());
 * }</pre>
 *
 * <h3>Quick Start — NLP Pipelines</h3>
 * 
 * <pre>{@code
 * var gen = Gollek.pipeline("text-generation", "Qwen/Qwen2.5-0.5B");
 * System.out.println(gen.process("Tell me about Java"));
 * }</pre>
 *
 * <h3>Quick Start — Multimodal AI</h3>
 * 
 * <pre>{@code
 * // Vision: describe an image
 * var result = Gollek.vision("gemini-2.0-flash")
 *         .image(Path.of("photo.jpg"))
 *         .prompt("What's in this image?")
 *         .generate();
 * System.out.println(result.text());
 *
 * // Audio: transcribe
 * var transcript = Gollek.audio("whisper-large-v3")
 *         .audioFile(Path.of("meeting.wav"))
 *         .generate();
 *
 * // Mixed modality
 * var analysis = Gollek.multimodal("gpt-4o")
 *         .text("Compare these images")
 *         .image(Path.of("a.png"))
 *         .image(Path.of("b.png"))
 *         .generate();
 * }</pre>
 *
 * @see tech.kayys.gollek.ml.autograd.GradTensor
 * @see PipelineFactory
 * @see tech.kayys.gollek.sdk.multimodal.VisionBuilder
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.1";

    private Gollek() {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation (like torch.*)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Creates a tensor from a flat float array with the given shape.
     *
     * @param data  flat array of values in row-major order
     * @param shape dimensions of the tensor; product must equal {@code data.length}
     * @return a new {@link GradTensor}
     */
    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    /**
     * Creates a 1-D tensor from a varargs float array.
     *
     * @param data the values
     * @return a 1-D {@link GradTensor} of length {@code data.length}
     */
    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * @param shape dimensions of the tensor
     * @return a zero-filled {@link GradTensor}
     */
    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    /**
     * Creates a tensor filled with ones.
     *
     * @param shape dimensions of the tensor
     * @return a ones-filled {@link GradTensor}
     */
    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    /**
     * Creates a tensor with values sampled from a standard normal distribution
     * (μ=0, σ=1).
     *
     * @param shape dimensions of the tensor
     * @return a randomly initialised {@link GradTensor}
     */
    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    /**
     * Creates a tensor with values sampled uniformly from {@code [0, 1)}.
     *
     * @param shape dimensions of the tensor
     * @return a randomly initialised {@link GradTensor}
     */
    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    /**
     * Creates a 1-D tensor with evenly spaced values from {@code start} (inclusive)
     * to {@code end} (exclusive) with the given {@code step}.
     *
     * @param start first value (inclusive)
     * @param end   upper bound (exclusive)
     * @param step  spacing between consecutive values; must be non-zero
     * @return a 1-D {@link GradTensor}
     */
    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    /**
     * Creates a 1-D tensor with integer values {@code [0, end)}.
     *
     * @param end upper bound (exclusive); must be &gt; 0
     * @return a 1-D {@link GradTensor} of length {@code end}
     */
    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    /**
     * Creates a 0-D (scalar) tensor wrapping a single float value.
     *
     * @param value the scalar value
     * @return a scalar {@link GradTensor}
     */
    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    /**
     * Creates an {@code n×n} identity matrix.
     *
     * @param n size of the square matrix; must be &gt; 0
     * @return an identity {@link GradTensor}
     */
    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    /**
     * Creates a tensor filled with a constant value.
     *
     * @param value the fill value
     * @param shape dimensions of the tensor
     * @return a constant-filled {@link GradTensor}
     */
    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Opens a no-gradient context that disables autograd tracking for all
     * operations performed within the block.
     *
     * <p>
     * Use with try-with-resources to ensure the context is properly closed:
     * 
     * <pre>{@code
     * try (var _ = Gollek.noGrad()) {
     *     float[] logits = model.forward(input).data();
     * }
     * }</pre>
     *
     * @return a {@link NoGrad} context handle; must be closed after use
     */
    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Pipeline Factory
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Creates an NLP pipeline for the given task and model.
     *
     * <h3>Built-in tasks</h3>
     * <ul>
     * <li>{@code "text-generation"} — auto-regressive text generation</li>
     * <li>{@code "text-classification"} — zero-shot sentiment/topic
     * classification</li>
     * <li>{@code "embedding"} — dense vector embeddings</li>
     * </ul>
     *
     * <h3>Plugin-contributed tasks</h3>
     * <p>
     * Additional pipeline tasks are auto-discovered from plugins on the classpath
     * via {@link java.util.ServiceLoader} and {@link PipelineProvider}.
     * </p>
     *
     * @param <I>     input type inferred from the task
     * @param <O>     output type inferred from the task
     * @param task    pipeline task name
     * @param modelId model identifier (HuggingFace repo ID or local path)
     * @return a configured {@link Pipeline} instance
     * @throws PipelineException if {@code task} is not registered
     */
    @SuppressWarnings("unchecked")
    public static <I, O> Pipeline<I, O> pipeline(String task, String modelId) {
        return PipelineFactory.create(task, modelId);
    }

    /**
     * Creates an NLP pipeline from a fully specified {@link PipelineConfig}.
     *
     * @param <I>    input type inferred from the task
     * @param <O>    output type inferred from the task
     * @param config pipeline configuration including task, model, and sampling
     *               parameters
     * @return a configured {@link Pipeline} instance
     * @throws PipelineException if the task in {@code config} is not registered
     */
    @SuppressWarnings("unchecked")
    public static <I, O> Pipeline<I, O> pipeline(PipelineConfig config) {
        return PipelineFactory.create(config);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Builders
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Creates a vision (image understanding) builder for the given model.
     * <p>
     * Uses a {@link GollekSdk} resolved via {@link java.util.ServiceLoader}.
     *
     * <pre>{@code
     * var result = Gollek.vision("gemini-2.0-flash")
     *         .image(Path.of("cat.jpg"))
     *         .prompt("What animal is this?")
     *         .generate();
     * }</pre>
     *
     * @param model model identifier (e.g., "gemini-2.0-flash", "gpt-4o")
     * @return a fluent {@link VisionBuilder}
     */
    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, resolveDefaultSdk());
    }

    /**
     * Creates an audio processing builder for the given model.
     * <p>
     * Supports transcription, translation, and audio understanding tasks.
     *
     * <pre>{@code
     * var result = Gollek.audio("whisper-large-v3")
     *         .audioFile(Path.of("meeting.wav"))
     *         .task("transcription")
     *         .generate();
     * }</pre>
     *
     * @param model model identifier (e.g., "whisper-large-v3", "gemini-2.0-flash")
     * @return a fluent {@link AudioBuilder}
     */
    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, resolveDefaultSdk());
    }

    /**
     * Creates a video understanding builder for the given model.
     *
     * <pre>{@code
     * var result = Gollek.video("gemini-2.0-pro")
     *         .videoFile(Path.of("clip.mp4"))
     *         .prompt("Summarize the key events")
     *         .generate();
     * }</pre>
     *
     * @param model model identifier
     * @return a fluent {@link VideoBuilder}
     */
    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, resolveDefaultSdk());
    }

    /**
     * Creates a mixed-modality builder for the given model.
     * <p>
     * Accepts arbitrary combinations of text, images, audio, video, and documents.
     *
     * <pre>{@code
     * var result = Gollek.multimodal("gpt-4o")
     *         .text("Compare these images")
     *         .image(Path.of("before.png"))
     *         .image(Path.of("after.png"))
     *         .generate();
     * }</pre>
     *
     * @param model model identifier
     * @return a fluent {@link MultimodalBuilder}
     */
    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, resolveDefaultSdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // SDK Builder
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Creates a builder for constructing a configured {@link GollekSdk} instance.
     *
     * <pre>{@code
     * var sdk = Gollek.sdk()
     *         .provider("gemini")
     *         .apiKey(System.getenv("GEMINI_API_KEY"))
     *         .model("gemini-2.0-flash")
     *         .build();
     * }</pre>
     *
     * @return a new SDK builder
     */
    public static GollekSdkBuilder sdk() {
        return GollekSdk.builder();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Hub
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Loads pretrained weights from a model identifier and returns them as a
     * name-to-tensor map.
     *
     * @param modelId HuggingFace repo ID, local file path, or Gollek model name
     * @return map of parameter names to their {@link GradTensor} values
     */
    public static java.util.Map<String, GradTensor> loadWeights(String modelId) {
        return ModelHub.loadWeights(modelId);
    }

    /**
     * Loads pretrained weights from a model identifier directly into a module's
     * named parameters.
     *
     * @param module  the target module whose parameters will be overwritten
     * @param modelId HuggingFace repo ID, local file path, or Gollek model name
     */
    public static void loadInto(Module module, String modelId) {
        ModelHub.loadInto(module, modelId);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Returns {@code true} if a CUDA-capable GPU is detected.
     */
    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    /**
     * Returns {@code true} if Apple Metal GPU acceleration is available.
     */
    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    /**
     * Returns the best available compute device in priority order:
     * CUDA &gt; Metal &gt; CPU.
     *
     * @return the recommended {@link Device} for this environment
     */
    public static Device defaultDevice() {
        if (isCudaAvailable())
            return Device.CUDA;
        if (isMetalAvailable())
            return Device.METAL;
        return Device.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Prints a formatted summary of the framework version and detected hardware
     * capabilities to standard output.
     */
    public static void printInfo() {
        System.out.println("╔══════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework             ║");
        System.out.println("║          Version " + VERSION + "                    ║");
        System.out.println("╠══════════════════════════════════════════╣");
        System.out.println("║  Device: " + pad(defaultDevice().toString(), 31) + "║");
        System.out.println("║  CUDA:   " + pad(String.valueOf(isCudaAvailable()), 31) + "║");
        System.out.println("║  Metal:  " + pad(String.valueOf(isMetalAvailable()), 31) + "║");
        System.out.println("║  Java:   " + pad(System.getProperty("java.version"), 31) + "║");

        // Show available pipelines
        var tasks = PipelineFactory.availableTasks();
        System.out.println("║  Tasks:  " + pad(String.join(", ", tasks), 31) + "║");

        System.out.println("╚══════════════════════════════════════════╝");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Internal
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Resolves the default GollekSdk via ServiceLoader.
     * Cached after first resolution.
     */
    private static volatile GollekSdk cachedSdk;

    private static GollekSdk resolveDefaultSdk() {
        if (cachedSdk != null)
            return cachedSdk;
        synchronized (Gollek.class) {
            if (cachedSdk != null)
                return cachedSdk;
            try {
                cachedSdk = GollekSdk.builder().build();
            } catch (Exception e) {
                throw new IllegalStateException(
                        "No GollekSdkProvider found. Add a provider to the classpath " +
                                "(e.g., gollek-sdk-core, gollek-plugin-openai, gollek-plugin-gemini).",
                        e);
            }
            return cachedSdk;
        }
    }

    private static String pad(String s, int len) {
        if (s.length() >= len)
            return s.substring(0, len);
        return s + " ".repeat(len - s.length());
    }
}
