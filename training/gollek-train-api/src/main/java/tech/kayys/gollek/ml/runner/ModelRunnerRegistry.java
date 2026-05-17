package tech.kayys.gollek.ml.runner;

import tech.kayys.gollek.core.model.ModelFormat;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Registry that manages ModelRunner instances and provides framework-specific
 * runner creation based on model format.
 *
 * <p>
 * The registry auto-selects the appropriate backend:
 * 
 * <pre>
 *   GGUF → GgufRunner
 *   SAFETENSORS → SafetensorRunner
 *   LITERT → LiteRTRunner
 *   ONNX → OnnxRunner
 *   TORCHSCRIPT → TorchScriptRunner
 *   TENSORRT → TensorRTRunner
 * </pre>
 *
 * <p>
 * Custom runners can be registered:
 * 
 * <pre>
 * ModelRunnerRegistry.get().register(ModelFormat.CUSTOM, CustomRunner::new);
 * </pre>
 *
 * @since 0.3.0
 */
public class ModelRunnerRegistry {

    private static volatile ModelRunnerRegistry instance;

    /** Runner factories by format */
    private final Map<ModelFormat, Function<RunnerConfig, ModelRunner>> runnerFactories = new ConcurrentHashMap<>();

    /** Active runner instances by ID */
    private final Map<String, ModelRunner> activeRunners = new ConcurrentHashMap<>();

    private ModelRunnerRegistry() {
        registerDefaults();
    }

    /**
     * Gets the singleton registry instance.
     */
    public static ModelRunnerRegistry get() {
        if (instance == null) {
            synchronized (ModelRunnerRegistry.class) {
                if (instance == null) {
                    instance = new ModelRunnerRegistry();
                }
            }
        }
        return instance;
    }

    /**
     * Resets the singleton instance (useful for testing).
     */
    public static void reset() {
        instance = null;
    }

    /**
     * Registers a runner factory for a model format.
     *
     * @param format  the model format
     * @param factory function to create runners for this format
     */
    public void register(ModelFormat format, Function<RunnerConfig, ModelRunner> factory) {
        runnerFactories.put(format, factory);
    }

    /**
     * Creates a new runner for the given format and path.
     *
     * @param id        runner identifier
     * @param format    model format
     * @param modelPath path to the model file/directory
     * @param device    target device
     * @param options   framework-specific options
     * @return configured model runner
     */
    public ModelRunner createRunner(String id, ModelFormat format, Path modelPath,
            RunnerDevice device, Map<String, Object> options) {
        Function<RunnerConfig, ModelRunner> factory = runnerFactories.get(format);
        if (factory == null) {
            throw new IllegalArgumentException("No runner registered for format: " + format);
        }

        RunnerConfig config = new RunnerConfig(id, format, modelPath, device, options);
        ModelRunner runner = factory.apply(config);

        activeRunners.put(id, runner);
        return runner;
    }

    /**
     * Gets an active runner by ID.
     *
     * @param id runner identifier
     * @return the runner, or null if not found
     */
    public ModelRunner getRunner(String id) {
        return activeRunners.get(id);
    }

    /**
     * Gets all active runners.
     */
    public Map<String, ModelRunner> getActiveRunners() {
        return Map.copyOf(activeRunners);
    }

    /**
     * Removes and closes a runner.
     *
     * @param id runner identifier
     */
    public void removeRunner(String id) {
        ModelRunner runner = activeRunners.remove(id);
        if (runner != null) {
            runner.close();
        }
    }

    /**
     * Closes all active runners.
     */
    public void closeAll() {
        for (ModelRunner runner : activeRunners.values()) {
            try {
                runner.close();
            } catch (Exception e) {
                // Log and continue
            }
        }
        activeRunners.clear();
    }

    /**
     * Registers default runners for all supported formats.
     */
    private void registerDefaults() {
        registerReflective(ModelFormat.ONNX,
                "tech.kayys.gollek.ml.runner.onnx.OnnxModelRunner",
                "tech.kayys.gollek.inference.onnx.OnnxModelRunner");
        registerUnsupported(ModelFormat.GGUF,
                "Unified GGUF ModelRunner is not available yet. Use the GGUF plugin runner path for now.");
        registerUnsupported(ModelFormat.SAFETENSORS,
                "Unified SafeTensors ModelRunner is not available yet. Use the safetensor serving runner path for now.");
        registerUnsupported(ModelFormat.LITERT,
                "Unified LiteRT ModelRunner is not available yet. Use the LiteRT plugin/provider path for now.");
        registerUnsupported(ModelFormat.TORCHSCRIPT,
                "Unified TorchScript ModelRunner is not available yet. Use the libtorch/provider path for now.");
        registerUnsupported(ModelFormat.TENSORRT,
                "Unified TensorRT ModelRunner is not available yet. Use the TensorRT plugin/provider path for now.");
    }

    private void registerReflective(ModelFormat format, String... candidateClassNames) {
        register(format, config -> instantiateReflective(format, config, candidateClassNames));
    }

    private void registerUnsupported(ModelFormat format, String message) {
        register(format, config -> {
            throw new UnsupportedOperationException(message);
        });
    }

    private ModelRunner instantiateReflective(ModelFormat format, RunnerConfig config, String... candidateClassNames) {
        RuntimeException failure = null;
        for (String className : candidateClassNames) {
            try {
                Class<?> clazz = Class.forName(className);
                return (ModelRunner) clazz.getDeclaredConstructor(RunnerConfig.class).newInstance(config);
            } catch (ClassNotFoundException e) {
                failure = new RuntimeException("No reflective ModelRunner registered for " + format
                        + ". Tried: " + String.join(", ", candidateClassNames), e);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create " + format + " runner from " + className, e);
            }
        }
        throw failure != null ? failure
                : new RuntimeException("No reflective ModelRunner registered for " + format);
    }

    /**
     * Configuration for creating a model runner.
     */
    public record RunnerConfig(
            String id,
            ModelFormat format,
            Path modelPath,
            RunnerDevice device,
            Map<String, Object> options) {
    }
}
