package tech.kayys.gollek.ml;

import tech.kayys.gollek.autograd.*;
import tech.kayys.gollek.core.util.ThreadPools;
import tech.kayys.gollek.ir.CrossEntropyGrad;
import tech.kayys.gollek.ir.LayerNormGrad;
import tech.kayys.gollek.ir.OpRegistry;
import tech.kayys.gollek.ir.SoftmaxGrad;
import tech.kayys.gollek.ir.schema.OpSchemaRegistry;
import tech.kayys.gollek.ml.tensor.Tensor;

/**
 * Central initialization point for Gollek ML.
 * Call this once at application startup.
 */
public final class GollekInitializer {
    private static volatile boolean initialized = false;
    private static GradRegistry gradRegistry;
    private static OpRegistry opRegistry;
    private static OpSchemaRegistry schemaRegistry;

    private GollekInitializer() {
    } // Prevent instantiation

    /**
     * Initialize all Gollek systems.
     * Call this once at application startup.
     */
    public static synchronized void initialize() {
        if (initialized)
            return;

        initializeGradRegistry();

        // Add shutdown hook for cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down Gollek...");
            // Clean up resources
            ThreadPools.shutdown();
            initialized = false;
        }));

        initialized = true;
    }

    private static void initializeGradRegistry() {
        gradRegistry = new GradRegistry();

        // Register all gradient functions
        gradRegistry.register("attention", new AttentionGrad());
        gradRegistry.register("matmul", new MatMulGrad());
        gradRegistry.register("cross_entropy", new CrossEntropyGrad());
        gradRegistry.register("softmax", new SoftmaxGrad());
        gradRegistry.register("layernorm", new LayerNormGrad());

        // ADD THESE:
        gradRegistry.register("gelu", new GeluGrad());
        gradRegistry.register("add", new AddGrad());
        gradRegistry.register("mul", new MulGrad());
        gradRegistry.register("sub", new SubGrad()); // You'll need to create this
        gradRegistry.register("div", new DivGrad()); // You'll need to create this
        gradRegistry.register("relu", new ReluGrad()); // You'll need to create this
        gradRegistry.register("sigmoid", new SigmoidGrad()); // Create this

        System.out.println("Registered " + gradRegistry.size() + " gradient functions");
    }

    private static void initializeOpRegistry() {
        opRegistry = new OpRegistry();

        // Register core operations
        // opRegistry.register(new OpDescriptor(new OpId("core", "add", 1)));
        // opRegistry.register(new OpDescriptor(new OpId("core", "mul", 1)));
        // ... etc
    }

    private static void initializeSchemaRegistry() {
        schemaRegistry = new OpSchemaRegistry();

        // Register schemas for validation
        // schemaRegistry.register(new OpSchema("add", 2, 2, 1, ...));
    }

    // Getters for registries
    public static GradRegistry getGradRegistry() {
        checkInitialized();
        return gradRegistry;
    }

    public static OpRegistry getOpRegistry() {
        checkInitialized();
        return opRegistry;
    }

    public static OpSchemaRegistry getSchemaRegistry() {
        checkInitialized();
        return schemaRegistry;
    }

    private static void checkInitialized() {
        if (!initialized) {
            throw new IllegalStateException(
                    "Gollek not initialized. Call GollekInitializer.initialize() first.");
        }
    }

    public static boolean isInitialized() {
        return initialized;
    }
}