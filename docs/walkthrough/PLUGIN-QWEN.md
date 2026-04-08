# Gollek Plugin System - Development Guide

## Overview

The Gollek Plugin System provides a **flexible, manifest-based architecture** for extending inference capabilities without modifying core code. It enables dynamic loading, hot-reload, and capability-based routing for multiple plugin types.

> **"Build AI features once. Extend with plugins. Run anywhere."**

---

## Architecture

### Plugin System Layers

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer                          │
│  - Capability-based routing                             │
│  - Plugin availability checking                         │
│  - Deployment mode configuration                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Plugin Manager Layer                       │
│  - PluginRegistry / PluginManager                       │
│  - PluginAvailabilityChecker                            │
│  - Lifecycle management                                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Plugin Loader Layer                        │
│  - ServiceLoader discovery                              │
│  - JAR manifest parsing                                 │
│  - ClassLoader isolation                                │
│  - Hot-reload support                                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Plugin Types                               │
│  - Runner Plugins    (model execution)                  │
│  - Kernel Plugins    (GPU kernels)                      │
│  - Provider Plugins  (cloud APIs)                       │
│  - Optimization Plugins (performance)                    │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
~/.gollek/plugins/
├── runners/
│   ├── gguf-runner.jar
│   ├── onnx-runner.jar
│   └── litert-runner.jar
├── kernels/
│   ├── cuda-kernel.jar
│   ├── rocm-kernel.jar
│   └── metal-kernel.jar
├── providers/
│   ├── openai-provider.jar
│   ├── anthropic-provider.jar
│   └── google-provider.jar
└── optimizations/
    ├── flash-attention-3.jar
    ├── paged-attention.jar
    └── kv-cache.jar
```

---

## Plugin Types

### 1. Runner Plugins

**Purpose**: Execute inference for specific model formats

**SPI Location**: `inference-gollek/core/spi/gollek-spi-inference/`

**Key Interface**: `RunnerPlugin`

**Capabilities**:
- Model format detection (GGUF, ONNX, TFLite)
- Architecture support (Llama, Mistral, Qwen, Gemma)
- Quantization support (Q4_K_M, Q5_K_M, Q8_0, etc.)
- Execution modes (sync, async, streaming)

**Example**:
```java
public class GGUFRunnerPlugin implements RunnerPlugin {
    
    @Override
    public String id() {
        return "gguf-runner";
    }
    
    @Override
    public Set<String> supportedFormats() {
        return Set.of("gguf");
    }
    
    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("llama", "mistral", "qwen", "gemma", "phi");
    }
    
    @Override
    public <T> RunnerResult<T> execute(
        RunnerRequest request,
        RunnerContext context) throws RunnerException {
        
        // Load model if needed
        ModelHandle model = loadModel(request.getModelPath());
        
        // Execute inference
        return switch (request.getType()) {
            case INFER -> infer(model, request);
            case EMBED -> embed(model, request);
            case TOKENIZE -> tokenize(model, request);
            default -> throw new UnknownRequestException(request.getType());
        };
    }
}
```

**Manifest**:
```manifest
Plugin-Id: gguf-runner
Plugin-Type: runner
Plugin-Version: 2.0.0
Plugin-Name: GGUF Runner
Plugin-Provider: tech.kayys.gollek.plugin.runner.gguf.GGUFRunnerPlugin
Plugin-Capabilities: gguf-inference, llama-architecture, mistral-architecture
Plugin-Deployment: standalone,microservice,hybrid
Plugin-Performance-Speedup: 1x (baseline)
```

---

### 2. Kernel Plugins

**Purpose**: Provide platform-specific GPU kernel implementations

**SPI Location**: `inference-gollek/core/spi/gollek-spi-plugin/`

**Key Interface**: `KernelPlugin`

**Capabilities**:
- Platform support (CUDA, ROCm, Metal, CPU)
- Kernel operations (GEMM, Attention, LayerNorm)
- Architecture support (Volta, Turing, Ampere, Hopper, Blackwell)
- Optimization features (FlashAttention, PagedAttention)

**Example**:
```java
public class CudaKernelPlugin implements KernelPlugin {
    
    @Override
    public String id() {
        return "cuda-kernel";
    }
    
    @Override
    public String platform() {
        return "cuda";
    }
    
    @Override
    public KernelValidationResult validate() {
        List<String> errors = new ArrayList<>();
        
        if (!isCudaAvailable()) {
            errors.add("CUDA not available");
        }
        
        int computeCap = getCudaComputeCapability();
        if (computeCap < 60) {
            errors.add("Compute capability 6.0+ required");
        }
        
        return KernelValidationResult.builder()
            .valid(errors.isEmpty())
            .errors(errors)
            .build();
    }
    
    @Override
    public <T> KernelResult<T> execute(
        KernelOperation operation,
        KernelContext context) throws KernelException {
        
        return switch (operation.getName()) {
            case "gemm" -> executeGemm(operation, context);
            case "attention" -> executeAttention(operation, context);
            case "layer_norm" -> executeLayerNorm(operation, context);
            default -> throw new UnknownOperationException("cuda", operation.getName());
        };
    }
    
    @Override
    public Set<String> supportedOperations() {
        return Set.of("gemm", "attention", "layer_norm", "activation", "quantize");
    }
}
```

**Manifest**:
```manifest
Plugin-Id: cuda-kernel
Plugin-Type: kernel
Plugin-Version: 2.0.0
Plugin-Name: CUDA Kernel
Plugin-Provider: tech.kayys.gollek.plugin.kernel.cuda.CudaKernelPlugin
Plugin-Capabilities: cuda-acceleration, flash-attention-2, flash-attention-3, paged-attention
Plugin-GPU-Requirement: NVIDIA GPU, CUDA 11.0+
Plugin-Minimum-Compute-Capability: 6.0
Plugin-Minimum-Memory: 4GB
Plugin-Performance-Speedup: 5-10x (vs CPU)
```

---

### 3. Provider Plugins

**Purpose**: Integrate cloud-based inference providers

**SPI Location**: `inference-gollek/core/spi/gollek-spi-provider/`

**Key Interface**: `LLMProvider`

**Capabilities**:
- API integration (OpenAI, Anthropic, Google, Mistral)
- Model support (GPT-4, Claude, Gemini, etc.)
- Features (streaming, vision, function calling)
- Rate limiting and quota management

**Example**:
```java
public class OpenAiProvider implements LLMProvider {
    
    @Override
    public String id() {
        return "openai-provider";
    }
    
    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(true)
            .vision(true)
            .functionCalling(true)
            .models(List.of("gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"))
            .build();
    }
    
    @Override
    public Uni<ProviderResponse> infer(ProviderRequest request) {
        // Call OpenAI API
        return openAiClient.chat()
            .model(request.getModel())
            .messages(request.getMessages())
            .maxTokens(request.getMaxTokens())
            .execute()
            .map(this::toProviderResponse);
    }
    
    @Override
    public Multi<ProviderResponse> inferStream(ProviderRequest request) {
        // Stream from OpenAI API
        return openAiClient.chat()
            .model(request.getModel())
            .messages(request.getMessages())
            .stream()
            .map(this::toProviderResponse);
    }
}
```

**Manifest**:
```manifest
Plugin-Id: openai-provider
Plugin-Type: provider
Plugin-Version: 2.0.0
Plugin-Name: OpenAI Provider
Plugin-Provider: tech.kayys.gollek.plugin.provider.openai.OpenAiProvider
Plugin-Capabilities: openai-api, gpt-4, gpt-3.5-turbo, embeddings, streaming, vision, function-calling
Plugin-Deployment: standalone,microservice,hybrid
```

---

### 4. Optimization Plugins

**Purpose**: Apply performance optimizations during inference

**SPI Location**: `inference-gollek/core/gollek-plugin-optimization-core/`

**Key Interface**: `OptimizationPlugin`

**Capabilities**:
- Attention optimizations (FlashAttention-2/3, PagedAttention)
- KV cache optimizations
- Memory management
- Batching strategies

**Example**:
```java
public class FlashAttention3Plugin implements OptimizationPlugin {
    
    @Override
    public String id() {
        return "flash-attention-3";
    }
    
    @Override
    public boolean isAvailable() {
        // Check for Hopper+ GPU
        return GpuDetector.isNvidia() &&
               GpuDetector.getComputeCapability() >= 90;
    }
    
    @Override
    public int priority() {
        return 100; // High priority
    }
    
    @Override
    public boolean apply(ExecutionContext context) {
        try {
            // Get Q, K, V buffers
            MemoryBuffer qBuffer = context.getBuffer("q").orElse(null);
            MemoryBuffer kBuffer = context.getBuffer("k").orElse(null);
            MemoryBuffer vBuffer = context.getBuffer("v").orElse(null);
            
            // Apply FlashAttention-3 kernel
            return applyFlashAttention3(
                qBuffer.getPointer(),
                kBuffer.getPointer(),
                vBuffer.getPointer(),
                context.getHiddenSize(),
                context.getNumHeads()
            );
            
        } catch (Exception e) {
            LOG.errorf("FlashAttention-3 failed: %s", e.getMessage());
            return false; // Fallback to standard attention
        }
    }
}
```

**Manifest**:
```manifest
Plugin-Id: flash-attention-3
Plugin-Type: optimization
Plugin-Version: 2.0.0
Plugin-Name: FlashAttention-3
Plugin-Provider: tech.kayys.gollek.plugin.optimization.fa3.FlashAttention3Plugin
Plugin-Capabilities: flash-attention-3, optimized-attention
Plugin-Dependencies: cuda-kernel
Plugin-GPU-Requirement: NVIDIA Hopper+ (H100), CUDA 12.0+
Plugin-Minimum-Compute-Capability: 9.0
Plugin-Minimum-Memory: 16GB
Plugin-Performance-Speedup: 2-3x (vs standard attention)
```

---

## Plugin Lifecycle

### Lifecycle States

```
DISCOVERED → VALIDATING → VALIDATED → INITIALIZING → ACTIVE → STOPPED
                ↓              ↓             ↓              ↓
              ERROR         ERROR        ERROR          ERROR
```

### Lifecycle Methods

```java
public interface GollekPlugin {
    
    // Identity
    String id();
    String name();
    String version();
    String description();
    
    // Validation (before initialization)
    default PluginValidationResult validate() {
        return PluginValidationResult.valid();
    }
    
    // Initialization
    void initialize(PluginContext context) throws PluginException;
    
    // Health check
    default PluginHealth health() {
        return PluginHealth.healthy();
    }
    
    // Availability check
    boolean isAvailable();
    
    // Shutdown
    default void shutdown() {
        // Cleanup resources
    }
}
```

### Lifecycle Example

```java
public class MyPlugin implements GollekPlugin {
    
    private volatile boolean initialized = false;
    private PluginConfig config;
    
    @Override
    public PluginValidationResult validate() {
        List<String> errors = new ArrayList<>();
        
        // Check hardware requirements
        if (!isHardwareAvailable()) {
            errors.add("Required hardware not available");
        }
        
        // Check dependencies
        if (!areDependenciesSatisfied()) {
            errors.add("Missing dependencies");
        }
        
        return PluginValidationResult.builder()
            .valid(errors.isEmpty())
            .errors(errors)
            .build();
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        if (initialized) {
            throw new PluginInitializationException("Already initialized");
        }
        
        try {
            this.config = context.getConfig();
            
            // Initialize resources
            initializeResources();
            
            initialized = true;
            LOG.infof("Plugin %s v%s initialized successfully", id(), version());
            
        } catch (Exception e) {
            throw new PluginInitializationException(
                "Failed to initialize plugin: " + e.getMessage(), e);
        }
    }
    
    @Override
    public PluginHealth health() {
        if (!initialized) {
            return PluginHealth.unhealthy("Not initialized");
        }
        
        try {
            // Perform health check
            boolean healthy = performHealthCheck();
            
            if (healthy) {
                return PluginHealth.healthy(Map.of(
                    "status", "running",
                    "uptime", getUptime()
                ));
            } else {
                return PluginHealth.unhealthy("Health check failed");
            }
            
        } catch (Exception e) {
            return PluginHealth.unhealthy("Health check error: " + e.getMessage());
        }
    }
    
    @Override
    public void shutdown() {
        if (initialized) {
            try {
                // Cleanup resources
                cleanupResources();
                LOG.infof("Plugin %s shutdown complete", id());
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down plugin %s", id());
            }
            initialized = false;
        }
    }
}
```

---

## Manifest-Based Plugin System

### Required Manifest Entries

```manifest
# Unique plugin identifier
Plugin-Id: my-plugin

# Plugin type (runner, kernel, provider, optimization)
Plugin-Type: runner

# Semantic version
Plugin-Version: 2.0.0

# Human-readable name
Plugin-Name: My Plugin

# Fully qualified main class
Plugin-Provider: com.example.MyPlugin
```

### Optional Manifest Entries

```manifest
# Capabilities provided
Plugin-Capabilities: capability-1, capability-2

# Plugin dependencies
Plugin-Dependencies: dependency-plugin-1, dependency-plugin-2

# Supported deployment modes
Plugin-Deployment: standalone,microservice,hybrid

# GPU requirements
Plugin-GPU-Requirement: NVIDIA GPU, CUDA 12.0+
Plugin-Minimum-Compute-Capability: 8.0
Plugin-Minimum-Memory: 8GB

# Performance metadata
Plugin-Performance-Speedup: 2-3x
Plugin-Performance-Memory-Overhead: 100MB

# Author and vendor
Plugin-Author: Your Name
Plugin-Vendor: Your Company
Plugin-License: MIT
Plugin-Homepage: https://example.com
Plugin-Documentation: https://example.com/docs
Plugin-Repository: https://github.com/example/plugin
```

### Building with Maven

```xml
<project>
    <groupId>com.example</groupId>
    <artifactId>my-gollek-plugin</artifactId>
    <version>2.0.0</version>
    
    <dependencies>
        <!-- Plugin SPI -->
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-plugin</artifactId>
            <version>1.0.0-SNAPSHOT</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.5.0</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addDefaultImplementationEntries>true</addDefaultImplementationEntries>
                        </manifest>
                        <manifestEntries>
                            <!-- Required -->
                            <Plugin-Id>my-plugin</Plugin-Id>
                            <Plugin-Type>runner</Plugin-Type>
                            <Plugin-Version>${project.version}</Plugin-Version>
                            <Plugin-Name>My Plugin</Plugin-Name>
                            <Plugin-Provider>com.example.MyPlugin</Plugin-Provider>
                            
                            <!-- Optional -->
                            <Plugin-Capabilities>my-capability, another-capability</Plugin-Capabilities>
                            <Plugin-Deployment>standalone,microservice</Plugin-Deployment>
                            <Plugin-Author>Your Name</Plugin-Author>
                            <Plugin-Vendor>Your Company</Plugin-Vendor>
                            <Plugin-License>MIT</Plugin-License>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

---

## Plugin Discovery & Loading

### Discovery Mechanisms

1. **ServiceLoader** - `META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin`
2. **JAR Manifest** - Parse manifest entries from JAR files
3. **Plugin Directory** - Scan `~/.gollek/plugins/` subdirectories
4. **Explicit Registration** - Programmatic registration

### Plugin Loader

```java
// Get plugin loader
PluginLoader loader = PluginLoader.getInstance();

// Load all plugins from default directory
List<GollekPlugin> plugins = loader.loadAll();

// Load specific plugin
GollekPlugin plugin = loader.loadFromJar(
    Path.of("/path/to/plugin.jar"));

// Hot-reload plugin
loader.reloadPlugin(Path.of("/path/to/new-version.jar"));

// Unload plugin
loader.unloadPlugin("my-plugin");

// Close loader (cleanup)
loader.close();
```

### Plugin Manager

```java
@Inject
PluginManager pluginManager;

// Initialize all plugins
pluginManager.initialize();

// Start all plugins
pluginManager.start();

// Get all plugins
Collection<GollekPlugin> allPlugins = pluginManager.all();

// Get plugin by ID
Optional<GollekPlugin> plugin = pluginManager.byId("my-plugin");

// Get plugins by capability
List<GollekPlugin> plugins = pluginManager.byCapability("cuda-acceleration");

// Check health
Map<String, PluginHealth> health = pluginManager.getHealthStatus();

// Stop all plugins
pluginManager.stop();
```

---

## Capability-Based Routing

### Plugin Availability Checker

```java
@Inject
PluginAvailabilityChecker pluginChecker;

// Check if capability is available
if (pluginChecker.hasCapability("gguf-inference")) {
    System.out.println("GGUF inference is available");
}

// Get plugins that provide capability
List<PluginDescriptor> plugins = pluginChecker.getPluginsForCapability(
    "cuda-acceleration");

for (PluginDescriptor plugin : plugins) {
    System.out.printf("Plugin: %s, Version: %s, Speedup: %s\n",
        plugin.getId(),
        plugin.getVersion(),
        plugin.getMetadata("speedup"));
}

// Get required plugins for capability
String required = pluginChecker.getRequiredPluginsForCapability(
    "flash-attention-3");

// List all available capabilities
Set<String> capabilities = pluginChecker.getAvailableCapabilities();
```

### Capability-Based Selection

```java
public class InferenceService {
    
    @Inject
    PluginManager pluginManager;
    
    @Inject
    PluginAvailabilityChecker pluginChecker;
    
    public InferenceResponse infer(String prompt) {
        // Find runner plugin with required capability
        List<GollekPlugin> runners = pluginManager.byCapability("gguf-inference");
        
        if (runners.isEmpty()) {
            throw new PluginNotAvailableException(
                "runner", "gguf-inference");
        }
        
        // Select best runner (could use priority, performance, etc.)
        RunnerPlugin runner = (RunnerPlugin) runners.get(0);
        
        // Find optimization plugins
        List<OptimizationPlugin> optimizations = pluginManager
            .byCapability("optimized-attention")
            .stream()
            .map(p -> (OptimizationPlugin) p)
            .filter(OptimizationPlugin::isAvailable)
            .sorted(Comparator.comparingInt(OptimizationPlugin::priority).reversed())
            .collect(Collectors.toList());
        
        // Execute with optimizations
        ExecutionContext context = createExecutionContext(prompt);
        
        for (OptimizationPlugin opt : optimizations) {
            opt.apply(context);
        }
        
        return runner.execute(buildRequest(prompt), RunnerContext.empty());
    }
}
```

---

## Deployment Modes

### STANDALONE Mode

**Description**: All plugins compiled into single JAR

**Configuration**:
```bash
-Dgollek.deployment.mode=standalone
```

**Pros**:
- Simple deployment
- No external dependencies
- Fast startup

**Cons**:
- No hot-reload
- Larger JAR size
- Requires rebuild for updates

**Use Case**: Edge deployment, embedded systems

---

### MICROSERVICE Mode

**Description**: Dynamic plugin loading from directory

**Configuration**:
```bash
-Dgollek.deployment.mode=microservice
-Dgollek.plugin.directory=~/.gollek/plugins
```

**Pros**:
- Hot-reload support
- Smaller initial footprint
- Easy plugin updates

**Cons**:
- More complex deployment
- Slower startup (plugin discovery)
- Runtime dependency management

**Use Case**: Development, cloud deployment

---

### HYBRID Mode (Default)

**Description**: Built-in plugins + dynamic loading

**Configuration**:
```bash
-Dgollek.deployment.mode=hybrid
```

**Pros**:
- Best of both worlds
- Core plugins built-in
- Optional plugins dynamic

**Cons**:
- Slightly more complex

**Use Case**: Production deployment

---

## Building and Running

### Build Commands

```bash
# Build entire Gollek platform
cd inference-gollek
mvn clean install

# Build specific plugin module
mvn clean install -pl plugins/runner/gguf-runner

# Build plugin JAR
cd plugins/runner/gguf-runner
mvn clean package

# Build with tests skipped
mvn clean install -DskipTests
```

### Running with Plugins

```bash
# Default (hybrid mode)
java -jar runtime/gollek-runtime.jar

# Microservice mode with custom plugin directory
java -Dgollek.deployment.mode=microservice \
     -Dgollek.plugin.directory=/opt/gollek/plugins \
     -jar runtime/gollek-runtime.jar

# With GPU acceleration
java -Dgollek.gpu.enabled=true \
     -Dgollek.gpu.device=cuda:0 \
     -Dgollek.plugin.directory=~/.gollek/plugins \
     -jar runtime/gollek-runtime.jar
```

### Docker Deployment

```yaml
version: '3.8'
services:
  gollek:
    image: gollek-ai/gollek:latest
    ports:
      - "8080:8080"
    environment:
      - GOLLEK_DEPLOYMENT_MODE=hybrid
      - GOLLEK_PLUGIN_DIRECTORY=/plugins
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ~/.gollek:/root/.gollek
      - ./plugins:/plugins
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Error Handling

### Exception Hierarchy

```
PluginException (base)
├── PluginInitializationException
├── PluginExecutionException
├── PluginNotFoundException
├── PluginValidationException
├── UnknownOperationException
└── NoPluginsAvailableException
```

### Error Handling Example

```java
try {
    // Check plugin availability
    if (!pluginChecker.hasCapability("cuda-acceleration")) {
        throw new PluginNotAvailableException(
            "kernel",
            "cuda-acceleration",
            pluginChecker.getPluginDirectory());
    }
    
    // Get and execute plugin
    List<GollekPlugin> kernels = pluginManager.byCapability("cuda-acceleration");
    KernelPlugin kernel = (KernelPlugin) kernels.get(0);
    
    KernelResult<Matrix> result = kernel.execute(operation, context);
    
} catch (PluginNotFoundException e) {
    LOG.errorf("Plugin not found: %s", e.getPluginId());
    LOG.info(e.getInstallationInstructions());
    
} catch (PluginInitializationException e) {
    LOG.errorf("Plugin initialization failed: %s", e.getMessage());
    
} catch (PluginExecutionException e) {
    LOG.errorf("Plugin execution failed: %s", e.getMessage(), e);
    
} catch (NoPluginsAvailableException e) {
    LOG.error("No plugins available");
    LOG.info(e.getInstallationInstructions());
    System.exit(1);
}
```

---

## Observability

### Plugin Metrics

```java
@Inject
PluginManager pluginManager;

// Get metrics
PluginMetrics metrics = pluginManager.getMetrics();

// Counters
long totalOps = metrics.getCounter("total_operations");
long errors = metrics.getErrorCount("execution_failed");

// Per-plugin metrics
PluginStats stats = metrics.getPluginStats("cuda-kernel");
System.out.printf("CUDA Kernel: count=%d, avg=%.2fms, success=%.2f%%\n",
    stats.getCount(),
    stats.getAverageDuration(),
    stats.getSuccessRate() * 100);

// Export all metrics
Map<String, Object> metricsMap = metrics.toMap();
```

### Health Monitoring

```java
// Get health status for all plugins
Map<String, PluginHealth> health = pluginManager.getHealthStatus();

health.forEach((pluginId, h) -> {
    if (h.isHealthy()) {
        LOG.infof("%s: ✓ %s", pluginId, h.getMessage());
    } else {
        LOG.warnf("%s: ✗ %s", pluginId, h.getMessage());
    }
});

// Get detailed health
PluginHealth health = pluginManager.health("cuda-kernel");
System.out.println("Status: " + (health.isHealthy() ? "Healthy" : "Unhealthy"));
System.out.println("Message: " + health.getMessage());
System.out.println("Details: " + health.getDetails());
```

### Logging

```properties
# Plugin logging configuration
logger.tech.kayys.gollek.plugin.level=INFO
logger.tech.kayys.gollek.plugin.category=Plugin

# Specific plugin logging
logger.tech.kayys.gollek.plugin.kernel.cuda.level=DEBUG
logger.tech.kayys.gollek.plugin.runner.gguf.level=DEBUG
```

---

## Testing

### Unit Tests

```java
class MyPluginTest {
    
    private GollekPlugin plugin;
    
    @BeforeEach
    void setUp() {
        plugin = new MyPlugin();
    }
    
    @Test
    void testValidation() {
        PluginValidationResult result = plugin.validate();
        
        assertTrue(result.isValid());
        assertTrue(result.getErrors().isEmpty());
    }
    
    @Test
    void testInitialization() {
        PluginContext context = PluginContext.empty();
        
        assertDoesNotThrow(() -> plugin.initialize(context));
        assertTrue(plugin.isAvailable());
    }
    
    @Test
    void testHealth() {
        PluginHealth health = plugin.health();
        
        assertTrue(health.isHealthy());
    }
    
    @Test
    void testExecution() throws PluginException {
        PluginRequest request = PluginRequest.builder()
            .operation("test-op")
            .parameter("param", "value")
            .build();
        
        PluginContext context = PluginContext.empty();
        
        PluginResult<Object> result = plugin.execute(request, context);
        
        assertTrue(result.isSuccess());
        assertNotNull(result.getData());
    }
}
```

### Integration Tests

```java
@QuarkusTest
class PluginIntegrationTest {
    
    @Inject
    PluginManager pluginManager;
    
    @Inject
    PluginAvailabilityChecker pluginChecker;
    
    @Test
    void testPluginDiscovery() {
        // Check plugin is discovered
        assertTrue(pluginChecker.hasCapability("gguf-inference"));
        
        // Get plugin
        List<GollekPlugin> plugins = pluginManager.byCapability("gguf-inference");
        assertFalse(plugins.isEmpty());
    }
    
    @Test
    void testPluginExecution() throws PluginException {
        RunnerPlugin runner = (RunnerPlugin) pluginManager.byId("gguf-runner");
        assertNotNull(runner);
        
        RunnerRequest request = RunnerRequest.builder()
            .type(RequestType.INFER)
            .modelPath("/models/test.gguf")
            .build();
        
        RunnerResult<InferenceResponse> result = runner.execute(
            request, RunnerContext.empty());
        
        assertTrue(result.isSuccess());
    }
    
    @Test
    void testHealthMonitoring() {
        Map<String, PluginHealth> health = pluginManager.getHealthStatus();
        
        health.forEach((id, h) -> {
            assertTrue(h.isHealthy(), 
                () -> "Plugin " + id + " is unhealthy: " + h.getMessage());
        });
    }
}
```

---

## Best Practices

### 1. Plugin Design

✅ **DO**:
- Keep plugins focused on single responsibility
- Implement comprehensive validation
- Provide detailed error messages
- Support graceful degradation
- Include health checks
- Document supported capabilities

❌ **DON'T**:
- Create monolithic plugins
- Hide error details
- Fail silently
- Skip validation
- Ignore resource cleanup

### 2. Resource Management

```java
@Override
public void shutdown() {
    // Clean up in reverse order of initialization
    if (nativeBuffer != null) {
        nativeBuffer.close();
    }
    if (cudaStream != 0) {
        CudaBindings.cudaStreamDestroy(cudaStream);
    }
    if (modelHandle != null) {
        unloadModel(modelHandle);
    }
}
```

### 3. Thread Safety

```java
private final ReentrantLock lock = new ReentrantLock();

@Override
public PluginResult<Object> execute(PluginRequest request, PluginContext context) {
    lock.lock();
    try {
        // Thread-safe execution
        return executeInternal(request, context);
    } finally {
        lock.unlock();
    }
}
```

### 4. Error Handling

```java
@Override
public PluginResult<Object> execute(PluginRequest request, PluginContext context) {
    try {
        // Validate inputs
        validateRequest(request);
        
        // Execute operation
        Object result = executeOperation(request, context);
        
        return PluginResult.success(result);
        
    } catch (PluginException e) {
        // Re-throw plugin exceptions
        throw e;
        
    } catch (Exception e) {
        // Wrap unexpected exceptions
        throw new PluginExecutionException(id(), 
            "Execution failed: " + e.getMessage(), e);
    }
}
```

### 5. Logging

```java
private static final Logger LOG = Logger.getLogger(MyPlugin.class);

@Override
public void initialize(PluginContext context) {
    LOG.infof("Initializing plugin %s v%s", id(), version());
    
    try {
        // Initialization logic
        LOG.debugf("Loading configuration: %s", config);
        
        LOG.infof("Plugin %s initialized successfully", id());
        
    } catch (Exception e) {
        LOG.errorf(e, "Failed to initialize plugin %s", id());
        throw new PluginInitializationException("Initialization failed", e);
    }
}
```

---

## Troubleshooting

### Plugin Not Discovered

**Symptoms**: Plugin JAR in directory but not found

**Solutions**:
1. Verify manifest entries:
   ```bash
   jar xf plugin.jar META-INF/MANIFEST.MF
   cat META-INF/MANIFEST.MF
   ```
2. Check required entries: `Plugin-Id`, `Plugin-Type`, `Plugin-Provider`
3. Verify deployment mode compatibility
4. Check logs: `tail -f ~/.gollek/logs/gollek.log | grep "Loading plugin"`

### Capability Not Found

**Symptoms**: `hasCapability()` returns false

**Solutions**:
1. Check `Plugin-Capabilities` in manifest
2. Verify capability name matches exactly (case-sensitive)
3. Check if plugin is loaded: `pluginChecker.getDiscoveredPlugins()`
4. Verify plugin passed validation

### Plugin Fails Validation

**Symptoms**: `PluginValidationResult` returns errors

**Solutions**:
1. Check hardware availability (GPU present)
2. Verify drivers installed (`nvidia-smi`, `rocm-smi`)
3. Check library dependencies
4. Review validation errors: `result.getErrors()`

### ClassLoader Issues

**Symptoms**: `ClassNotFoundException` or `NoClassDefFoundError`

**Solutions**:
1. Ensure dependencies are in plugin JAR or provided
2. Check parent-first package configuration
3. Verify no conflicting library versions
4. Review ClassLoader isolation strategy

### Hot-Reload Not Working

**Symptoms**: Plugin doesn't reload on file change

**Solutions**:
1. Check file watcher started (review logs)
2. Verify JAR file not locked by another process
3. Check file permissions allow reading
4. Verify plugin directory is being watched

---

## Resources

### Documentation
- **Main Docs**: https://gollek-ai.github.io
- **Plugin Guide**: https://gollek-ai.github.io/docs/plugins
- **SPI Reference**: https://gollek-ai.github.io/docs/spi
- **Examples**: https://github.com/gollek-ai/gollek/tree/main/plugins

### Source Code
- **Plugin SPI**: `inference-gollek/core/spi/gollek-spi-plugin/`
- **Plugin Core**: `inference-gollek/core/gollek-plugin-core/`
- **Example Plugins**: `inference-gollek/plugins/`

### Community
- **GitHub**: https://github.com/gollek-ai
- **Discussions**: https://github.com/gollek-ai/gollek/discussions
- **Issues**: https://github.com/gollek-ai/gollek/issues

---

**Last Updated**: March 2026
**Version**: 2.0.0
**Status**: ✅ Production Ready
