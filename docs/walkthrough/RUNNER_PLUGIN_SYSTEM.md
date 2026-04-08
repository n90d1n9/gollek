# Runner Plugin System

Model format runner plugins for flexible, modular AI inference support.

## Overview

The Runner Plugin system provides a modular architecture for supporting different model formats, enabling:

- **Hot-Reload**: Add/remove model format support without restarting
- **Selective Deployment**: Include only needed runners in your deployment
- **Performance**: Optimized runners for specific formats
- **Extensibility**: Create custom runners for proprietary formats

### Available Runners

| Runner | Format | Backend | CPU | GPU | Quantization |
|--------|--------|---------|-----|-----|--------------|
| **GGUF** | .gguf | llama.cpp | ✅ | ✅ | ✅ |
| **ONNX** | .onnx | ONNX Runtime | ✅ | ✅ | ✅ |
| **TensorRT** | .engine/.plan | TensorRT | ❌ | ✅ | ✅ |
| **LibTorch** | .pt/.bin | PyTorch | ✅ | ✅ | ✅ |
| **TFLite** | .litertlm | TensorFlow Lite | ✅ | ✅ | ✅ |
| **Safetensors** | .safetensors | Custom | ✅ | ✅ | ❌ |

---

## Quick Start

### Installation

#### Step 1: Build Runner Core

```bash
cd inference-gollek/core/gollek-plugin-runner-core
mvn clean install
```

#### Step 2: Build Runner Plugins

```bash
# Build GGUF runner
cd inference-gollek/plugins/gollek-plugin-runner-gguf
mvn clean install -Pinstall-plugin

# Build ONNX runner
cd ../gollek-plugin-runner-onnx
mvn clean install -Pinstall-plugin

# Build all runners
cd inference-gollek/plugins
mvn clean install -Pinstall-plugin
```

### Configuration

Create `~/.gollek/plugins/runners/runner-config.json`:

```json
{
  "gguf-runner": {
    "enabled": true,
    "n_gpu_layers": -1,
    "n_ctx": 4096,
    "n_batch": 512,
    "flash_attn": true
  },
  "onnx-runner": {
    "enabled": true,
    "execution_provider": "CUDAExecutionProvider",
    "intra_op_num_threads": 4
  },
  "tensorrt-runner": {
    "enabled": true,
    "max_workspace_size": 4294967296,
    "fp16_mode": true
  }
}
```

### Usage Example

```java
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;
import tech.kayys.gollek.plugin.runner.RunnerSession;

// Get manager instance
RunnerPluginManager manager = RunnerPluginManager.getInstance();

// Register runners
manager.register(new GGUFRunnerPlugin());
manager.register(new OnnxRunnerPlugin());

// Initialize with configuration
Map<String, Object> config = loadConfig();
manager.initialize(config);

// Create session for model
String modelPath = "models/llama-3-8b.gguf";
Optional<RunnerSession> sessionOpt = manager.createSession(modelPath, Map.of());

if (sessionOpt.isPresent()) {
    RunnerSession session = sessionOpt.get();
    
    // Execute inference
    InferenceRequest request = createRequest();
    InferenceResponse response = session.infer(request).await().atMost(Duration.ofSeconds(30));
    
    System.out.println(response.getContent());
    
    // Or streaming
    session.stream(request)
        .subscribe().with(chunk -> System.out.print(chunk.getDelta()));
    
    // Close session when done
    manager.closeSession(session.getSessionId());
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer                          │
│  InferenceRequest → Runner Selection                    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           RunnerPluginManager                           │
│  - Plugin discovery                                     │
│  - Format-based routing                                 │
│  - Session management                                   │
│  - Lifecycle management                                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Runner Plugins                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  GGUF    │ │   ONNX   │ │TensorRT  │ │TFLite    │  │
│  │ (llama)  │ │ (onnx)   │ │  (TRT)   │ │  (TF)    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Native Backends                               │
│  libllama.so | libonnxruntime.so | libnvinfer.so       │
└─────────────────────────────────────────────────────────┘
```

---

## Creating a Custom Runner Plugin

### Step 1: Implement RunnerPlugin Interface

```java
package com.example.plugin;

import tech.kayys.gollek.plugin.runner.*;
import java.util.*;

public class CustomRunnerPlugin implements RunnerPlugin {
    
    @Override
    public String id() {
        return "custom-runner";
    }
    
    @Override
    public String name() {
        return "Custom Format Runner";
    }
    
    @Override
    public String description() {
        return "Support for custom model format";
    }
    
    @Override
    public Set<String> supportedFormats() {
        return Set.of(".custom", ".mdl");
    }
    
    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("transformer", "cnn", "rnn");
    }
    
    @Override
    public boolean supportsModel(String modelPath) {
        return modelPath.toLowerCase().endsWith(".custom");
    }
    
    @Override
    public boolean isAvailable() {
        // Check if native library is loaded
        try {
            System.loadLibrary("custom-inference");
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }
    
    @Override
    public RunnerSession createSession(String modelPath, Map<String, Object> config) {
        return new CustomRunnerSession(modelPath, config);
    }
}
```

### Step 2: Implement RunnerSession

```java
public class CustomRunnerSession implements RunnerSession {
    
    private final String sessionId;
    private final String modelPath;
    private final Map<String, Object> config;
    
    public CustomRunnerSession(String modelPath, Map<String, Object> config) {
        this.sessionId = UUID.randomUUID().toString();
        this.modelPath = modelPath;
        this.config = config;
        
        // Load model
        loadModel(modelPath);
    }
    
    @Override
    public String getSessionId() {
        return sessionId;
    }
    
    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        // Execute inference
        return Uni.createFrom().item(executeInference(request));
    }
    
    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        // Stream inference
        return executeStreaming(request);
    }
    
    @Override
    public void close() {
        // Release resources
        unloadModel();
    }
    
    // ... other methods
}
```

### Step 3: Create pom.xml

```xml
<project>
    <groupId>com.example</groupId>
    <artifactId>gollek-plugin-runner-custom</artifactId>
    <version>1.0.0</version>
    
    <dependencies>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-plugin-runner-core</artifactId>
            <version>1.0.0-SNAPSHOT</version>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <configuration>
                    <archive>
                        <manifestEntries>
                            <Plugin-Id>custom-runner</Plugin-Id>
                            <Plugin-Class>com.example.plugin.CustomRunnerPlugin</Plugin-Class>
                            <Plugin-Type>runner</Plugin-Type>
                            <Supported-Formats>.custom,.mdl</Supported-Formats>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

### Step 4: Build and Deploy

```bash
mvn clean package
cp target/gollek-plugin-runner-custom-1.0.0.jar ~/.gollek/plugins/runners/
```

---

## Configuration Reference

### GGUF Runner

```json
{
  "gguf-runner": {
    "enabled": true,
    "n_gpu_layers": -1,
    "n_ctx": 4096,
    "n_batch": 512,
    "n_threads": 8,
    "flash_attn": true,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "lora": [
      {
        "path": "lora-adapter.gguf",
        "scale": 1.0
      }
    ]
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable runner |
| `n_gpu_layers` | `-1` | Layers to offload to GPU (-1 = all) |
| `n_ctx` | `4096` | Context size |
| `n_batch` | `512` | Batch size |
| `n_threads` | `CPU count` | Number of threads |
| `flash_attn` | `true` | Enable Flash Attention |
| `cache_type_k` | `f16` | KV cache type for K |
| `cache_type_v` | `f16` | KV cache type for V |

### ONNX Runner

```json
{
  "onnx-runner": {
    "enabled": true,
    "execution_provider": "CUDAExecutionProvider",
    "intra_op_num_threads": 4,
    "inter_op_num_threads": 2,
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 4294967296
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable runner |
| `execution_provider` | `CPUExecutionProvider` | EP to use |
| `intra_op_num_threads` | `0` | Intra-op threads |
| `inter_op_num_threads` | `0` | Inter-op threads |
| `gpu_mem_limit` | `Max` | GPU memory limit |

### TensorRT Runner

```json
{
  "tensorrt-runner": {
    "enabled": true,
    "max_workspace_size": 4294967296,
    "fp16_mode": true,
    "int8_mode": false,
    "max_batch_size": 32,
    "timing_cache_size": 1073741824
  }
}
```

---

## Performance Tuning

### GGUF Optimization

```json
{
  "gguf-runner": {
    "n_gpu_layers": -1,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "defrag_thold": 0.1
  }
}
```

### Multi-Runner Configuration

```json
{
  "gguf-runner": {
    "enabled": true,
    "priority": 100
  },
  "onnx-runner": {
    "enabled": true,
    "priority": 50
  },
  "tensorrt-runner": {
    "enabled": true,
    "priority": 75
  }
}
```

---

## Troubleshooting

### Runner Not Loading

```bash
# Check JAR is in correct directory
ls -la ~/.gollek/plugins/runners/*.jar

# Check manifest
unzip -p gollek-plugin-runner-gguf.jar META-INF/MANIFEST.MF

# Check logs
tail -f ~/.gollek/logs/gollek.log | grep runner
```

### Native Library Not Found

```bash
# Set library path
export LD_LIBRARY_PATH=/path/to/native/libs:$LD_LIBRARY_PATH

# Or copy to system directory
sudo cp libllama.so /usr/local/lib/
sudo ldconfig
```

### Model Format Not Supported

```java
// Check which runner supports the model
RunnerPluginManager manager = RunnerPluginManager.getInstance();
Optional<RunnerPlugin> plugin = manager.findPluginForModel("model.gguf");

if (plugin.isPresent()) {
    System.out.println("Supported by: " + plugin.get().name());
} else {
    System.out.println("No runner supports this model format");
}
```

---

## Resources

- [Runner Plugin Core](inference-gollek/core/gollek-plugin-runner-core/README.md)
- [GGUF Runner](inference-gollek/plugins/gollek-plugin-runner-gguf/README.md)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

---

[Back to Optimization Plugins](/docs/optimization-plugins) &nbsp; [View Architecture](/docs/architecture)
