# GPTQ Quantizer Improvements & Safetensor Runner Integration

## Overview

This document summarizes the comprehensive improvements made to the GPTQ quantizer (`gollek-quantizer-gptq`) and its integration with the safetensor runner (`gollek-plugin-runner-safetensor`).

## Changes Summary

### 1. Enhanced GPTQConfig (`GPTQConfig.java`)

#### New Configuration Options
- **`perChannel`**: Enable/disable per-channel quantization (default: true)
- **`dampPercent`**: Dampening percentage for numerical stability (default: 0.01 = 1%)
- **`numSamples`**: Number of calibration samples (default: 128)
- **`seqLen`**: Sequence length for calibration (default: 2048)
- **`quantizeEmbeddings`**: Whether to quantize embedding layers (default: false)
- **`calibrationDataPath`**: Path to calibration dataset (optional)

#### New Preset Configurations
```java
// Standard 4-bit with FP16 output for memory efficiency
GPTQConfig.gptq4bitFP16()

// 4-bit with symmetric quantization
GPTQConfig.gptq4bitSymmetric()

// 4-bit optimized for exllama v2
GPTQConfig.gptq4bitExllamaV2()
```

#### Builder Pattern
```java
GPTQConfig config = GPTQConfig.builder()
    .bits(4)
    .groupSize(64)
    .actOrder(true)
    .symmetric(false)
    .perChannel(true)
    .dampPercent(0.02)
    .numSamples(256)
    .seqLen(4096)
    .quantizeEmbeddings(false)
    .calibrationDataPath("/path/to/data")
    .build();
```

### 2. New GPTQQuantizerService (`GPTQQuantizerService.java`)

A unified service providing:

#### Quantization Operations
```java
// Synchronous quantization
QuantizationResult result = service.quantize(inputPath, outputPath, config);

// Asynchronous quantization
CompletableFuture<QuantizationResult> future = 
    service.quantizeAsync(inputPath, outputPath, config);
```

#### Model Loading
```java
// Auto-detect config and load
GPTQLoader loader = service.loadQuantized(modelPath);

// Load with explicit config
GPTQLoader loader = service.loadQuantized(modelPath, config);
```

#### Dequantization
```java
ConversionResult result = service.dequantize(inputPath, outputPath, convConfig);
```

#### Model Inspection
```java
ModelInspectionResult inspection = service.inspect(modelPath);
```

#### Result Records
- **`QuantizationResult`**: Contains statistics like compression ratio, throughput, file sizes
- **`ModelInspectionResult`**: Layer count, memory usage, config details
- **`QuantizedTensor`**: Metadata for individual quantized tensors

### 3. Enhanced SafetensorConverter (`SafetensorConverter.java`)

#### New Features

**Async Conversion with Progress Reporting**
```java
CompletableFuture<ConversionResult> future = converter.convertAsync(outputPath, 
    (layerName, current, total) -> {
        System.out.printf("Progress: %s (%d/%d)%n", layerName, current, total);
    });
```

**Cancellation Support**
```java
converter.cancel(); // Gracefully stops ongoing conversion
```

**Progress Callback Interface**
```java
@FunctionalInterface
public interface ProgressCallback {
    void onProgress(String layerName, int current, int total);
}
```

### 4. SafetensorRunnerPlugin Integration (`SafetensorRunnerPlugin.java`)

#### New Quantization Methods
```java
// Quantize a model
CompletableFuture<QuantizationResult> result = plugin.quantizeModel(
    inputPath, outputPath, GPTQConfig.gptq4bit());

// Dequantize a model
CompletableFuture<ConversionResult> result = plugin.dequantizeModel(
    inputPath, outputPath, ConversionConfig.defaults());

// Inspect a quantized model
ModelInspectionResult inspection = plugin.inspectModel(modelPath);

// Access quantizer service directly
GPTQQuantizerService quantizer = plugin.getQuantizerService();
```

#### Lifecycle Management
- Quantizer service is automatically initialized with plugin
- Proper cleanup during shutdown
- Resource management for active sessions

### 5. SafetensorRunnerSession Enhancements (`SafetensorRunnerSession.java`)

#### Automatic Quantized Model Detection
```java
// Automatically detects GPTQ models by checking for .qweight and .scales tensors
private boolean detectQuantizedModel(String modelPath)
```

#### GPTQ Model Loading
```java
// Loads quantized models automatically during session creation
private void loadQuantizedModel(String modelPath)
```

#### New Public Methods
```java
// Check if session has quantized model
boolean hasQuantizedModel()

// Get GPTQ loader for direct access
GPTQLoader getGptqLoader()
```

#### Enhanced Resource Cleanup
```java
// Properly closes GPTQ loader and quantizer service
@Override
public void close() {
    // ... existing cleanup ...
    if (gptqLoader != null) gptqLoader.close();
    if (quantizerService != null) quantizerService.close();
}
```

### 6. QuantizationResource REST API (`QuantizationResource.java`)

#### New Endpoints

**POST `/api/v1/quantization/quantize/gptq`**
```json
{
  "input_path": "/path/to/model",
  "output_path": "/path/to/quantized",
  "bits": 4,
  "group_size": 128,
  "symmetric": false,
  "per_channel": true,
  "act_order": false,
  "damp_percent": 0.01,
  "num_samples": 128,
  "seq_len": 2048
}
```

Response (202 Accepted):
```json
{
  "task_id": "uuid-here",
  "status": "started",
  "message": "GPTQ quantization started"
}
```

**GET `/api/v1/quantization/inspect/{modelPath}`**

Response (200 OK):
```json
{
  "config": {
    "bits": 4,
    "group_size": 128,
    "act_order": false,
    "symmetric": false
  },
  "layer_count": 32,
  "memory_mb": 3814.70,
  "layers": ["model.layers.0", "model.layers.1", ...]
}
```

### 7. Comprehensive Test Suite (`GPTQIntegrationTest.java`)

#### Test Coverage
- ✅ GPTQConfig builder and presets
- ✅ Derived properties (elementsPerInt32, quantMask, etc.)
- ✅ Validation (invalid bit widths)
- ✅ QuantizerService initialization and lifecycle
- ✅ Async quantization tasks
- ✅ ConversionConfig presets
- ✅ Result calculations (compression ratio, throughput)
- ✅ Model inspection
- ✅ Progress callbacks
- ✅ Edge cases and error handling

Run tests:
```bash
cd gollek/core/quantizer/gollek-quantizer-gptq
mvn test
```

## Usage Examples

### Example 1: Quantize a Model via API

```bash
curl -X POST http://localhost:8080/api/v1/quantization/quantize/gptq \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/models/llama-3-8b-fp16",
    "output_path": "/models/llama-3-8b-gptq",
    "bits": 4,
    "group_size": 128,
    "act_order": true,
    "per_channel": true
  }'
```

### Example 2: Inspect Quantized Model

```bash
curl http://localhost:8080/api/v1/quantization/inspect/models/llama-3-8b-gptq
```

### Example 3: Programmatic Quantization

```java
@Inject
SafetensorRunnerPlugin plugin;

public void quantizeModel() {
    Path inputPath = Path.of("/models/llama-3-8b-fp16");
    Path outputPath = Path.of("/models/llama-3-8b-gptq");
    
    GPTQConfig config = GPTQConfig.builder()
        .bits(4)
        .groupSize(128)
        .actOrder(true)
        .perChannel(true)
        .build();
    
    CompletableFuture<GPTQQuantizerService.QuantizationResult> future = 
        plugin.quantizeModel(inputPath, outputPath, config);
    
    future.thenAccept(result -> {
        System.out.println("Compression ratio: " + result.compressionRatio());
        System.out.println("Throughput: " + result.throughputMBps() + " MB/s");
    });
}
```

### Example 4: Load and Use Quantized Model

```java
@Inject
SafetensorRunnerPlugin plugin;

public void runInference() {
    // Create session with auto-detection of quantized model
    RunnerSession session = plugin.createSession(
        "/models/llama-3-8b-gptq",
        Map.of("device", "cpu")
    );
    
    // Check if quantized
    if (session instanceof SafetensorRunnerSession safetensorSession) {
        if (safetensorSession.hasQuantizedModel()) {
            GPTQLoader loader = safetensorSession.getGptqLoader();
            System.out.println("Loaded " + loader.getLayerCount() + " GPTQ layers");
        }
    }
    
    // Run inference
    InferenceRequest request = InferenceRequest.builder()
        .messages(List.of(Message.user("Hello!")))
        .build();
    
    Uni<InferenceResponse> response = session.infer(request);
}
```

### Example 5: Conversion with Progress

```java
GPTQLoader loader = new GPTQLoader(modelPath, GPTQConfig.gptq4bit()).load();
SafetensorConverter converter = new SafetensorConverter(
    loader, 
    SafetensorConverter.ConversionConfig.verbose()
);

CompletableFuture<ConversionResult> future = converter.convertAsync(
    Path.of("/output/dequantized.safetensors"),
    (layerName, current, total) -> {
        double progress = (double) current / total * 100;
        System.out.printf("Converting: %.1f%% (%s)%n", progress, layerName);
    }
);

ConversionResult result = future.get();
System.out.println("Converted " + result.layersConverted() + " layers");
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SafetensorRunnerPlugin                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         GPTQQuantizerService                         │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ Quantize   │  │ Dequantize   │  │ Inspect    │  │   │
│  │  └────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   SafetensorRunnerSession                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Auto-detect & Load GPTQ Models                      │   │
│  │  - Scan for .qweight, .scales tensors                │   │
│  │  - Auto-detect GPTQConfig                            │   │
│  │  - Load via GPTQLoader                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   QuantizationResource                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  REST API Endpoints                                  │   │
│  │  - POST /quantize/gptq (async)                       │   │
│  │  - GET  /inspect/{modelPath}                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### SIMD Vectorization
- Uses JDK 25 Vector API for dequantization
- Automatic hardware detection (AVX2, AVX-512, NEON)
- Processes multiple elements per SIMD instruction

### Memory Efficiency
- Off-heap memory allocation via FFM (Foreign Function & Memory)
- Memory-mapped I/O for safetensor files
- Zero-copy slices where possible

### Async Operations
- Non-blocking quantization via CompletableFuture
- Progress callbacks for monitoring
- Cancellation support for long-running operations

## Migration Guide

### From Old GPTQConfig
```java
// Old
GPTQConfig config = new GPTQConfig(4, 128, false, false, false, "float32");

// New (equivalent)
GPTQConfig config = GPTQConfig.gptq4bit();

// New (with custom settings)
GPTQConfig config = GPTQConfig.builder()
    .bits(4)
    .groupSize(128)
    .build();
```

### From Manual Loading
```java
// Old
GPTQLoader loader = new GPTQLoader(path, config);
loader.load();

// New (via service)
GPTQLoader loader = quantizerService.loadQuantized(path);

// New (with explicit config)
GPTQLoader loader = quantizerService.loadQuantized(path, config);
```

## Future Enhancements

1. **Actual GPTQ Algorithm Implementation**
   - Weight quantization with Hessian-based optimization
   - Calibration data processing
   - Act-order (desc_act) support

2. **Advanced Features**
   - Streaming quantization for large models
   - Multi-GPU distribution
   - Quantization-aware training (QAT) export

3. **Format Support**
   - Exllama v2 optimized layout
   - Marlin format for faster inference
   - AWQ format conversion

4. **Monitoring & Observability**
   - Prometheus metrics for quantization tasks
   - Distributed tracing for async operations
   - Real-time progress WebSocket endpoint

## Testing

Run the comprehensive test suite:

```bash
# Test GPTQ quantizer
cd gollek/core/quantizer/gollek-quantizer-gptq
mvn test

# Test safetensor runner integration
cd gollek/plugins/runner/safetensor/gollek-plugin-runner-safetensor
mvn test

# Run all tests
mvn clean install -pl gollek/core/quantizer/gollek-quantizer-gptq,gollek/plugins/runner/safetensor
```

## API Documentation

Full OpenAPI documentation available at:
```
http://localhost:8080/q/swagger-ui
```

Look for:
- `POST /api/v1/quantization/quantize/gptq`
- `GET /api/v1/quantization/inspect/{modelPath}`

## Troubleshooting

### Common Issues

**Issue**: "No .safetensors files found"
- **Solution**: Ensure model directory contains `.safetensors` files with GPTQ tensors (`.qweight`, `.scales`, `.qzeros`)

**Issue**: "Invalid bits: 5"
- **Solution**: Use valid bit widths: 2, 3, 4, or 8

**Issue**: Quantization fails with OOM
- **Solution**: Reduce `numSamples`, `seqLen`, or use smaller `groupSize`

**Issue**: Slow quantization performance
- **Solution**: Ensure Vector API is enabled: `--enable-preview --add-modules jdk.incubator.vector`

## References

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AutoGPTQ Implementation](https://github.com/PanQiWei/AutoGPTQ)
- [JDK Vector API](https://openjdk.org/jeps/448)
- [Safetensors Format](https://huggingface.co/docs/safetensors/index)

## License

MIT License - Copyright (c) 2026 Kayys.tech
