# Gollek LiteRT Flutter Plugin

High-performance LiteRT inference engine for Flutter with multiplatform support for iOS and macOS. **Pure Swift implementation** - no Objective-C needed!

## Features

- ✅ **Pure Swift**: No Objective-C, clean Swift codebase with C interop
- ✅ **Multiplatform Support**: iOS (13.0+) and macOS (10.14+)
- ✅ **Batched Inference**: Run multiple inputs in a single batch for improved throughput
- ✅ **Streaming Inference**: Auto-regressive decoding for LLMs and sequential models
- ✅ **Model Caching**: LRU cache with memory quota management
- ✅ **Performance Metrics**: Latency percentiles, memory usage, throughput tracking
- ✅ **Hardware Acceleration**: CPU (XNNPACK), GPU (Metal), CoreML, NNAPI
- ✅ **Async/Await APIs**: Modern Swift and Dart concurrency support
- ✅ **Thread Safety**: All inference operations are thread-safe

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flutter Application                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GollekEngine │  │   Batching   │  │   Streaming      │  │
│  │              │  │  Manager     │  │   Session        │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                  │                    │            │
│  ┌──────▼──────────────────▼────────────────────▼─────────┐ │
│  │           Method Channel / Platform Interface           │ │
│  └──────┬──────────────────┬────────────────────┬─────────┘ │
└─────────┼──────────────────┼────────────────────┼───────────┘
          │                  │                    │
┌─────────▼──────────────────▼────────────────────▼───────────┐
│                   Native Platform Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pure Swift Plugin (iOS/macOS)                       │   │
│  │  - GollekEngineFlutterPlugin                         │   │
│  │  - Method channel handlers                           │   │
│  │  - GollekEngine (Swift C wrapper)                    │   │
│  │  - GollekStreamSession                               │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  Swift Extensions (GollekEngine+Extensions.swift)    │   │
│  │  - Async/await APIs                                  │   │
│  │  - GollekModelCache (LRU cache)                      │   │
│  │  - GollekBatchingManager                             │   │
│  │  - Combine publishers                                │   │
│  └─────────────────────┬────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      C++ Core Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  gollek_engine.h (C API)                             │   │
│  │  - gollek_engine_create/destroy                      │   │
│  │  - gollek_load_model_from_file/buffer                │   │
│  │  - gollek_set_input/invoke/get_output                │   │
│  │  - gollek_set_batch_input/get_batch_output           │   │
│  │  - gollek_start_streaming/stream_next/end_streaming  │   │
│  │  - gollek_get_metrics/reset_metrics                  │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  gollek::Engine (C++ class)                          │   │
│  │  - TFLite interpreter management                     │   │
│  │  - Memory pool for zero-GC allocations               │   │
│  │  - Metrics collection with percentiles               │   │
│  │  - Thread-safe invoke with mutex                     │   │
│  └─────────────────────┬────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  TensorFlow Lite / LiteRT                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ CPU (XNNPACK)│  │ GPU (Metal)  │  │ CoreML / NNAPI   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Add to pubspec.yaml

```yaml
dependencies:
  gollek_engine_flutter:
    path: ../gollek_engine_flutter  # Adjust path as needed
```

### 2. iOS Configuration

Add to your iOS Podfile (if not already present):

```ruby
platform :ios, '13.0'

# Ensure C++17 support
post_install do |installer|
  installer.pods_project.targets.each do |target|
    if target.name == 'gollek_engine_flutter'
      target.build_configurations.each do |config|
        config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++17'
        config.build_settings['CLANG_CXX_LIBRARY'] = 'libc++'
      end
    end
  end
end
```

### 3. macOS Configuration

Add to your macOS Podfile:

```ruby
platform :osx, '10.14'
```

## Usage Examples

### Basic Inference

```dart
import 'package:gollek_engine_flutter/gollek_engine_flutter.dart';

void main() async {
  // Create engine with default config
  final engine = await GollekEngine.create(
    config: GollekConfig(
      numThreads: 4,
      delegate: GollekDelegate.auto,
      enableXnnpack: true,
    ),
  );

  // Load model from assets
  await engine.loadModelFromAssets('assets/mobilenet_v2.litertlm');

  // Prepare input tensor (example: 1x224x224x3 float32 image)
  final inputData = Float32List(1 * 224 * 224 * 3).buffer.asUint8List();
  // ... fill inputData with your image pixels ...

  // Run inference
  final outputData = await engine.infer(inputData);

  // Parse output (example: classification scores)
  final scores = Float32List.view(outputData.buffer);
  final topClass = scores.indexOf(scores.reduce((a, b) => a > b ? a : b));
  print('Top class: $topClass with confidence: ${scores[topClass]}');

  // Clean up
  await engine.destroy();
}
```

### Batched Inference

```dart
void batchedInference() async {
  final engine = await GollekEngine.create();
  await engine.loadModelFromAssets('assets/model.litertlm');

  // Prepare multiple inputs
  final inputs = <Uint8List>[
    createInputTensor(0),
    createInputTensor(1),
    createInputTensor(2),
    createInputTensor(3),
  ];

  // Run batch inference
  final outputs = await engine.inferBatch(inputs);

  // Process outputs
  for (int i = 0; i < outputs.length; i++) {
    print('Output $i: ${outputs[i].lengthInBytes} bytes');
  }

  await engine.destroy();
}
```

### Using Batching Manager

```dart
void highThroughputInference() async {
  final engine = await GollekEngine.create();
  await engine.loadModelFromAssets('assets/model.litertlm');

  // Create batching manager
  final batching = GollekBatchingManager(
    engine,
    maxDelay: Duration(milliseconds: 10),
    maxBatchSize: 32,
  );

  // Submit multiple requests
  final futures = <Future<Uint8List>>[];
  for (int i = 0; i < 100; i++) {
    futures.add(batching.submit(createInputTensor(i)));
  }

  // Wait for all results
  final results = await Future.wait(futures);
  print('Received ${results.length} results');

  // Clean up
  batching.dispose();
  await engine.destroy();
}
```

### Streaming Inference (LLMs)

```dart
void streamingInference() async {
  final engine = await GollekEngine.create();
  await engine.loadModelFromAssets('assets/llm_model.litertlm');

  // Create streaming session
  final session = GollekStreamSession(engine);
  
  // Start streaming with prompt
  final promptData = encodePrompt('What is machine learning?');
  await session.start(promptData, maxTokens: 100);

  // Process tokens as they arrive
  await for (final tokenData in session.stream()) {
    final token = decodeToken(tokenData);
    stdout.write(token);
  }

  print('\nStreaming complete');
  await session.close();
  await engine.destroy();
}
```

### Model Caching & Warmup

```dart
void modelCaching() async {
  // Get singleton cache instance
  final cache = GollekModelCache();

  // Register model sizes for better memory management
  cache.registerModelSize('assets/mobilenet_v2.litertlm', 14 * 1024 * 1024); // 14MB
  cache.registerModelSize('assets/resnet50.litertlm', 25 * 1024 * 1024); // 25MB

  // Warm up models with dummy inputs
  final dummyInput = Float32List(1 * 224 * 224 * 3).buffer.asUint8List();
  await cache.warmUp('assets/mobilenet_v2.litertlm', dummyInput);

  // Get cached engines
  final engine1 = await cache.getOrCreate('assets/mobilenet_v2.litertlm');
  final engine2 = await cache.getOrCreate('assets/resnet50.litertlm');

  // Use engines
  final output1 = await engine1.infer(dummyInput);
  final output2 = await engine2.infer(dummyInput);

  // Check cache stats
  print('Cache size: ${cache.cacheSize} models');
  print('Current memory: ${cache.currentBytes / (1024 * 1024)} MB');

  // Clear cache when done
  cache.clear();
}
```

### Performance Metrics

```dart
void monitorPerformance() async {
  final engine = await GollekEngine.create();
  await engine.loadModelFromAssets('assets/model.litertlm');

  // Run some inferences
  for (int i = 0; i < 50; i++) {
    await engine.infer(createInputTensor(i));
  }

  // Get metrics
  final metrics = await engine.getMetrics();
  print('Total inferences: ${metrics.totalInferences}');
  print('Failed inferences: ${metrics.failedInferences}');
  print('Average latency: ${metrics.avgLatencyMs.toStringAsFixed(2)} ms');
  print('P50 latency: ${metrics.p50LatencyMs.toStringAsFixed(2)} ms');
  print('P95 latency: ${metrics.p95LatencyMs.toStringAsFixed(2)} ms');
  print('P99 latency: ${metrics.p99LatencyMs.toStringAsFixed(2)} ms');
  print('Peak memory: ${metrics.peakMemoryBytes / (1024 * 1024)} MB');
  print('Current memory: ${metrics.currentMemoryBytes / (1024 * 1024)} MB');

  // Reset metrics if needed
  await engine.resetMetrics();

  await engine.destroy();
}
```

### Tensor Introspection

```dart
void inspectModel() async {
  final engine = await GollekEngine.create();
  await engine.loadModelFromAssets('assets/model.litertlm');

  // Get input count
  final inputCount = await engine.getInputCount();
  print('Model has $inputCount input(s)');

  // Get input tensor info
  for (int i = 0; i < inputCount; i++) {
    final info = await engine.getInputInfo(i);
    print('Input $i:');
    print('  Name: ${info.name}');
    print('  Type: ${info.type}');
    print('  Dims: ${info.dims}');
    print('  Size: ${info.byteSize} bytes');
    print('  Scale: ${info.scale}');
    print('  Zero point: ${info.zeroPoint}');
  }

  // Get output tensor info
  final outputCount = await engine.getOutputCount();
  for (int i = 0; i < outputCount; i++) {
    final info = await engine.getOutputInfo(i);
    print('Output $i: ${info.name}, dims: ${info.dims}');
  }

  await engine.destroy();
}
```

## Configuration Options

### GollekConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numThreads` | int | 4 | Number of CPU threads for inference |
| `delegate` | GollekDelegate | auto | Hardware acceleration preference |
| `enableXnnpack` | bool | true | Enable XNNPACK CPU optimizations |
| `useMemoryPool` | bool | true | Use pre-allocated memory pool |
| `poolSizeBytes` | int | 0 (16MB) | Memory pool size (0 = default) |

### GollekDelegate

| Value | Description |
|-------|-------------|
| `none` | No acceleration, basic CPU |
| `cpu` | XNNPACK CPU optimizations |
| `gpu` | Metal GPU acceleration (iOS/macOS) |
| `nnapi` | Android NNAPI (not used on iOS/macOS) |
| `coreml` | Apple CoreML (iOS only) |
| `auto` | Automatically select best delegate |

## Performance Tips

1. **Use Batching**: For high-throughput scenarios, use `GollekBatchingManager` to batch requests
2. **Warm Up Models**: Run a dummy inference to load and optimize the model before real usage
3. **Cache Models**: Use `GollekModelCache` to avoid reloading models
4. **Monitor Metrics**: Track latency percentiles to detect performance regressions
5. **Choose Right Delegate**: Use `auto` for automatic selection, or explicitly set based on your needs
6. **Memory Pool**: Enable memory pool for faster allocations and less GC pressure

## Native Swift Usage (iOS/macOS)

If you need to use the engine directly in Swift without Flutter:

```swift
import gollek_engine_flutter

// Create engine
let engine = try GollekEngine(config: GollekConfig())

// Load model
try engine.loadModel(from: "/path/to/model.litertlm")

// Single inference
let output = try await engine.inferAsync(input: inputData)

// Batch inference
let outputs = try await engine.inferBatchAsync(inputs: [input1, input2, input3])

// Streaming
for try await chunk in engine.streamInference(input: promptData) {
    processToken(chunk)
}

// Metrics
if let metrics = engine.metrics {
    print("Avg latency: \(metrics.avgLatencyMs) ms")
    print("P95 latency: \(metrics.p95LatencyMs) ms")
}
```

## Building from Source

### Prerequisites

- Flutter SDK 3.x
- Xcode 15+ (for iOS/macOS)
- CocoaPods
- CMake 3.22+

### Build Steps

```bash
# Navigate to plugin directory
cd gollek/mobile/gollek_engine_flutter

# Get Flutter dependencies
flutter pub get

# iOS
cd ios
pod install
cd ..

# macOS
cd macos
pod install
cd ..

# Run example app
cd example
flutter run
```

## API Reference

### GollekEngine

| Method | Description |
|--------|-------------|
| `create()` | Create new engine instance |
| `loadModel()` | Load model from file path |
| `loadModelFromAssets()` | Load model from Flutter assets |
| `infer()` | Run single inference |
| `inferBatch()` | Run batched inference |
| `startStreaming()` | Start streaming session |
| `streamNext()` | Get next streaming chunk |
| `endStreaming()` | End streaming session |
| `getInputInfo()` | Get input tensor info |
| `getOutputInfo()` | Get output tensor info |
| `getMetrics()` | Get performance metrics |
| `resetMetrics()` | Reset metrics counters |
| `destroy()` | Destroy engine and free resources |

### GollekBatchingManager

| Method | Description |
|--------|-------------|
| `submit()` | Submit input for batched processing |
| `dispose()` | Dispose manager and cancel pending requests |

### GollekStreamSession

| Method | Description |
|--------|-------------|
| `start()` | Start streaming with input |
| `next()` | Get next chunk (returns null when done) |
| `stream()` | Get async stream of chunks |
| `close()` | Close session and free resources |

### GollekModelCache

| Method | Description |
|--------|-------------|
| `getOrCreate()` | Get or create cached engine |
| `warmUp()` | Warm up model with dummy input |
| `evict()` | Evict specific model from cache |
| `clear()` | Clear all cached models |

## Troubleshooting

### Model Loading Fails

- Verify model file exists in assets
- Check model path is correct
- Ensure model is valid TFLite flatbuffer
- Check file permissions

### Inference Fails

- Verify input tensor size matches model input
- Check output buffer is large enough
- Ensure model is loaded before inference
- Check delegate compatibility

### Performance Issues

- Enable XNNPACK in config
- Use batching for high-throughput scenarios
- Monitor metrics to identify bottlenecks
- Consider using GPU or CoreML delegates
- Warm up models before production use

## License

MIT License - Copyright (c) 2026 Kayys.tech

See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Support

- GitHub Issues: https://github.com/wayang-platform/gollek/issues
- Documentation: https://wayang-platform.github.io/gollek
- Email: team@wayang.dev
