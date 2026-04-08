/// Gollek engine configuration options.
class GollekConfig {
  /// Number of CPU threads (0 = auto).
  final int numThreads;
  
  /// Delegate preference (0=NONE, 1=CPU, 2=GPU, 3=NNAPI, 5=CoreML, 6=AUTO).
  final int delegate;
  
  /// Enable XNNPACK optimization.
  final bool enableXnnpack;
  
  /// Use memory pool for faster allocations.
  final bool useMemoryPool;
  
  /// Memory pool size in bytes (0 = default 16MB).
  final int poolSizeBytes;

  const GollekConfig({
    this.numThreads = 4,
    this.delegate = 6, // AUTO
    this.enableXnnpack = true,
    this.useMemoryPool = true,
    this.poolSizeBytes = 0,
  });

  Map<String, dynamic> toMap() {
    return {
      'numThreads': numThreads,
      'delegate': delegate,
      'enableXnnpack': enableXnnpack,
      'useMemoryPool': useMemoryPool,
      'poolSizeBytes': poolSizeBytes,
    };
  }
}

/// Delegate types for hardware acceleration.
enum GollekDelegate {
  none(0),
  cpu(1),
  gpu(2),
  nnapi(3),
  hexagon(4),
  coreml(5),
  auto(6);

  final int value;
  const GollekDelegate(this.value);
  
  static GollekDelegate fromValue(int value) {
    return GollekDelegate.values.firstWhere(
      (e) => e.value == value,
      orElse: () => GollekDelegate.auto,
    );
  }
}

/// Tensor data type.
enum GollekTensorType {
  float32(0),
  float16(1),
  int32(2),
  uint8(3),
  int64(4),
  int8(6),
  bool(9);

  final int value;
  const GollekTensorType(this.value);
  
  static GollekTensorType fromValue(int value) {
    return GollekTensorType.values.firstWhere(
      (e) => e.value == value,
      orElse: () => GollekTensorType.float32,
    );
  }
}
