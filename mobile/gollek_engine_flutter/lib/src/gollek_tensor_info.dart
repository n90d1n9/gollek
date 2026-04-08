/// Tensor information descriptor.
class GollekTensorInfo {
  /// Tensor name.
  final String name;
  
  /// Tensor data type.
  final int type;
  
  /// Tensor dimensions.
  final List<int> dims;
  
  /// Total size in bytes.
  final int byteSize;
  
  /// Quantization scale (0.0 if none).
  final double scale;
  
  /// Quantization zero point (0 if none).
  final int zeroPoint;

  const GollekTensorInfo({
    required this.name,
    required this.type,
    required this.dims,
    required this.byteSize,
    required this.scale,
    required this.zeroPoint,
  });

  factory GollekTensorInfo.fromMap(Map<String, dynamic> map) {
    return GollekTensorInfo(
      name: map['name'] as String? ?? '',
      type: map['type'] as int? ?? 0,
      dims: (map['dims'] as List?)?.map((e) => e as int).toList() ?? [],
      byteSize: map['byteSize'] as int? ?? 0,
      scale: (map['scale'] as num?)?.toDouble() ?? 0.0,
      zeroPoint: map['zeroPoint'] as int? ?? 0,
    );
  }

  @override
  String toString() {
    return 'GollekTensorInfo{name: $name, dims: $dims, byteSize: $byteSize}';
  }
}
