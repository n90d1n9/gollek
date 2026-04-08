/// Performance metrics from the inference engine.
class GollekMetrics {
  /// Total number of inferences run.
  final int totalInferences;
  
  /// Number of failed inferences.
  final int failedInferences;
  
  /// Average latency in milliseconds.
  final double avgLatencyMs;
  
  /// 50th percentile latency in milliseconds.
  final double p50LatencyMs;
  
  /// 95th percentile latency in milliseconds.
  final double p95LatencyMs;
  
  /// 99th percentile latency in milliseconds.
  final double p99LatencyMs;
  
  /// Peak memory usage in bytes.
  final int peakMemoryBytes;
  
  /// Current memory usage in bytes.
  final int currentMemoryBytes;
  
  /// Active delegate being used.
  final int activeDelegate;

  const GollekMetrics({
    required this.totalInferences,
    required this.failedInferences,
    required this.avgLatencyMs,
    required this.p50LatencyMs,
    required this.p95LatencyMs,
    required this.p99LatencyMs,
    required this.peakMemoryBytes,
    required this.currentMemoryBytes,
    required this.activeDelegate,
  });

  factory GollekMetrics.fromMap(Map<String, dynamic> map) {
    return GollekMetrics(
      totalInferences: map['totalInferences'] as int? ?? 0,
      failedInferences: map['failedInferences'] as int? ?? 0,
      avgLatencyMs: (map['avgLatencyMs'] as num?)?.toDouble() ?? 0.0,
      p50LatencyMs: (map['p50LatencyMs'] as num?)?.toDouble() ?? 0.0,
      p95LatencyMs: (map['p95LatencyMs'] as num?)?.toDouble() ?? 0.0,
      p99LatencyMs: (map['p99LatencyMs'] as num?)?.toDouble() ?? 0.0,
      peakMemoryBytes: map['peakMemoryBytes'] as int? ?? 0,
      currentMemoryBytes: map['currentMemoryBytes'] as int? ?? 0,
      activeDelegate: map['activeDelegate'] as int? ?? 0,
    );
  }

  @override
  String toString() {
    return 'GollekMetrics{avgLatencyMs: ${avgLatencyMs.toStringAsFixed(2)}ms, '
        'p95LatencyMs: ${p95LatencyMs.toStringAsFixed(2)}ms, '
        'totalInferences: $totalInferences}';
  }
}
