import 'dart:collection';

import 'gollek_config.dart';
import 'gollek_engine.dart';

/// LRU cache for loaded models with memory quota management.
///
/// Pre-load and warm up models to reduce first-inference latency.
///
/// Example usage:
/// ```dart
/// final cache = GollekModelCache();
///
/// // Warm up model
/// await cache.warmUp('assets/model.litertlm', dummyInput);
///
/// // Get cached engine
/// final engine = await cache.getOrCreate('assets/model.litertlm');
///
/// // Use engine
/// final output = await engine.infer(inputData);
///
/// // Clear cache when done
/// cache.clear();
/// ```
class GollekModelCache {
  static final GollekModelCache _instance = GollekModelCache._internal();

  factory GollekModelCache() => _instance;
  GollekModelCache._internal();

  final Map<String, _CacheEntry> _cache = LinkedHashMap();
  final Map<String, Future<GollekEngine>> _pendingLoads = HashMap();
  final Map<String, int> _modelSizes = {};

  int _currentBytes = 0;
  final int maxBytes;

  /// Create cache with optional memory quota (default 512MB).
  GollekModelCache.withQuota({this.maxBytes = 512 * 1024 * 1024});

  /// Get or create a cached engine for the given model.
  Future<GollekEngine> getOrCreate(
    String modelPath, {
    GollekConfig config = const GollekConfig(),
  }) async {
    // Check cache first
    if (_cache.containsKey(modelPath)) {
      final entry = _cache.remove(modelPath)!;
      entry.lastAccessed = DateTime.now();
      _cache[modelPath] = entry;
      return entry.engine;
    }

    // Check if already loading
    if (_pendingLoads.containsKey(modelPath)) {
      return await _pendingLoads[modelPath]!;
    }

    // Load new model
    final future = GollekEngine.create(config: config)
        .then((engine) async {
          await engine.loadModelFromAssets(modelPath);

          // Estimate model size (rough approximation)
          final approxSize = _modelSizes[modelPath] ?? 0;

          // Evict if needed
          while (_currentBytes + approxSize > maxBytes && _cache.isNotEmpty) {
            _evictOldest();
          }

          _cache[modelPath] = _CacheEntry(engine, approxSize);
          _currentBytes += approxSize;
          _pendingLoads.remove(modelPath);

          return engine;
        })
        .catchError((e) {
          _pendingLoads.remove(modelPath);
          throw e;
        });

    _pendingLoads[modelPath] = future;
    return future;
  }

  /// Warm up a model by running a dummy inference.
  Future<void> warmUp(
    String modelPath,
    Uint8List dummyInput, {
    GollekConfig config = const GollekConfig(),
  }) async {
    final engine = await getOrCreate(modelPath, config: config);
    await engine.infer(dummyInput);
  }

  /// Register model size for better cache management.
  void registerModelSize(String modelPath, int sizeBytes) {
    _modelSizes[modelPath] = sizeBytes;
  }

  /// Evict a specific model from cache.
  void evict(String modelPath) {
    final entry = _cache.remove(modelPath);
    if (entry != null) {
      entry.engine.destroy();
      _currentBytes -= entry.bytes;
    }
  }

  /// Clear all cached models.
  void clear() {
    for (final entry in _cache.values) {
      entry.engine.destroy();
    }
    _cache.clear();
    _currentBytes = 0;
  }

  /// Get number of cached models.
  int get cacheSize => _cache.length;

  /// Get current memory usage in bytes.
  int get currentBytes => _currentBytes;

  void _evictOldest() {
    if (_cache.isEmpty) return;

    final oldestKey = _cache.keys.first;
    evict(oldestKey);
  }
}

class _CacheEntry {
  final GollekEngine engine;
  final int bytes;
  DateTime lastAccessed;

  _CacheEntry(this.engine, this.bytes) : lastAccessed = DateTime.now();
}
