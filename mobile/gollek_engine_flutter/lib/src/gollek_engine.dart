import 'dart:typed_data';

import 'gollek_config.dart';
import 'gollek_metrics.dart';
import 'gollek_tensor_info.dart';
import '../gollek_engine_flutter_platform_interface.dart';

/// Gollek TFLite Inference Engine
///
/// Provides APIs for model loading, inference, batching, streaming, and metrics.
///
/// Example usage:
/// ```dart
/// final engine = GollekEngine(config: GollekConfig());
/// await engine.loadModelFromAssets('assets/model.litertlm');
/// final output = await engine.infer(inputData);
/// await engine.destroy();
/// ```
class GollekEngine {
  String? _engineId;
  bool _isDestroyed = false;

  /// Create a new inference engine with the given configuration.
  static Future<GollekEngine> create({
    GollekConfig config = const GollekConfig(),
  }) async {
    final engine = GollekEngine._();
    final result = await GollekEngineFlutterPlatform.instance.createEngine(
      config.toMap(),
    );
    engine._engineId = result['engineId'] as String;
    return engine;
  }

  GollekEngine._();

  /// Check if engine is valid and not destroyed.
  bool get isValid => _engineId != null && !_isDestroyed;

  /// Load model from an absolute file path.
  Future<void> loadModel(String modelPath) async {
    _checkValid();
    await GollekEngineFlutterPlatform.instance.loadModel(_engineId!, modelPath);
  }

  /// Load model from Flutter assets.
  Future<void> loadModelFromAssets(String assetPath) async {
    _checkValid();
    await GollekEngineFlutterPlatform.instance.loadModelFromAssets(
      _engineId!,
      assetPath,
    );
  }

  /// Run single inference.
  Future<Uint8List> infer(Uint8List inputData) async {
    _checkValid();
    return await GollekEngineFlutterPlatform.instance.infer(
      _engineId!,
      inputData,
    );
  }

  /// Run batched inference with multiple inputs.
  Future<List<Uint8List>> inferBatch(List<Uint8List> inputs) async {
    _checkValid();
    return await GollekEngineFlutterPlatform.instance.inferBatch(
      _engineId!,
      inputs,
    );
  }

  /// Start a streaming inference session.
  Future<String> startStreaming(
    Uint8List inputData, {
    int maxTokens = 0,
  }) async {
    _checkValid();
    return await GollekEngineFlutterPlatform.instance.startStreaming(
      _engineId!,
      inputData,
      maxTokens: maxTokens,
    );
  }

  /// Get next chunk from a streaming session.
  Future<({Uint8List outputData, bool isDone})> streamNext(
    String sessionId,
  ) async {
    _checkValid();
    final result = await GollekEngineFlutterPlatform.instance.streamNext(
      sessionId,
    );
    return (
      outputData: result['outputData'] as Uint8List,
      isDone: result['isDone'] as bool,
    );
  }

  /// End a streaming session.
  Future<void> endStreaming(String sessionId) async {
    _checkValid();
    await GollekEngineFlutterPlatform.instance.endStreaming(sessionId);
  }

  /// Get input tensor info at index.
  Future<GollekTensorInfo> getInputInfo(int index) async {
    _checkValid();
    final map = await GollekEngineFlutterPlatform.instance.getInputInfo(
      _engineId!,
      index,
    );
    return GollekTensorInfo.fromMap(map);
  }

  /// Get output tensor info at index.
  Future<GollekTensorInfo> getOutputInfo(int index) async {
    _checkValid();
    final map = await GollekEngineFlutterPlatform.instance.getOutputInfo(
      _engineId!,
      index,
    );
    return GollekTensorInfo.fromMap(map);
  }

  /// Get performance metrics.
  Future<GollekMetrics> getMetrics() async {
    _checkValid();
    final map = await GollekEngineFlutterPlatform.instance.getMetrics(
      _engineId!,
    );
    return GollekMetrics.fromMap(map);
  }

  /// Reset performance metrics.
  Future<void> resetMetrics() async {
    _checkValid();
    await GollekEngineFlutterPlatform.instance.resetMetrics(_engineId!);
  }

  /// Get number of input tensors.
  Future<int> getInputCount() async {
    _checkValid();
    return await GollekEngineFlutterPlatform.instance.getInputCount(_engineId!);
  }

  /// Get number of output tensors.
  Future<int> getOutputCount() async {
    _checkValid();
    return await GollekEngineFlutterPlatform.instance.getOutputCount(
      _engineId!,
    );
  }

  /// Destroy the engine and free resources.
  Future<void> destroy() async {
    if (_isDestroyed || _engineId == null) return;
    _isDestroyed = true;
    await GollekEngineFlutterPlatform.instance.destroyEngine(_engineId!);
    _engineId = null;
  }

  void _checkValid() {
    if (!_isDestroyed && _engineId == null) {
      throw StateError('Engine not initialized. Call create() first.');
    }
    if (_isDestroyed) {
      throw StateError('Engine has been destroyed.');
    }
  }
}
