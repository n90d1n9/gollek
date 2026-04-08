import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'gollek_engine_flutter_method_channel.dart';

/// Platform interface for Gollek TFLite engine.
abstract class GollekEngineFlutterPlatform extends PlatformInterface {
  GollekEngineFlutterPlatform() : super(token: _token);

  static final Object _token = Object();

  static GollekEngineFlutterPlatform _instance = MethodChannelGollekEngineFlutter();

  static GollekEngineFlutterPlatform get instance => _instance;

  static set instance(GollekEngineFlutterPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Create a new engine instance.
  Future<Map<String, dynamic>> createEngine(Map<String, dynamic> config) {
    throw UnimplementedError('createEngine() has not been implemented.');
  }

  /// Destroy an engine instance.
  Future<void> destroyEngine(String engineId) {
    throw UnimplementedError('destroyEngine() has not been implemented.');
  }

  /// Load model from file path.
  Future<void> loadModel(String engineId, String modelPath) {
    throw UnimplementedError('loadModel() has not been implemented.');
  }

  /// Load model from Flutter assets.
  Future<void> loadModelFromAssets(String engineId, String assetPath) {
    throw UnimplementedError('loadModelFromAssets() has not been implemented.');
  }

  /// Run single inference.
  Future<Uint8List> infer(String engineId, Uint8List inputData) {
    throw UnimplementedError('infer() has not been implemented.');
  }

  /// Run batched inference.
  Future<List<Uint8List>> inferBatch(String engineId, List<Uint8List> inputs) {
    throw UnimplementedError('inferBatch() has not been implemented.');
  }

  /// Start streaming inference.
  Future<String> startStreaming(String engineId, Uint8List inputData,
      {int maxTokens = 0}) {
    throw UnimplementedError('startStreaming() has not been implemented.');
  }

  /// Get next chunk from streaming session.
  Future<Map<String, dynamic>> streamNext(String sessionId) {
    throw UnimplementedError('streamNext() has not been implemented.');
  }

  /// End streaming session.
  Future<void> endStreaming(String sessionId) {
    throw UnimplementedError('endStreaming() has not been implemented.');
  }

  /// Get input tensor info.
  Future<Map<String, dynamic>> getInputInfo(String engineId, int index) {
    throw UnimplementedError('getInputInfo() has not been implemented.');
  }

  /// Get output tensor info.
  Future<Map<String, dynamic>> getOutputInfo(String engineId, int index) {
    throw UnimplementedError('getOutputInfo() has not been implemented.');
  }

  /// Get performance metrics.
  Future<Map<String, dynamic>> getMetrics(String engineId) {
    throw UnimplementedError('getMetrics() has not been implemented.');
  }

  /// Reset performance metrics.
  Future<void> resetMetrics(String engineId) {
    throw UnimplementedError('resetMetrics() has not been implemented.');
  }

  /// Get input tensor count.
  Future<int> getInputCount(String engineId) {
    throw UnimplementedError('getInputCount() has not been implemented.');
  }

  /// Get output tensor count.
  Future<int> getOutputCount(String engineId) {
    throw UnimplementedError('getOutputCount() has not been implemented.');
  }

  /// Get platform version (for testing).
  Future<String?> getPlatformVersion() {
    throw UnimplementedError('getPlatformVersion() has not been implemented.');
  }
}
