import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'gollek_engine_flutter_platform_interface.dart';

/// Method channel implementation for Gollek TFLite engine.
class MethodChannelGollekEngineFlutter extends GollekEngineFlutterPlatform {
  @visibleForTesting
  final methodChannel = const MethodChannel('gollek_engine_flutter');

  @override
  Future<Map<String, dynamic>> createEngine(Map<String, dynamic> config) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'createEngine',
      config,
    );
    return Map<String, dynamic>.from(result!);
  }

  @override
  Future<void> destroyEngine(String engineId) async {
    await methodChannel.invokeMethod<void>(
      'destroyEngine',
      {'engineId': engineId},
    );
  }

  @override
  Future<void> loadModel(String engineId, String modelPath) async {
    await methodChannel.invokeMethod<void>(
      'loadModel',
      {'engineId': engineId, 'modelPath': modelPath},
    );
  }

  @override
  Future<void> loadModelFromAssets(String engineId, String assetPath) async {
    await methodChannel.invokeMethod<void>(
      'loadModelFromAssets',
      {'engineId': engineId, 'assetName': assetPath},
    );
  }

  @override
  Future<Uint8List> infer(String engineId, Uint8List inputData) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'infer',
      {
        'engineId': engineId,
        'inputData': inputData,
      },
    );
    final outputData = result!['outputData'] as Uint8List;
    return outputData;
  }

  @override
  Future<List<Uint8List>> inferBatch(
      String engineId, List<Uint8List> inputs) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'inferBatch',
      {
        'engineId': engineId,
        'inputDatas': inputs,
      },
    );
    final outputDatas = result!['outputDatas'] as List;
    return outputDatas.cast<Uint8List>();
  }

  @override
  Future<String> startStreaming(String engineId, Uint8List inputData,
      {int maxTokens = 0}) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'startStreaming',
      {
        'engineId': engineId,
        'inputData': inputData,
        'maxTokens': maxTokens,
      },
    );
    return result!['sessionId'] as String;
  }

  @override
  Future<Map<String, dynamic>> streamNext(String sessionId) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'streamNext',
      {'sessionId': sessionId},
    );
    return Map<String, dynamic>.from(result!);
  }

  @override
  Future<void> endStreaming(String sessionId) async {
    await methodChannel.invokeMethod<void>(
      'endStreaming',
      {'sessionId': sessionId},
    );
  }

  @override
  Future<Map<String, dynamic>> getInputInfo(String engineId, int index) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'getInputInfo',
      {
        'engineId': engineId,
        'index': index,
      },
    );
    return Map<String, dynamic>.from(result!);
  }

  @override
  Future<Map<String, dynamic>> getOutputInfo(String engineId, int index) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'getOutputInfo',
      {
        'engineId': engineId,
        'index': index,
      },
    );
    return Map<String, dynamic>.from(result!);
  }

  @override
  Future<Map<String, dynamic>> getMetrics(String engineId) async {
    final result = await methodChannel.invokeMethod<Map<dynamic, dynamic>>(
      'getMetrics',
      {'engineId': engineId},
    );
    return Map<String, dynamic>.from(result!);
  }

  @override
  Future<void> resetMetrics(String engineId) async {
    await methodChannel.invokeMethod<void>(
      'resetMetrics',
      {'engineId': engineId},
    );
  }

  @override
  Future<int> getInputCount(String engineId) async {
    // For now, we'll get this from tensor info at index 0
    try {
      final info = await getInputInfo(engineId, 0);
      return 1; // Placeholder - should be implemented in native
    } catch (e) {
      return 0;
    }
  }

  @override
  Future<int> getOutputCount(String engineId) async {
    // For now, we'll get this from tensor info at index 0
    try {
      final info = await getOutputInfo(engineId, 0);
      return 1; // Placeholder - should be implemented in native
    } catch (e) {
      return 0;
    }
  }

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }
}
