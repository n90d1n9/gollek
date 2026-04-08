import 'dart:async';
import 'dart:typed_data';

import 'gollek_engine.dart';

/// A streaming inference session.
/// 
/// Streaming returns partial results (e.g., tokens, video frames) as they
/// become available. This is especially useful for LLMs or real-time video.
/// 
/// Example usage:
/// ```dart
/// final engine = await GollekEngine.create();
/// final session = GollekStreamSession(engine);
/// 
/// await session.start(inputData);
/// while (!session.isDone) {
///   final chunk = await session.next();
///   if (chunk != null) {
///     processChunk(chunk);
///   }
/// }
/// await session.close();
/// ```
class GollekStreamSession {
  final GollekEngine _engine;
  String? _sessionId;
  bool _isDone = false;
  bool _isRunning = false;
  bool _isClosed = false;

  GollekStreamSession(this._engine);

  /// Check if streaming is complete.
  bool get isDone => _isDone;

  /// Check if session is currently running.
  bool get isRunning => _isRunning;

  /// Start streaming inference on the given input.
  Future<void> start(Uint8List input, {int maxTokens = 0}) async {
    if (_isClosed) {
      throw StateError('Session has been closed');
    }
    if (_isRunning) {
      throw StateError('Session already running');
    }

    _sessionId = await _engine.startStreaming(input, maxTokens: maxTokens);
    _isRunning = true;
    _isDone = false;
  }

  /// Get the next chunk from the streaming session.
  /// Returns null when streaming is complete.
  Future<Uint8List?> next() async {
    if (_isClosed || _sessionId == null) {
      throw StateError('Session not initialized or closed');
    }
    if (_isDone) {
      return null;
    }

    final result = await _engine.streamNext(_sessionId!);
    _isDone = result.isDone;
    
    return result.isDone && result.outputData.isEmpty
        ? null
        : result.outputData;
  }

  /// Stream all chunks as an async iterable.
  Stream<Uint8List> stream() async* {
    if (_isClosed || _sessionId == null) {
      throw StateError('Session not initialized or closed');
    }

    while (!_isDone) {
      final chunk = await next();
      if (chunk != null) {
        yield chunk;
      }
    }
  }

  /// Close the streaming session and free resources.
  Future<void> close() async {
    if (_isClosed) return;
    _isClosed = true;
    _isRunning = false;
    
    if (_sessionId != null) {
      await _engine.endStreaming(_sessionId!);
      _sessionId = null;
    }
  }
}
