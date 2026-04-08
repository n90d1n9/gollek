import 'dart:async';
import 'dart:typed_data';

import 'gollek_engine.dart';

/// Manages batching of inference requests for improved throughput.
/// 
/// Batching combines multiple inference requests into a single batch,
/// reducing per-request overhead and improving throughput on GPU/NPU.
/// 
/// Example usage:
/// ```dart
/// final engine = await GollekEngine.create();
/// final batching = GollekBatchingManager(engine);
/// 
/// final results = await Future.wait([
///   batching.submit(input1),
///   batching.submit(input2),
///   batching.submit(input3),
/// ]);
/// 
/// batching.dispose();
/// ```
class GollekBatchingManager {
  final GollekEngine _engine;
  final Duration _maxDelay;
  final int _maxBatchSize;

  final List<_PendingRequest> _pending = [];
  Timer? _batchTimer;
  int _nextId = 0;
  bool _isDisposed = false;

  GollekBatchingManager(
    this._engine, {
    Duration maxDelay = const Duration(milliseconds: 10),
    int maxBatchSize = 32,
  })  : _maxDelay = maxDelay,
        _maxBatchSize = maxBatchSize;

  /// Submit a single input, get a Future that completes with the output.
  Future<Uint8List> submit(Uint8List input) {
    if (_isDisposed) {
      throw StateError('BatchingManager has been disposed');
    }

    final completer = Completer<Uint8List>();
    final request = _PendingRequest(_nextId++, input, completer);
    _pending.add(request);
    _scheduleBatch();
    return completer.future;
  }

  void _scheduleBatch() {
    if (_batchTimer != null || _isDisposed) return;
    _batchTimer = Timer(_maxDelay, () => _processBatch());
  }

  Future<void> _processBatch() async {
    _batchTimer = null;
    if (_pending.isEmpty || _isDisposed) return;

    // Take up to _maxBatchSize requests
    final batch = <_PendingRequest>[];
    while (batch.length < _maxBatchSize && _pending.isNotEmpty) {
      batch.add(_pending.removeAt(0));
    }

    try {
      // Run batch inference
      final inputs = batch.map((r) => r.input).toList();
      final outputs = await _engine.inferBatch(inputs);

      // Complete each request with its output
      for (int i = 0; i < batch.length; i++) {
        if (i < outputs.length) {
          batch[i].completer.complete(outputs[i]);
        } else {
          batch[i].completer.completeError(
            StateError('Output mismatch in batch'),
          );
        }
      }
    } catch (e) {
      // Complete all requests with error
      for (final request in batch) {
        request.completer.completeError(e);
      }
    }

    // Schedule next batch if pending requests remain
    if (_pending.isNotEmpty && !_isDisposed) {
      _scheduleBatch();
    }
  }

  /// Dispose the batching manager and cancel pending requests.
  void dispose() {
    _isDisposed = true;
    _batchTimer?.cancel();
    _batchTimer = null;
    
    for (final req in _pending) {
      req.completer.completeError(
        StateError('Batching manager disposed'),
      );
    }
    _pending.clear();
  }
}

class _PendingRequest {
  final int id;
  final Uint8List input;
  final Completer<Uint8List> completer;

  _PendingRequest(this.id, this.input, this.completer);
}
