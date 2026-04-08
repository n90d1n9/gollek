import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import 'litert_ffi_bindings.dart';
import 'litert_ffi_types.dart';

/// High-level Dart runner for LiteRT 2.0.
///
/// Wraps the native FFI bindings in a safe, easy-to-use API.
class LitertFfiRunner {
  final LitertFfiBindings _bindings;
  Pointer<LitertEnvironment>? _env;
  Pointer<LitertModel>? _model;
  Pointer<LitertOptions>? _options;
  Pointer<LitertCompiledModel>? _compiledModel;
  Pointer<LitertSignature>? _signature;

  // Tensor info
  Pointer<LitertRankedTensorType>? _inputTensorType;
  Pointer<LitertRankedTensorType>? _outputTensorType;

  bool _isInitialized = false;

  LitertFfiRunner({String? libraryPath}) : _bindings = LitertFfiBindings(libraryPath: libraryPath);

  /// Initialize the runner with a model and hardware options.
  void initialize(String modelPath, {bool useGpu = false, bool useNpu = false}) {
    if (_isInitialized) return;

    _env = _bindings.createEnvironment();
    _model = _bindings.createModelFromFile(modelPath);
    _options = _bindings.createOptions();

    // Set hardware accelerators (bitmask-based API)
    int accelerators = LitertHwAccelerator.cpu;
    if (useGpu) accelerators |= LitertHwAccelerator.gpu;
    if (useNpu) accelerators |= LitertHwAccelerator.npu;

    _bindings.setHardwareAccelerators(_options!, accelerators);

    _compiledModel = _bindings.createCompiledModel(_env!, _model!, _options!);

    // Introspect primary signature
    final numSigs = _bindings.getNumModelSignatures(_model!);
    if (numSigs > 0) {
      _signature = _bindings.getModelSignature(_model!, 0);
      
      // Get input/output tensor types for buffer creation
      final inputTensor = _bindings.getSignatureInputTensor(_signature!, 0);
      final outputTensor = _bindings.getSignatureOutputTensor(_signature!, 0);
      
      _inputTensorType = _bindings.getRankedTensorType(inputTensor);
      _outputTensorType = _bindings.getRankedTensorType(outputTensor);
    }

    _isInitialized = true;
  }

  /// Run inference with a single input buffer and return the output buffer.
  Uint8List infer(Uint8List inputData) {
    _checkInitialized();

    if (_inputTensorType == null || _outputTensorType == null) {
      throw StateError('Model signature or tensor types not found.');
    }

    // 1. Create Input TensorBuffer wrapping our host memory.
    final inputPtr = _copyToNative(inputData);
    final inputBufferData = _bindings.createTensorBufferFromHostMemory(
      _inputTensorType!,
      inputPtr.cast<NativeType>(),
      inputData.length,
    );

    // 2. Create Output TensorBuffer wrapping our host memory.
    final outputSize = 1001 * 4; 
    final outputPtr = malloc.allocate<Uint8>(outputSize);
    final outputBufferData = _bindings.createTensorBufferFromHostMemory(
      _outputTensorType!,
      outputPtr.cast<NativeType>(),
      outputSize,
    );

    // 3. Prepare Buffer Arrays for Run API
    final inBuffersArr = calloc<Pointer<LitertTensorBuffer>>(1);
    inBuffersArr[0] = inputBufferData;

    final outBuffersArr = calloc<Pointer<LitertTensorBuffer>>(1);
    outBuffersArr[0] = outputBufferData;

    try {
      // 4. Run Inference (Signature Index 0, 1 Input, 1 Output)
      // Note: We MUST NOT manually lock the buffers here; the native runtime handles locking
      // the buffers during the inference execution.
      _bindings.runCompiledModel(
        _compiledModel!,
        0, 
        1, 
        inBuffersArr,
        1, 
        outBuffersArr,
      );

      // 5. Copy Result directly from our host memory.
      final result = Uint8List.fromList(outputPtr.asTypedList(outputSize));
      return result;
    } finally {
      _bindings.destroyTensorBuffer(inputBufferData);
      _bindings.destroyTensorBuffer(outputBufferData);
      calloc.free(inBuffersArr);
      calloc.free(outBuffersArr);
      malloc.free(inputPtr);
      malloc.free(outputPtr);
    }
  }

  /// Free all native resources.
  void close() {
    if (_compiledModel != null) _bindings.destroyCompiledModel(_compiledModel!);
    if (_options != null) _bindings.destroyOptions(_options!);
    if (_model != null) _bindings.destroyModel(_model!);
    if (_env != null) _bindings.destroyEnvironment(_env!);
    
    if (_inputTensorType != null) malloc.free(_inputTensorType!);
    if (_outputTensorType != null) malloc.free(_outputTensorType!);
    
    _isInitialized = false;
  }

  Pointer<Uint8> _copyToNative(Uint8List data) {
    final ptr = malloc.allocate<Uint8>(data.length);
    final list = ptr.asTypedList(data.length);
    list.setAll(0, data);
    return ptr;
  }

  void _checkInitialized() {
    if (!_isInitialized) {
      throw StateError('LitertFfiRunner is not initialized. Call initialize() first.');
    }
  }
}
