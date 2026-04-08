import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

import 'litert_ffi_types.dart';

/// Dart FFI bindings for the LiteRT 2.0 C API.
class LitertFfiBindings {
  late final DynamicLibrary _lib;

  // Environment
  late final int Function(int, Pointer<NativeType>, Pointer<Pointer<LitertEnvironment>>) _createEnvironment;
  late final void Function(Pointer<LitertEnvironment>) _destroyEnvironment;

  // Model
  late final int Function(Pointer<Utf8>, Pointer<Pointer<LitertModel>>) _createModelFromFile;
  late final void Function(Pointer<LitertModel>) _destroyModel;
  late final int Function(Pointer<LitertModel>, Pointer<Size>) _getNumModelSignatures;
  late final int Function(Pointer<LitertModel>, int, Pointer<Pointer<LitertSignature>>) _getModelSignature;

  // Options
  late final int Function(Pointer<Pointer<LitertOptions>>) _createOptions;
  late final void Function(Pointer<LitertOptions>) _destroyOptions;
  late final int Function(Pointer<LitertOptions>, int) _setOptionsHardwareAccelerators;

  // CompiledModel
  late final int Function(Pointer<LitertEnvironment>, Pointer<LitertModel>, Pointer<LitertOptions>,
      Pointer<Pointer<LitertCompiledModel>>) _createCompiledModel;
  late final void Function(Pointer<LitertCompiledModel>) _destroyCompiledModel;
  late final int Function(Pointer<LitertCompiledModel>, int, int, Pointer<Pointer<LitertTensorBuffer>>, int,
      Pointer<Pointer<LitertTensorBuffer>>) _runCompiledModel;

  // Tensor Introspection
  late final int Function(Pointer<LitertTensor>, Pointer<LitertRankedTensorType>) _getRankedTensorType;
  late final int Function(Pointer<LitertSignature>, int, Pointer<Pointer<LitertTensor>>) _getSignatureInputTensorByIndex;
  late final int Function(Pointer<LitertSignature>, int, Pointer<Pointer<LitertTensor>>) _getSignatureOutputTensorByIndex;

  // TensorBuffer
  late final int Function(Pointer<LitertRankedTensorType>, Pointer<NativeType>, int, Pointer<NativeType>,
      Pointer<Pointer<LitertTensorBuffer>>) _createTensorBufferFromHostMemory;
  late final void Function(Pointer<LitertTensorBuffer>) _destroyTensorBuffer;
  late final int Function(Pointer<LitertTensorBuffer>, Pointer<Pointer<NativeType>>, Pointer<Size>) _lockTensorBuffer;
  late final int Function(Pointer<LitertTensorBuffer>) _unlockTensorBuffer;

  // Status Utility
  late final Pointer<Utf8> Function(int) _getStatusString;

  LitertFfiBindings({String? libraryPath}) {
    final path = libraryPath ?? _resolveLibraryPath();
    _lib = DynamicLibrary.open(path);
    _initializeFunctions();
  }

  void _initializeFunctions() {
    _getStatusString = _lib.lookupFunction<Pointer<Utf8> Function(Int32), Pointer<Utf8> Function(int)>('LiteRtGetStatusString');

    _createEnvironment = _lib.lookupFunction<Int32 Function(Int32, Pointer<NativeType>, Pointer<Pointer<LitertEnvironment>>),
        int Function(int, Pointer<NativeType>, Pointer<Pointer<LitertEnvironment>>)>('LiteRtCreateEnvironment');

    _destroyEnvironment = _lib.lookupFunction<Void Function(Pointer<LitertEnvironment>), void Function(Pointer<LitertEnvironment>)>(
        'LiteRtDestroyEnvironment');

    _createModelFromFile = _lib.lookupFunction<Int32 Function(Pointer<Utf8>, Pointer<Pointer<LitertModel>>),
        int Function(Pointer<Utf8>, Pointer<Pointer<LitertModel>>)>('LiteRtCreateModelFromFile');

    _destroyModel = _lib.lookupFunction<Void Function(Pointer<LitertModel>), void Function(Pointer<LitertModel>)>('LiteRtDestroyModel');

    _getNumModelSignatures = _lib.lookupFunction<Int32 Function(Pointer<LitertModel>, Pointer<Size>),
        int Function(Pointer<LitertModel>, Pointer<Size>)>('LiteRtGetNumModelSignatures');

    _getModelSignature = _lib.lookupFunction<Int32 Function(Pointer<LitertModel>, Size, Pointer<Pointer<LitertSignature>>),
        int Function(Pointer<LitertModel>, int, Pointer<Pointer<LitertSignature>>)>('LiteRtGetModelSignature');

    _createOptions =
        _lib.lookupFunction<Int32 Function(Pointer<Pointer<LitertOptions>>), int Function(Pointer<Pointer<LitertOptions>>)>('LiteRtCreateOptions');

    _destroyOptions =
        _lib.lookupFunction<Void Function(Pointer<LitertOptions>), void Function(Pointer<LitertOptions>)>('LiteRtDestroyOptions');

    _setOptionsHardwareAccelerators =
        _lib.lookupFunction<Int32 Function(Pointer<LitertOptions>, Int32), int Function(Pointer<LitertOptions>, int)>(
            'LiteRtSetOptionsHardwareAccelerators');

    _createCompiledModel = _lib.lookupFunction<
        Int32 Function(
            Pointer<LitertEnvironment>, Pointer<LitertModel>, Pointer<LitertOptions>, Pointer<Pointer<LitertCompiledModel>>),
        int Function(Pointer<LitertEnvironment>, Pointer<LitertModel>, Pointer<LitertOptions>,
            Pointer<Pointer<LitertCompiledModel>>)>('LiteRtCreateCompiledModel');

    _destroyCompiledModel = _lib.lookupFunction<Void Function(Pointer<LitertCompiledModel>), void Function(Pointer<LitertCompiledModel>)>(
        'LiteRtDestroyCompiledModel');

    _runCompiledModel = _lib.lookupFunction<
        Int32 Function(Pointer<LitertCompiledModel>, Size, Size, Pointer<Pointer<LitertTensorBuffer>>, Size,
            Pointer<Pointer<LitertTensorBuffer>>),
        int Function(Pointer<LitertCompiledModel>, int, int, Pointer<Pointer<LitertTensorBuffer>>, int,
            Pointer<Pointer<LitertTensorBuffer>>)>('LiteRtRunCompiledModel');

    _getRankedTensorType = _lib.lookupFunction<Int32 Function(Pointer<LitertTensor>, Pointer<LitertRankedTensorType>),
        int Function(Pointer<LitertTensor>, Pointer<LitertRankedTensorType>)>('LiteRtGetRankedTensorType');

    _getSignatureInputTensorByIndex = _lib.lookupFunction<Int32 Function(Pointer<LitertSignature>, Size, Pointer<Pointer<LitertTensor>>),
        int Function(Pointer<LitertSignature>, int, Pointer<Pointer<LitertTensor>>)>('LiteRtGetSignatureInputTensorByIndex');

    _getSignatureOutputTensorByIndex = _lib.lookupFunction<Int32 Function(Pointer<LitertSignature>, Size, Pointer<Pointer<LitertTensor>>),
        int Function(Pointer<LitertSignature>, int, Pointer<Pointer<LitertTensor>>)>('LiteRtGetSignatureOutputTensorByIndex');

    // TensorBuffer
    _createTensorBufferFromHostMemory = _lib.lookupFunction<
        Int32 Function(Pointer<LitertRankedTensorType>, Pointer<NativeType>, Size, Pointer<NativeType>, Pointer<Pointer<LitertTensorBuffer>>),
        int Function(Pointer<LitertRankedTensorType>, Pointer<NativeType>, int, Pointer<NativeType>,
            Pointer<Pointer<LitertTensorBuffer>>)>('LiteRtCreateTensorBufferFromHostMemory');

    _destroyTensorBuffer =
        _lib.lookupFunction<Void Function(Pointer<LitertTensorBuffer>), void Function(Pointer<LitertTensorBuffer>)>('LiteRtDestroyTensorBuffer');

    _lockTensorBuffer = _lib.lookupFunction<Int32 Function(Pointer<LitertTensorBuffer>, Pointer<Pointer<NativeType>>, Pointer<Size>),
        int Function(Pointer<LitertTensorBuffer>, Pointer<Pointer<NativeType>>, Pointer<Size>)>('LiteRtLockTensorBuffer');

    _unlockTensorBuffer =
        _lib.lookupFunction<Int32 Function(Pointer<LitertTensorBuffer>), int Function(Pointer<LitertTensorBuffer>)>('LiteRtUnlockTensorBuffer');
  }

  String _resolveLibraryPath() {
    if (Platform.isMacOS) return 'libLiteRt.dylib';
    if (Platform.isLinux || Platform.isAndroid) return 'libLiteRt.so';
    if (Platform.isWindows) return 'LiteRt.dll';
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  // --- Public Wrappers ---

  String getStatusString(int status) => _getStatusString(status).toDartString();

  void checkStatus(int status) {
    if (status != LitertStatus.ok.value) {
      throw Exception('LiteRT Error ($status): ${getStatusString(status)}');
    }
  }

  Pointer<LitertEnvironment> createEnvironment() {
    final ptr = calloc<Pointer<LitertEnvironment>>();
    try {
      checkStatus(_createEnvironment(0, nullptr, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  void destroyEnvironment(Pointer<LitertEnvironment> env) => _destroyEnvironment(env);

  Pointer<LitertModel> createModelFromFile(String path) {
    final pathPtr = path.toNativeUtf8();
    final modelPtr = calloc<Pointer<LitertModel>>();
    try {
      checkStatus(_createModelFromFile(pathPtr, modelPtr));
      return modelPtr.value;
    } finally {
      calloc.free(pathPtr);
      calloc.free(modelPtr);
    }
  }

  void destroyModel(Pointer<LitertModel> model) => _destroyModel(model);

  int getNumModelSignatures(Pointer<LitertModel> model) {
    final ptr = calloc<Size>();
    try {
      checkStatus(_getNumModelSignatures(model, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  Pointer<LitertSignature> getModelSignature(Pointer<LitertModel> model, int index) {
    final ptr = calloc<Pointer<LitertSignature>>();
    try {
      checkStatus(_getModelSignature(model, index, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  Pointer<LitertOptions> createOptions() {
    final ptr = calloc<Pointer<LitertOptions>>();
    try {
      checkStatus(_createOptions(ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  void destroyOptions(Pointer<LitertOptions> options) => _destroyOptions(options);

  void setHardwareAccelerators(Pointer<LitertOptions> options, int accelerators) {
    checkStatus(_setOptionsHardwareAccelerators(options, accelerators));
  }

  Pointer<LitertCompiledModel> createCompiledModel(Pointer<LitertEnvironment> env, Pointer<LitertModel> model, Pointer<LitertOptions> options) {
    final ptr = calloc<Pointer<LitertCompiledModel>>();
    try {
      checkStatus(_createCompiledModel(env, model, options, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  void destroyCompiledModel(Pointer<LitertCompiledModel> model) => _destroyCompiledModel(model);

  void runCompiledModel(Pointer<LitertCompiledModel> model, int sigIndex, int numInputs,
      Pointer<Pointer<LitertTensorBuffer>> inputBuffers, int numOutputs, Pointer<Pointer<LitertTensorBuffer>> outputBuffers) {
    checkStatus(_runCompiledModel(model, sigIndex, numInputs, inputBuffers, numOutputs, outputBuffers));
  }

  Pointer<LitertTensorBuffer> createTensorBufferFromHostMemory(Pointer<LitertRankedTensorType> type, Pointer<NativeType> hostMem, int size) {
    final ptr = calloc<Pointer<LitertTensorBuffer>>();
    try {
      checkStatus(_createTensorBufferFromHostMemory(type, hostMem, size, nullptr, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  void destroyTensorBuffer(Pointer<LitertTensorBuffer> buffer) => _destroyTensorBuffer(buffer);

  Pointer<NativeType> lockTensorBuffer(Pointer<LitertTensorBuffer> buffer) {
    final dataPtr = calloc<Pointer<NativeType>>();
    final sizePtr = calloc<Size>();
    try {
      checkStatus(_lockTensorBuffer(buffer, dataPtr, sizePtr));
      return dataPtr.value;
    } finally {
      calloc.free(dataPtr);
      calloc.free(sizePtr);
    }
  }

  void unlockTensorBuffer(Pointer<LitertTensorBuffer> buffer) => checkStatus(_unlockTensorBuffer(buffer));

  Pointer<LitertTensor> getSignatureInputTensor(Pointer<LitertSignature> sig, int index) {
    final ptr = calloc<Pointer<LitertTensor>>();
    try {
      checkStatus(_getSignatureInputTensorByIndex(sig, index, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  Pointer<LitertTensor> getSignatureOutputTensor(Pointer<LitertSignature> sig, int index) {
    final ptr = calloc<Pointer<LitertTensor>>();
    try {
      checkStatus(_getSignatureOutputTensorByIndex(sig, index, ptr));
      return ptr.value;
    } finally {
      calloc.free(ptr);
    }
  }

  Pointer<LitertRankedTensorType> getRankedTensorType(Pointer<LitertTensor> tensor) {
    // LiteRtRankedTensorType is a struct (at least 8 bytes for type + layout).
    // Allocate a buffer to hold the struct data. 128 bytes is plenty.
    final structPtr = malloc.allocate<Uint8>(128).cast<LitertRankedTensorType>();
    try {
      checkStatus(_getRankedTensorType(tensor, structPtr));
      return structPtr;
    } catch (e) {
      malloc.free(structPtr);
      rethrow;
    }
  }
}
