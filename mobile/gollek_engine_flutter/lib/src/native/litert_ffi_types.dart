import 'dart:ffi';

/// LiteRT status codes (from litert_common.h).
enum LitertStatus {
  ok(0),
  errorInvalidArgument(1),
  errorMemoryAllocation(2),
  errorRuntimeFailure(3),
  errorMissingInput(4),
  errorUnsupported(5),
  errorNotFound(6),
  errorTimeout(7),
  errorWrongVersion(8),
  errorUnknown(9);

  final int value;
  const LitertStatus(this.value);

  static LitertStatus fromInt(int value) {
    return LitertStatus.values.firstWhere(
      (e) => e.value == value,
      orElse: () => LitertStatus.errorUnknown,
    );
  }
}

/// LiteRT tensor element types (from litert_common.h).
enum LitertType {
  noType(0),
  float32(1),
  int32(2),
  uint8(3),
  int64(4),
  string(5),
  bool(6),
  int16(7),
  complex64(8),
  int8(9),
  float16(10),
  float64(11),
  complex128(12),
  uint64(13),
  resource(14),
  variant(15),
  uint32(16),
  uint16(17),
  int4(18),
  float8e5m2(19);

  final int value;
  const LitertType(this.value);

  static LitertType fromInt(int value) {
    return LitertType.values.firstWhere(
      (e) => e.value == value,
      orElse: () => LitertType.noType,
    );
  }
}

/// Hardware accelerator bitmasks (from litert_common.h).
class LitertHwAccelerator {
  static const int none = 0;
  static const int cpu = 1; // 1 << 0
  static const int gpu = 2; // 1 << 1
  static const int npu = 4; // 1 << 2
}

/// TensorBuffer Lock Modes (from litert_common.h).
class LitertLockMode {
  static const int read = 0;
  static const int write = 1;
  static const int readWrite = 2;
}

// Opaque Types (Wrappers for native pointers)
final class LitertEnvironment extends Opaque {}
final class LitertModel extends Opaque {}
final class LitertOptions extends Opaque {}
final class LitertCompiledModel extends Opaque {}
final class LitertTensorBuffer extends Opaque {}
final class LitertTensorBufferRequirements extends Opaque {}
final class LitertSignature extends Opaque {}
final class LitertSubgraph extends Opaque {}
final class LitertTensor extends Opaque {}
final class LitertRankedTensorType extends Opaque {}
final class LitertLayout extends Opaque {}
