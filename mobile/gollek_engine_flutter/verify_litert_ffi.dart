import 'dart:io';
import 'dart:typed_data';
import 'lib/src/native/litert_ffi_runner.dart';
import 'lib/src/native/litert_ffi_types.dart';

void main() {
  print('╔══════════════════════════════════════════════════════════╗');
  print('║       LiteRT 2.0 Dart FFI Verification                ║');
  print('╚══════════════════════════════════════════════════════════╝');
  print('');

  final home = Platform.environment['HOME'];
  final libPath = '$home/.gollek/libs/libLiteRt.dylib';
  final modelPath = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/examples/jbang/edge/models/1.tflite';

  if (!File(libPath).existsSync()) {
    print('❌ Library not found: $libPath');
    exit(1);
  }
  if (!File(modelPath).existsSync()) {
    print('❌ Model not found: $modelPath');
    exit(1);
  }

  print('Library:  $libPath');
  print('Model:    $modelPath');
  print('');

  try {
    final runner = LitertFfiRunner(libraryPath: libPath);
    
    print('Step 1: Initializing runner...         ');
    runner.initialize(modelPath, useGpu: false);
    print('✓ Runner initialized');

    print('Step 2: Preparing dummy input...       ');
    // MobileNetV2 input: 224x224x3 float32 = 150528 floats = 602112 bytes
    final inputSize = 224 * 224 * 3 * 4;
    final inputData = Uint8List(inputSize);
    print('✓ Input data ready (${inputData.length} bytes)');

    print('Step 3: Running inference...           ');
    final stopwatch = Stopwatch()..start();
    final output = runner.infer(inputData);
    stopwatch.stop();
    
    print('✓ Inference successful (${stopwatch.elapsedMilliseconds}ms)');
    print('  Output size: ${output.length} bytes');

    print('Step 4: Cleanup...                      ');
    runner.close();
    print('✓ Runner closed');

    print('');
    print('══════════════════════════════════════════════════════════');
    print('  ✅ STATUS: LiteRT 2.0 Dart FFI is STABLE');
    print('══════════════════════════════════════════════════════════');
  } catch (e, stack) {
    print('❌ Verification failed: $e');
    print(stack);
    exit(1);
  }
}
