import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gollek_engine_flutter/gollek_engine_flutter_method_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  MethodChannelGollekEngineFlutter platform = MethodChannelGollekEngineFlutter();
  const MethodChannel channel = MethodChannel('gollek_engine_flutter');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          return '42';
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });
}
