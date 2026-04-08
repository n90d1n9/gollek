import 'package:flutter_test/flutter_test.dart';
import 'package:gollek_engine_flutter/gollek_engine_flutter.dart';
import 'package:gollek_engine_flutter/gollek_engine_flutter_platform_interface.dart';
import 'package:gollek_engine_flutter/gollek_engine_flutter_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockGollekEngineFlutterPlatform
    with MockPlatformInterfaceMixin
    implements GollekEngineFlutterPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final GollekEngineFlutterPlatform initialPlatform = GollekEngineFlutterPlatform.instance;

  test('$MethodChannelGollekEngineFlutter is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelGollekEngineFlutter>());
  });

  test('getPlatformVersion', () async {
    GollekEngineFlutter gollekEngineFlutterPlugin = GollekEngineFlutter();
    MockGollekEngineFlutterPlatform fakePlatform = MockGollekEngineFlutterPlatform();
    GollekEngineFlutterPlatform.instance = fakePlatform;

    expect(await gollekEngineFlutterPlugin.getPlatformVersion(), '42');
  });
}
