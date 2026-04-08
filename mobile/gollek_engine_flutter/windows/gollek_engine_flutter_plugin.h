#ifndef FLUTTER_PLUGIN_GOLLEK_ENGINE_FLUTTER_PLUGIN_H_
#define FLUTTER_PLUGIN_GOLLEK_ENGINE_FLUTTER_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace gollek_engine_flutter {

class GollekEngineFlutterPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  GollekEngineFlutterPlugin();

  virtual ~GollekEngineFlutterPlugin();

  // Disallow copy and assign.
  GollekEngineFlutterPlugin(const GollekEngineFlutterPlugin&) = delete;
  GollekEngineFlutterPlugin& operator=(const GollekEngineFlutterPlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace gollek_engine_flutter

#endif  // FLUTTER_PLUGIN_GOLLEK_ENGINE_FLUTTER_PLUGIN_H_
