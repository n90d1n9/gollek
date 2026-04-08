#include "include/gollek_engine_flutter/gollek_engine_flutter_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "gollek_engine_flutter_plugin.h"

void GollekEngineFlutterPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  gollek_engine_flutter::GollekEngineFlutterPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
