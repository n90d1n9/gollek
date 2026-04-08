#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint gollek_engine_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'gollek_engine_flutter'
  s.version          = '0.2.0'
  s.summary          = 'Gollek LiteRT Inference Engine for Flutter (Pure Swift)'
  s.description      = <<-DESC
High-performance LiteRT inference engine for Flutter with support
for batching, streaming, caching, and metrics. Pure Swift implementation.
                       DESC
  s.homepage         = 'https://github.com/wayang-platform/gollek'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Wayang Platform' => 'team@wayang.dev' }
  s.source           = { :path => '.' }
  
  # Source files - Swift only + C++ core
  s.source_files = [
    'Classes/**/*.swift',
    '../../plugins/runner/edge/gollek-native-core/include/**/*.h',
    '../../plugins/runner/edge/gollek-native-core/src/**/*.cpp',
    '../../plugins/runner/edge/gollek-native-core/platform/ios/**/*.swift'
  ]
  
  # iOS platform
  s.platform = :ios, '13.0'
  
  # Flutter dependency
  s.dependency 'Flutter'
  
  # TFLite dependency - use TensorFlowLiteSwift (Swift native API)
  s.dependency 'TensorFlowLiteSwift', '~> 2.16.0'
  
  # Compiler settings for C++
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'SWIFT_VERSION' => '5.0',
    'HEADER_SEARCH_PATHS' => '"$(PODS_ROOT)/TensorFlowLiteSwift/tensorflow/lite/swift/Sources" "$(PODS_TARGET_SRCROOT)/../../plugins/runner/litert/gollek-native-core/include"'
  }
  
  s.swift_version = '5.0'
  
  # Frameworks
  s.frameworks = 'Accelerate'
  
  # Weak frameworks for optional features
  s.weak_frameworks = 'Metal', 'CoreML'
end
