import Flutter
import UIKit
import Foundation

/// Flutter plugin implementation for Gollek TFLite inference engine on iOS.
/// Pure Swift implementation - no Objective-C needed.
public class GollekEngineFlutterPlugin: NSObject, FlutterPlugin {
    
    // MARK: - Plugin Registration
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "gollek_engine_flutter",
            binaryMessenger: registrar.messenger()
        )
        let instance = GollekEngineFlutterPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    // MARK: - Engine Management
    
    private var engines: [String: GollekEngine] = [:]
    private var streamingSessions: [String: GollekStreamSession] = [:]
    private var engineCounter: Int = 0
    
    // MARK: - Method Channel Handler
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as? [String: Any]
        
        switch call.method {
        case "createEngine":
            handleCreateEngine(args: args, result: result)
        case "destroyEngine":
            handleDestroyEngine(args: args, result: result)
        case "loadModel":
            handleLoadModel(args: args, result: result)
        case "loadModelFromAssets":
            handleLoadModelFromAssets(args: args, result: result)
        case "infer":
            handleInfer(args: args, result: result)
        case "inferBatch":
            handleInferBatch(args: args, result: result)
        case "startStreaming":
            handleStartStreaming(args: args, result: result)
        case "streamNext":
            handleStreamNext(args: args, result: result)
        case "endStreaming":
            handleEndStreaming(args: args, result: result)
        case "getMetrics":
            handleGetMetrics(args: args, result: result)
        case "resetMetrics":
            handleResetMetrics(args: args, result: result)
        case "getInputInfo":
            handleGetInputInfo(args: args, result: result)
        case "getOutputInfo":
            handleGetOutputInfo(args: args, result: result)
        case "getPlatformVersion":
            result("iOS " + UIDevice.current.systemVersion)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    // MARK: - Engine Lifecycle Management
    
    private func handleCreateEngine(args: [String: Any]?, result: @escaping FlutterResult) {
        do {
            var config = GollekConfig()
            
            if let numThreads = args?["numThreads"] as? Int {
                config.numThreads = Int32(numThreads)
            }
            
            if let delegateValue = args?["delegate"] as? Int {
                config.delegate = GollekDelegate(rawValue: Int32(delegateValue)) ?? .auto
            }
            
            if let enableXnnpack = args?["enableXnnpack"] as? Bool {
                config.enableXnnpack = enableXnnpack
            }
            
            let engine = try GollekEngine(config: config)
            engineCounter += 1
            let engineId = "engine_\(engineCounter)"
            engines[engineId] = engine
            
            result(["engineId": engineId])
        } catch {
            result(FlutterError(code: "ENGINE_CREATE_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleDestroyEngine(args: [String: Any]?, result: @escaping FlutterResult) {
        guard let engineId = args?["engineId"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId required", details: nil))
            return
        }
        
        engines.removeValue(forKey: engineId)
        streamingSessions = streamingSessions.filter { !$0.key.hasPrefix("\(engineId)_stream") }
        
        result([:])
    }
    
    // MARK: - Model Loading
    
    private func handleLoadModel(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let modelPath = args?["modelPath"] as? String,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and modelPath required", details: nil))
            return
        }
        
        do {
            try engine.loadModel(from: modelPath)
            result(["success": true])
        } catch {
            result(FlutterError(code: "MODEL_LOAD_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleLoadModelFromAssets(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let assetName = args?["assetName"] as? String,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and assetName required", details: nil))
            return
        }
        
        let key = FlutterDartProject.lookupKey(forAsset: assetName)
        guard let modelPath = Bundle.main.path(forResource: key, ofType: nil) else {
            result(FlutterError(code: "ASSET_NOT_FOUND", message: "Asset '\(assetName)' not found", details: nil))
            return
        }
        
        do {
            try engine.loadModel(from: modelPath)
            result(["success": true])
        } catch {
            result(FlutterError(code: "MODEL_LOAD_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    // MARK: - Inference
    
    private func handleInfer(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let inputData = args?["inputData"] as? FlutterStandardTypedData,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and inputData required", details: nil))
            return
        }
        
        do {
            let output = try engine.infer(input: inputData.data)
            result(["outputData": FlutterStandardTypedData(bytes: output)])
        } catch {
            result(FlutterError(code: "INFER_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleInferBatch(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let inputDatas = args?["inputDatas"] as? [FlutterStandardTypedData],
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and inputDatas required", details: nil))
            return
        }
        
        do {
            let inputs = inputDatas.map { $0.data }
            let outputs = try engine.inferBatch(inputs: inputs)
            let outputDatas = outputs.map { FlutterStandardTypedData(bytes: $0) }
            result(["outputDatas": outputDatas])
        } catch {
            result(FlutterError(code: "BATCH_INFER_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    // MARK: - Streaming
    
    private func handleStartStreaming(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let inputData = args?["inputData"] as? FlutterStandardTypedData,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and inputData required", details: nil))
            return
        }
        
        let maxTokens = args?["maxTokens"] as? Int32 ?? 0
        
        do {
            let session = try engine.startStreaming(input: inputData.data, maxTokens: maxTokens)
            let sessionId = "\(engineId)_stream_\(UUID().uuidString)"
            streamingSessions[sessionId] = session
            result(["sessionId": sessionId])
        } catch {
            result(FlutterError(code: "STREAM_START_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleStreamNext(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let sessionId = args?["sessionId"] as? String,
            let session = streamingSessions[sessionId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "sessionId required", details: nil))
            return
        }
        
        do {
            let (data, isDone) = try session.next()
            result([
                "outputData": FlutterStandardTypedData(bytes: data),
                "isDone": isDone
            ])
        } catch {
            result(FlutterError(code: "STREAM_NEXT_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleEndStreaming(args: [String: Any]?, result: @escaping FlutterResult) {
        guard let sessionId = args?["sessionId"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "sessionId required", details: nil))
            return
        }
        
        streamingSessions.removeValue(forKey: sessionId)
        result([:])
    }
    
    // MARK: - Metrics
    
    private func handleGetMetrics(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let engine = engines[engineId],
            let metrics = engine.metrics
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId required", details: nil))
            return
        }
        
        result([
            "totalInferences": metrics.totalInferences,
            "failedInferences": metrics.failedInferences,
            "avgLatencyMs": metrics.avgLatencyMs,
            "p50LatencyMs": metrics.p50LatencyMs,
            "p95LatencyMs": metrics.p95LatencyMs,
            "p99LatencyMs": metrics.p99LatencyMs,
            "peakMemoryBytes": metrics.peakMemoryBytes,
            "currentMemoryBytes": metrics.currentMemoryBytes,
            "activeDelegate": metrics.activeDelegate.rawValue
        ])
    }
    
    private func handleResetMetrics(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId required", details: nil))
            return
        }
        
        do {
            try engine.resetMetrics()
            result([:])
        } catch {
            result(FlutterError(code: "METRICS_FAILED", message: error.localizedDescription, details: nil))
        }
    }
    
    // MARK: - Tensor Info
    
    private func handleGetInputInfo(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let index = args?["index"] as? Int,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and index required", details: nil))
            return
        }
        
        do {
            let info = try engine.inputInfo(at: index)
            result([
                "name": info.name,
                "type": info.type.rawValue,
                "dims": info.dims,
                "byteSize": info.byteSize,
                "scale": info.scale,
                "zeroPoint": info.zeroPoint
            ])
        } catch {
            result(FlutterError(code: "INVALID_INDEX", message: error.localizedDescription, details: nil))
        }
    }
    
    private func handleGetOutputInfo(args: [String: Any]?, result: @escaping FlutterResult) {
        guard
            let engineId = args?["engineId"] as? String,
            let index = args?["index"] as? Int,
            let engine = engines[engineId]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "engineId and index required", details: nil))
            return
        }
        
        do {
            let info = try engine.outputInfo(at: index)
            result([
                "name": info.name,
                "type": info.type.rawValue,
                "dims": info.dims,
                "byteSize": info.byteSize,
                "scale": info.scale,
                "zeroPoint": info.zeroPoint
            ])
        } catch {
            result(FlutterError(code: "INVALID_INDEX", message: error.localizedDescription, details: nil))
        }
    }
}

