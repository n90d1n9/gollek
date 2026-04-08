# Enhanced Runner Plugin System v2.0 - Implementation Plan

**Date**: 2026-03-25  
**Status**: 🔄 In Progress  
**Priority**: High

---

## Overview

Enhancing the runner plugin system with the same comprehensive improvements as the kernel plugin system v2.0, providing production-ready features for model format support.

---

## Key Improvements (Planned)

### 1. Enhanced SPI Interface ✅

**File**: `RunnerPlugin.java` (Updated)

**New Features**:
- Comprehensive lifecycle management (`validate()`, `initialize()`, `health()`, `shutdown()`)
- Typed operations with `RunnerRequest`, `RunnerContext`, `RunnerResult<T>`
- Async execution support with `executeAsync()`
- Model loading/unloading with `loadModel()`, `unloadModel()`
- Request type support (INFER, EMBED, RERANK, CLASSIFY, TOKENIZE)
- Format detection utilities

**Comparison**:
| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Lifecycle** | Basic | ✅ validate/initialize/health/shutdown |
| **Type Safety** | Map-based | ✅ Generic RunnerResult<T> |
| **Operations** | infer/stream/embed | ✅ Unified execute() with RequestType |
| **Model Mgmt** | createSession | ✅ loadModel/unloadModel |
| **Async** | Uni/Multi | ✅ CompletionStage support |
| **Validation** | None | ✅ Pre-initialization validation |
| **Error Handling** | Generic | ✅ Specific exception hierarchy |

---

### 2. Supporting Classes (To Create)

#### Core Types
- [ ] `RunnerRequest` - Typed operation request
- [ ] `RunnerContext` - Execution context with configuration
- [ ] `RunnerResult<T>` - Typed result with metadata
- [ ] `RunnerConfig` - Type-safe configuration
- [ ] `RunnerValidationResult` - Validation with errors/warnings
- [ ] `RunnerHealth` - Health status with details
- [ ] `ModelHandle` - Loaded model reference
- [ ] `ModelLoadRequest` - Model loading parameters

#### Exception Hierarchy
- [ ] `RunnerException` (base)
- [ ] `RunnerInitializationException` (unchecked)
- [ ] `RunnerExecutionException` (unchecked)
- [ ] `RunnerNotFoundException` (unchecked)
- [ ] `ModelLoadException` (unchecked)
- [ ] `UnknownRequestException` (unchecked)

#### Management
- [ ] `RunnerPluginManager` (Enhanced) - Lifecycle, metrics, health
- [ ] `RunnerPluginLoader` - ClassLoader isolation, hot-reload
- [ ] `RunnerMetrics` - Metrics collection
- [ ] `RunnerPluginProducer` - CDI producer

---

### 3. Engine Integration (To Create)

#### Integration Components
- [ ] `RunnerPluginIntegration` - Bridge with main plugin system
- [ ] `RunnerPluginProducer` - CDI producer for injection
- [ ] Update `PluginSystemIntegrator` - Use runner plugin producer

#### Usage Pattern
```java
@Inject
RunnerPluginManager runnerManager;

public void loadModel() {
    ModelLoadRequest request = ModelLoadRequest.builder()
        .modelPath("/path/to/model.gguf")
        .format("gguf")
        .config(runnerConfig)
        .build();

    ModelHandle model = runnerManager.loadModel(
        request, RunnerContext.empty());
}
```

---

## Implementation Status

### Phase 1: Core SPI ✅
- [x] Enhanced `RunnerPlugin` interface
- [x] `RequestType` enum
- [ ] Supporting classes (RunnerRequest, RunnerContext, etc.)
- [ ] Exception hierarchy

### Phase 2: Management
- [ ] Enhanced `RunnerPluginManager`
- [ ] `RunnerPluginLoader` with ClassLoader isolation
- [ ] `RunnerMetrics` for observability

### Phase 3: Engine Integration
- [ ] `RunnerPluginIntegration`
- [ ] `RunnerPluginProducer` (CDI)
- [ ] Update `PluginSystemIntegrator`

### Phase 4: Migration
- [ ] Update existing GGUF runner plugin
- [ ] Update other runner plugins (ONNX, TensorRT, etc.)
- [ ] Migration guide

### Phase 5: Documentation
- [ ] Enhanced runner plugin architecture doc
- [ ] Usage examples
- [ ] Website updates

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Core SPI | 1 day | ✅ 50% Complete |
| Phase 2: Management | 2 days | ⏳ Pending |
| Phase 3: Integration | 1 day | ⏳ Pending |
| Phase 4: Migration | 2 days | ⏳ Pending |
| Phase 5: Documentation | 1 day | ⏳ Pending |

**Total**: 7 days

---

## Benefits

### For Developers
- ✅ Type-safe operations with generics
- ✅ Comprehensive error handling
- ✅ Better IDE support with typed APIs
- ✅ Consistent lifecycle management

### For Operations
- ✅ ClassLoader isolation prevents conflicts
- ✅ Hot-reload support for updates
- ✅ Health monitoring and metrics
- ✅ Better troubleshooting with validation

### For Users
- ✅ Faster model loading
- ✅ Better error messages
- ✅ More reliable inference
- ✅ Consistent API across formats

---

## Migration Path

### From v1.0 to v2.0

**Old Code**:
```java
RunnerSession session = runner.createSession(modelPath, config);
Uni<InferenceResponse> response = session.infer(request);
```

**New Code**:
```java
ModelLoadRequest loadRequest = ModelLoadRequest.builder()
    .modelPath(modelPath)
    .config(config)
    .build();

ModelHandle model = runner.loadModel(loadRequest, context);

RunnerRequest inferRequest = RunnerRequest.builder()
    .type(RequestType.INFER)
    .inferenceRequest(request)
    .build();

RunnerResult<InferenceResponse> result = runner.execute(
    inferRequest, context);
```

### Backward Compatibility

- ✅ Legacy methods marked `@Deprecated` but still work
- ✅ Gradual migration path
- ✅ No breaking changes for existing code

---

## Next Steps

1. **Complete Phase 1** - Create remaining supporting classes
2. **Start Phase 2** - Enhance RunnerPluginManager
3. **Create integration components** - Bridge with engine
4. **Update GGUF plugin** - Migrate to v2.0 SPI
5. **Write documentation** - Complete usage guide

---

**Status**: 🔄 **IMPLEMENTATION IN PROGRESS**

The runner plugin system enhancement has begun with the core SPI interface updated. Supporting classes and integration components will be created following the same pattern as the successful kernel plugin system v2.0.
