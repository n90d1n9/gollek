# Gollek SPI Code Organization Plan

## Current Structure (in `gollek-spi/`)

```
gollek-spi/src/main/java/tech/kayys/gollek/spi/
├── auth/                    (1 file)   → Keep in gollek-spi (common)
├── context/                 (3 files)  → Keep in gollek-spi (common)
├── error/                   (?)        → Keep in gollek-spi (common)
├── exception/               (?)        → Keep in gollek-spi (common)
├── execution/               (?)        → Move to gollek-spi-inference
├── inference/               (?)        → Move to gollek-spi-inference
├── model/                   (?)        → Move to gollek-spi-model
├── observability/           (?)        → Keep in gollek-spi (common)
├── plugin/                  (?)        → Move to gollek-spi-plugin
├── provider/                (?)        → Move to gollek-spi-provider
├── registry/                (?)        → Depends on content
├── routing/                 (?)        → Move to gollek-spi-provider
├── storage/                 (?)        → Move to gollek-spi-model
├── stream/                  (?)        → Keep in gollek-spi (common)
└── tool/                    (?)        → Keep in gollek-spi (common)
```

## Target Structure

### 1. Keep in `gollek-spi` (Common SPI)
**Purpose**: Shared interfaces, utilities, and types used by ALL SPI modules

**Packages to keep**:
- `auth/` - Authentication constants
- `context/` - Request/Engine context
- `error/` - Error codes
- `exception/` - Common exceptions
- `observability/` - Metrics, audit (cross-cutting concern)
- `stream/` - Streaming utilities
- `tool/` - Tool definitions (used by multiple SPIs)

**Total**: ~15-20 files

### 2. Move to `gollek-spi-plugin`
**Purpose**: Plugin lifecycle and extension points

**Packages to move**:
- `plugin/` - All plugin interfaces (GollekPlugin, PluginRegistry, etc.)

**Files to move**:
- GollekPlugin.java
- PluginContext.java
- PluginRegistry.java
- PluginState.java
- PluginHealth.java
- PluginException.java
- PromptPlugin.java
- StreamingPlugin.java
- ObservabilityPlugin.java
- ReasoningPlugin.java
- BackpressureMode.java
- (All other plugin/*.java files)

**Total**: ~12-15 files

### 3. Move to `gollek-spi-provider`
**Purpose**: Provider management and routing

**Packages to move**:
- `provider/` - All provider interfaces
- `routing/` - Routing logic

**Files to move**:
- LLMProvider.java
- StreamingProvider.java
- ProviderRegistry.java
- ProviderInfo.java
- ProviderConfig.java
- ProviderMetadata.java
- ProviderCapabilities.java
- ProviderFeature.java
- ProviderHealth.java
- ProviderMetrics.java
- ProviderRequest.java
- ProviderResponse.java
- ProviderCandidate.java
- ProviderDescriptor.java
- ProviderContext.java
- RoutingDecision.java
- RoutingContext.java
- ProviderRoutingContext.java
- (All other provider/*.java and routing/*.java files)

**Total**: ~20-25 files

### 4. Move to `gollek-spi-model`
**Purpose**: Model management and storage

**Packages to move**:
- `model/` - Model interfaces
- `storage/` - Storage interfaces

**Files to move**:
- ModelRegistry.java
- ModelManifest.java
- ModelRef.java
- ModelFormat.java
- DeviceType.java
- HealthStatus.java
- ComputeRequirements.java
- ResourceRequirements.java
- ResourceMetrics.java
- StorageRequirements.java
- SupportedDevice.java
- ModelStorageService.java
- (All other model/*.java and storage/*.java files)

**Total**: ~15-18 files

### 5. Move to `gollek-spi-inference`
**Purpose**: Inference execution

**Packages to move**:
- `inference/` - Inference interfaces
- `execution/` - Execution status

**Files to move**:
- All inference/*.java files
- All execution/*.java files

**Total**: ~8-10 files

---

## Migration Strategy

### Phase 1: Prepare Target Modules
1. Update POMs of target SPI modules with proper dependencies
2. Create package structure in target modules
3. Ensure `gollek-spi` (common) is built first

### Phase 2: Move Code
1. Move plugin code → `gollek-spi-plugin`
2. Move provider code → `gollek-spi-provider`
3. Move model code → `gollek-spi-model`
4. Move inference code → `gollek-spi-inference`
5. Keep common code in `gollek-spi`

### Phase 3: Update Dependencies
1. Update `gollek-spi-plugin/pom.xml` to depend on `gollek-spi`
2. Update `gollek-spi-provider/pom.xml` to depend on `gollek-spi`
3. Update `gollek-spi-model/pom.xml` to depend on `gollek-spi`
4. Update `gollek-spi-inference/pom.xml` to depend on `gollek-spi` and `gollek-spi-model`

### Phase 4: Update Import Statements
1. Update imports in moved files
2. Update imports in files that reference moved code
3. Verify no circular dependencies

### Phase 5: Update External Dependencies
1. Update all modules that depend on `gollek-spi` to use specific SPI modules
2. OR keep `gollek-spi` as aggregator that re-exports all SPIs

### Phase 6: Test and Validate
1. Build all SPI modules
2. Run tests
3. Verify dependent modules still compile

---

## Dependency Graph

```
gollek-spi (common)
    ↑
    ├── gollek-spi-model
    ├── gollek-spi-plugin
    └── gollek-spi-provider
            ↑
            └── gollek-spi-inference
```

---

## Recommendation

**Option A: Keep `gollek-spi` as Aggregator** (RECOMMENDED)
- `gollek-spi` contains common code + re-exports all other SPIs
- Users can depend on just `gollek-spi` to get everything
- Simpler for most users
- Backward compatible with existing code

**Option B: Split Completely**
- `gollek-spi` contains ONLY common code
- Users must depend on specific SPI modules they need
- More granular control
- Requires updating all 45+ dependent modules

**Recommendation**: Start with **Option A** for backward compatibility, then gradually migrate to **Option B** over time.

---

## Next Steps

1. ✅ Decide on Option A or B
2. ⏳ Execute migration in phases
3. ⏳ Update documentation
4. ⏳ Test thoroughly

---

**Status**: Planning Complete  
**Ready for**: Phase 1 Execution
