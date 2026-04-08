# Gollek SafeTensor Module Reorganization

## 🎯 New Module Structure

The SafeTensor module has been reorganized into focused, concern-specific submodules for better maintainability and clearer dependencies.

```
inference-gollek/extension/runner/safetensor/
├── pom.xml (parent)
├── gollek-safetensor-api/          # Public API & SPI interfaces
├── gollek-safetensor-core/         # Core inference engine
├── gollek-safetensor-loader/       # Safetensor file format loader
├── gollek-safetensor-text/         # Text model architectures & inference
├── gollek-safetensor-audio/        # Audio (TTS/STT) support
├── gollek-safetensor-vision/       # Vision & multimodal support
├── gollek-safetensor-rag/          # RAG & vector database integration
├── gollek-safetensor-tooling/      # Tool calling & function support
├── gollek-safetensor-session/      # Conversation session management
├── gollek-safetensor-routing/      # Model routing & multi-tenancy
└── gollek-safetensor-integration/  # Integration tests & examples
```

## 📦 Module Responsibilities

### 1. gollek-safetensor-api
**Purpose:** Public API, SPI interfaces, and common types

**Contains:**
- `ModelArchitecture` SPI interface
- `ModelConfig` record
- `GenerationConfig` 
- `InferenceResponse` 
- `InferenceRequest`
- Common exceptions and utilities

**Dependencies:** None (foundational module)

---

### 2. gollek-safetensor-loader
**Purpose:** Safetensor file format parsing and weight loading

**Contains:**
- `SafetensorHeaderParser`
- `SafetensorFFMLoader`
- `SafetensorDType` & converters
- `SafetensorTensor`
- `SafetensorShardIndex` & `SafetensorShardLoader`
- `SafetensorWeightBridge`
- `SafetensorLoadResult` & `SafetensorLoadCache`
- `SafetensorMetrics`
- `SafetensorValidator`

**Dependencies:** 
- `gollek-safetensor-api`
- `gollek-runner-libtorch`

---

### 3. gollek-safetensor-core
**Purpose:** Core inference engine and generation

**Contains:**
- `DirectInferenceEngine`
- `DirectForwardPass`
- `FlashAttentionKernel`
- `SlidingWindowAttention`
- `KVCacheManager`
- `BeamSearchDecoder`
- `LogprobsEngine`
- `MoeForwardPass`
- `ModelWarmupService`
- `QuantizationEngine`

**Lifecycle Management:**
- `ModelHotSwapManager`
- `MemoryPressureMonitor`
- `GracefulShutdownHandler`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-loader`
- `gollek-runner-libtorch`

---

### 4. gollek-safetensor-text
**Purpose:** Text model architectures and tokenization

**Contains:**
- `TextModelFamilies` (all text architectures)
- `AdditionalModelFamilies`
- `ModelArchitectureDetectorV2`
- `HuggingFaceTokenizer`
- `TokenizerFactory`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 5. gollek-safetensor-audio
**Purpose:** Audio processing (TTS and STT)

**Contains:**
- `SpeechT5Engine` (TTS)
- `WhisperEngine` (STT)
- `AudioResource` (REST endpoints)
- `AudioConfig`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 6. gollek-safetensor-vision
**Purpose:** Vision encoding and multimodal inference

**Contains:**
- `VisionEncoder` (CLIP ViT)
- `MultimodalInferenceEngine`
- `MultimodalModelFamilies`
- `VisionModelRegistry`
- `VisionConfig`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 7. gollek-safetensor-rag
**Purpose:** RAG pipeline and vector database integration

**Contains:**
- `QdrantVectorStore`
- `RagPipeline`
- `RagResource` (REST endpoints)
- `EmbeddingInferenceEngine`
- `EmbeddingsResource`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`
- `gollek-safetensor-text`

---

### 8. gollek-safetensor-tooling
**Purpose:** Tool calling and function execution

**Contains:**
- `ToolCallingEngine`
- `ToolDefinition`
- `ToolCall`
- `ToolCallingEngineTest`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 9. gollek-safetensor-session
**Purpose:** Conversation session management

**Contains:**
- `ConversationSessionManager`
- `ConversationSession`
- `SessionConfig`
- `ConversationSessionManagerTest`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 10. gollek-safetensor-routing
**Purpose:** Model routing and multi-tenancy

**Contains:**
- `ModelRouter`
- `ModelRouterResource`
- `MultiTenantQuotaService`
- `LoraAdapterRouter`
- `OpenAiCompatibleResource`

**Dependencies:**
- `gollek-safetensor-api`
- `gollek-safetensor-core`

---

### 11. gollek-safetensor-integration
**Purpose:** Integration tests and examples

**Contains:**
- `ModelFamilyIntegrationTest`
- End-to-end workflow tests
- Example configurations

**Dependencies:**
- All other modules (test scope)

---

## 🔧 Module Dependency Graph

```
                    ┌─────────────┐
                    │     API     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Loader  │      │  Core   │      │  Text   │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         │         ┌──────┴──────┐        │
         │         │             │        │
    ┌────▼────┐ ┌──▼──┐   ┌────▼────┐   │
    │  Audio  │ │Vision│   │   RAG   │◄──┘
    └─────────┘ └─────┘   └─────────┘
    
    ┌──────────┐  ┌─────────┐  ┌─────────┐
    │ Tooling  │  │ Session │  │ Routing │
    └──────────┘  └─────────┘  └─────────┘
```

---

## 📊 File Distribution

| Module | Java Files | Key Components |
|--------|------------|----------------|
| **api** | 5-8 | SPI interfaces, configs |
| **loader** | 15-20 | Safetensor parsing |
| **core** | 20-25 | Inference engine, attention, MoE |
| **text** | 20-25 | Model families, tokenizer |
| **audio** | 3-5 | SpeechT5, Whisper |
| **vision** | 5-7 | Vision encoder, multimodal |
| **rag** | 5-7 | Qdrant, RAG pipeline |
| **tooling** | 3-5 | Tool calling |
| **session** | 2-3 | Conversation management |
| **routing** | 5-7 | Model router, quotas |
| **integration** | 5-10 | Integration tests |

**Total:** ~90-120 Java files

---

## 🚀 Benefits of Reorganization

1. **Clear Separation of Concerns:** Each module has a single, well-defined responsibility
2. **Better Dependency Management:** Modules only depend on what they need
3. **Easier Testing:** Smaller, focused modules are easier to test
4. **Improved Maintainability:** Changes are isolated to specific modules
5. **Flexible Deployment:** Can include only needed modules
6. **Better Documentation:** Each module can have focused documentation
7. **Parallel Development:** Teams can work on different modules independently

---

## 📝 Migration Steps

1. **Backup current structure**
2. **Create new module directories**
3. **Move files to appropriate modules**
4. **Update POM files**
5. **Update package declarations if needed**
6. **Fix import statements**
7. **Run tests to verify**
8. **Update documentation**

---

## 🔨 Build Commands

```bash
# Build all modules
mvn clean install -pl inference-gollek/extension/runner/safetensor -am

# Build specific module
mvn clean install -pl inference-gollek/extension/runner/safetensor/gollek-safetensor-core -am

# Run tests for all modules
mvn test -pl inference-gollek/extension/runner/safetensor

# Skip tests for quick build
mvn clean install -pl inference-gollek/extension/runner/safetensor -am -DskipTests
```

---

## 📋 Module POM Template

Each module POM should follow this structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>tech.kayys.gollek</groupId>
        <artifactId>gollek-safetensor-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>gollek-safetensor-{module-name}</artifactId>
    <name>Gollek SafeTensor {Module Name}</name>
    <description>{Module description}</description>

    <dependencies>
        <!-- Internal dependencies -->
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-safetensor-api</artifactId>
            <version>${project.version}</version>
        </dependency>
        
        <!-- External dependencies -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-arc</artifactId>
        </dependency>
        
        <!-- Test dependencies -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-junit5</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
```

---

## ✅ Verification Checklist

- [ ] All modules have POM files
- [ ] Dependencies are correctly declared
- [ ] No circular dependencies
- [ ] All files moved to correct modules
- [ ] Package declarations updated
- [ ] Import statements fixed
- [ ] Tests pass for all modules
- [ ] Documentation updated
- [ ] Build script updated

---

**Reorganization Date:** 2026-03-20  
**Status:** In Progress  
**Next Action:** Complete file migration and POM updates
