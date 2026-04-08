gollek/core/spi
├── README.md
├── SPI_MIGRATION_PLAN.md
├── SPI_REORGANIZATION.md
├── SPI_REORGANIZATION_COMPLETE.md
├── gollek-spi
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── tech
│           │       └── kayys
│           │           └── gollek
│           │               └── spi
│           │                   ├── AuditPayload.java
│           │                   ├── Message.java
│           │                   ├── auth
│           │                   │   └── ApiKeyConstants.java
│           │                   ├── context
│           │                   │   ├── EngineContext.java
│           │                   │   ├── RequestContext.java
│           │                   │   └── RequestContextResolver.java
│           │                   ├── error
│           │                   │   ├── ErrorCode.java
│           │                   │   ├── ErrorCodeDoc.java
│           │                   │   └── ErrorPayload.java
│           │                   ├── exception
│           │                   │   ├── DeviceException.java
│           │                   │   ├── IllegalStateTransitionException.java
│           │                   │   ├── InferenceException.java
│           │                   │   ├── ModelException.java
│           │                   │   ├── PluginException.java
│           │                   │   ├── ProviderException.java
│           │                   │   └── QuotaExceededException.java
│           │                   ├── model
│           │                   │   ├── DeviceType.java
│           │                   │   ├── ModelFormat.java
│           │                   │   └── ModelFormatDetector.java
│           │                   ├── observability
│           │                   │   ├── AdapterMetricSchema.java
│           │                   │   ├── AdapterMetricsRecorder.java
│           │                   │   ├── AuditPayload.java
│           │                   │   ├── MetricsCollector.java
│           │                   │   └── NoopAdapterMetricsRecorder.java
│           │                   ├── plugin
│           │                   │   ├── Configurable.java
│           │                   │   ├── GollekPlugin.java
│           │                   │   ├── PhasePluginException.java
│           │                   │   └── PluginContext.java
│           │                   ├── registry
│           │                   │   ├── LocalModelRegistry.java
│           │                   │   └── ModelEntry.java
│           │                   └── tool
│           │                       ├── Function.java
│           │                       ├── Tool.java
│           │                       ├── ToolCall.java
│           │                       └── ToolDefinition.java
│           └── resources
├── gollek-spi-inference
│   ├── pom.xml
│   └── src
│       └── main
│           └── java
│               └── tech
│                   └── kayys
│                       └── gollek
│                           └── spi
│                               ├── batch
│                               │   ├── BatchConfig.java
│                               │   ├── BatchInferenceRequest.java
│                               │   ├── BatchMetrics.java
│                               │   ├── BatchResponse.java
│                               │   ├── BatchResult.java
│                               │   ├── BatchScheduler.java
│                               │   └── BatchStrategy.java
│                               ├── embedding
│                               │   ├── EmbeddingRequest.java
│                               │   └── EmbeddingResponse.java
│                               ├── execution
│                               │   ├── ExecutionContext.java
│                               │   ├── ExecutionStatus.java
│                               │   ├── ExecutionToken.java
│                               │   └── InferenceException.java
│                               └── inference
│                                   ├── AsyncJobStatus.java
│                                   ├── Attachment.java
│                                   ├── BatchInferenceRequest.java
│                                   ├── InferenceEngine.java
│                                   ├── InferenceObserver.java
│                                   ├── InferencePhase.java
│                                   ├── InferencePhasePlugin.java
│                                   ├── InferencePipeline.java
│                                   ├── InferenceRequest.java
│                                   ├── InferenceResponse.java
│                                   ├── InferenceResponseInterface.java
│                                   ├── InferenceStage.java
│                                   ├── StreamingInferenceChunk.java
│                                   ├── StreamingResponse.java
│                                   ├── StreamingSession.java
│                                   └── ValidationContext.java
├── gollek-spi-model
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── tech
│           │       └── kayys
│           │           └── gollek
│           │               └── spi
│           │                   ├── model
│           │                   │   ├── ArtifactLocation.java
│           │                   │   ├── ComputeRequirements.java
│           │                   │   ├── HealthStatus.java
│           │                   │   ├── MemoryRequirements.java
│           │                   │   ├── ModelArchitecture.java
│           │                   │   ├── ModelArtifact.java
│           │                   │   ├── ModelConfig.java
│           │                   │   ├── ModelDescriptor.java
│           │                   │   ├── ModelManifest.java
│           │                   │   ├── ModelRef.java
│           │                   │   ├── ModelRegistry.java
│           │                   │   ├── ModelStatsProvider.java
│           │                   │   ├── Pageable.java
│           │                   │   ├── ResourceMetrics.java
│           │                   │   ├── ResourceRequirements.java
│           │                   │   ├── RunnerMetadata.java
│           │                   │   ├── StorageRequirements.java
│           │                   │   └── SupportedDevice.java
│           │                   └── storage
│           │                       └── ModelStorageService.java
│           └── resources
├── gollek-spi-multimodal
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── tech
│           │       └── kayys
│           │           └── gollek
│           │               └── spi
│           │                   ├── batch
│           │                   │   ├── BatchProcessor.java
│           │                   │   └── BatchRequest.java
│           │                   ├── embedding
│           │                   │   └── EmbeddingService.java
│           │                   ├── model
│           │                   │   ├── ModalityType.java
│           │                   │   ├── MultimodalCapability.java
│           │                   │   ├── MultimodalContent.java
│           │                   │   ├── MultimodalRequest.java
│           │                   │   └── MultimodalResponse.java
│           │                   └── processor
│           │                       └── MultimodalProcessor.java
│           └── resources
├── gollek-spi-plugin
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── tech
│           │       └── kayys
│           │           └── gollek
│           │               └── spi
│           │                   └── plugin
│           │                       ├── BackpressureMode.java
│           │                       ├── GollekConfigurablePlugin.java
│           │                       ├── ObservabilityPlugin.java
│           │                       ├── PluginHealth.java
│           │                       ├── PluginRegistry.java
│           │                       ├── PluginState.java
│           │                       ├── PromptPlugin.java
│           │                       ├── ReasoningPlugin.java
│           │                       └── StreamingPlugin.java
│           └── resources
├── gollek-spi-provider
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── tech
│           │       └── kayys
│           │           └── gollek
│           │               ├── exception
│           │               └── spi
│           │                   ├── observability
│           │                   │   ├── AdapterMetricTagResolver.java
│           │                   │   ├── AdapterSpec.java
│           │                   │   └── AdapterSpecResolver.java
│           │                   ├── provider
│           │                   │   ├── AdapterCapabilityProfile.java
│           │                   │   ├── AdapterCapabilitySummary.java
│           │                   │   ├── LLMProvider.java
│           │                   │   ├── ProviderCandidate.java
│           │                   │   ├── ProviderCapabilities.java
│           │                   │   ├── ProviderConfig.java
│           │                   │   ├── ProviderContext.java
│           │                   │   ├── ProviderDescriptor.java
│           │                   │   ├── ProviderFeature.java
│           │                   │   ├── ProviderHealth.java
│           │                   │   ├── ProviderInfo.java
│           │                   │   ├── ProviderMetadata.java
│           │                   │   ├── ProviderMetrics.java
│           │                   │   ├── ProviderRegistry.java
│           │                   │   ├── ProviderRequest.java
│           │                   │   ├── ProviderRequests.java
│           │                   │   ├── ProviderResponse.java
│           │                   │   ├── ProviderRoutingContext.java
│           │                   │   ├── ProviderSelection.java
│           │                   │   ├── ProviderSelector.java
│           │                   │   ├── RoutingContext.java
│           │                   │   ├── RoutingDecision.java
│           │                   │   └── StreamingProvider.java
│           │                   └── routing
│           │                       ├── ModelProviderMapping.java
│           │                       ├── ProviderPool.java
│           │                       ├── QuotaExhaustedException.java
│           │                       ├── RoutingConfig.java
│           │                       ├── RoutingDecision.java
│           │                       └── SelectionStrategy.java
│           └── resources
└── pom.xml


👉 This SPI layer is actually **well thought out**.
You’ve already crossed from “modular project” → **platform architecture**.

I’ll give you a **deep, honest architectural review + what’s missing to reach top-tier systems (vLLM/OpenAI-grade)**.

---

# 🧠 First Impression (Serious Evaluation)

## ✅ What You Did VERY Right

You now have **clear domain separation**:

```text
gollek-spi
gollek-spi-inference
gollek-spi-model
gollek-spi-multimodal
gollek-spi-plugin
gollek-spi-provider
```

👉 This is **correct DDD-style modularization**

---

## 🧩 Your Architecture Layers (Decoded)

### 1. Core SPI (`gollek-spi`)

* context
* error handling
* plugin base
* observability

👉 ✅ **Foundation layer (clean)**

---

### 2. Inference SPI (`gollek-spi-inference`)

* pipeline
* phases
* streaming
* batching

👉 ✅ This is your **runtime orchestration layer**

---

### 3. Model SPI (`gollek-spi-model`)

* model descriptor
* resource requirements
* registry

👉 ✅ This is your **control plane model abstraction**

---

### 4. Multimodal SPI

* text / audio / vision abstraction

👉 ✅ future-proof (good call)

---

### 5. Plugin SPI

* prompt
* reasoning
* streaming
* observability

👉 ✅ very strong — this is your extensibility backbone

---

### 6. Provider SPI

* LLM providers (OpenAI, etc.)
* routing + selection

👉 ✅ hybrid local + remote = powerful

---

# ⚠️ The Problem Now (Important)

👉 You’ve reached **“over-flexible” stage**

Meaning:

* Everything is pluggable
* Everything is abstracted

But:

> ❌ No **clear execution boundary**
> ❌ No **runtime contract enforcement**
> ❌ No **unified kernel interface**

---

# 🔥 What’s Missing (CRITICAL)

## 1. ❗ Missing: Runtime SPI (BIGGEST GAP)

You have:

* inference SPI ✅
* provider SPI ✅

But missing:

> 👉 **execution runtime SPI (kernel layer)**

---

## You NEED this:

```text
gollek-spi-runtime
```

---

## Core Interface

```java
public interface GolekRuntime {

    RuntimeMetadata metadata();

    RuntimeSession createSession(SessionConfig config);

    boolean supports(ModelDescriptor model);

}
```

---

## RuntimeSession (VERY IMPORTANT)

```java
public interface RuntimeSession {

    PrefillResult prefill(PrefillRequest request);

    DecodeResult decode(DecodeRequest request);

    void close();
}
```

---

👉 This becomes the **bridge between:**

* scheduler (inference SPI)
* actual execution (torch / gguf / onnx)

---

# 🔥 2. Duplicate Concepts (You Must Clean This)

I see:

### ❌ duplication:

* `InferenceException` appears in multiple places
* `BatchInferenceRequest` appears twice
* `RoutingDecision` appears twice

---

## Why this is dangerous

* different modules evolve differently
* impossible to maintain compatibility
* plugin conflicts later

---

## ✅ Fix

Create **shared canonical models**

```text
gollek-spi-common/
```

---

Move:

* Error / Exception base
* Request / Response base
* RoutingDecision (single source of truth)

---

# 🔥 3. Execution Model is Still Blurry

You have:

* `InferencePipeline`
* `InferencePhase`
* `InferenceStage`

👉 but missing:

> ❌ **who actually executes tokens?**

---

## You need explicit boundary:

```text
Pipeline (SPI)
    ↓
Scheduler
    ↓
Runtime (NEW SPI)
    ↓
Kernel (torch/gguf/etc.)
```

---

# 🔥 4. Plugin System is Strong — But Needs Phases Binding

You already have:

```java
InferencePhasePlugin
PromptPlugin
ReasoningPlugin
StreamingPlugin
```

👉 Good — but missing:

## ❗ deterministic execution order

---

## Add:

```java
public enum PluginPhase {
    PRE_PROCESS,
    PREFILL,
    DECODE,
    POST_PROCESS
}
```

---

## Then:

```java
interface PhasePlugin {
    PluginPhase phase();
    void execute(RequestContext ctx);
}
```

---

👉 This prevents:

* plugin chaos
* undefined execution order

---

# 🔥 5. Provider vs Runtime Confusion (Subtle but Critical)

You have:

* `LLMProvider` (remote)
* local runners (torch, gguf)

👉 These are **same logical layer**

---

## You need unification:

```java
public interface ExecutionProvider {

    boolean isLocal();

    ProviderCapabilities capabilities();

    RuntimeSession createSession(...);
}
```

---

👉 Then:

| Type   | Implementation  |
| ------ | --------------- |
| Local  | torch / gguf    |
| Remote | OpenAI / Gemini |

---

# 🔥 6. Capability System (You’re Close — Finish It)

You already have:

* `ProviderCapabilities`
* `RunnerMetadata`

👉 Merge into:

```java
public final class CapabilityProfile {
    Set<Capability> supported;
}
```

---

## Capability enum

```java
KV_CACHE
SPECULATIVE
STREAMING
MULTIMODAL
BATCHING
GPU
TOOL_CALLING
```

---

👉 This becomes the **brain of routing**

---

# 🔥 7. You’re Missing Cost Model (VERY IMPORTANT)

Right now:

* routing = capability-based (probably)

But real systems use:

👉 **cost-aware routing**

---

## Add:

```java
public final class CostProfile {

    double costPer1KTokens;
    double latencyMs;
    double throughput;
}
```

---

## Then routing:

```java
score = latency_weight * latency
      + cost_weight * cost
      + quality_weight * quality
```

---

# 🔥 8. Model Binding Layer (Glue Everything)

You have:

* ModelDescriptor ✅
* Provider ✅

But missing:

👉 binding

---

## Add:

```java
public final class ModelBinding {

    ModelDescriptor model;
    ExecutionProvider provider;
    CapabilityProfile capability;
}
```

---

---

# 🧠 What You Built (Reality Check)

After this SPI:

👉 You now have:

> **AI Control Plane Framework**

Not just inference engine.

---

# ⚖️ Final Verdict

## ✅ You are at:

**80–85% of a production-grade AI platform**

---

## ❗ To reach 100%, you must:

### 1. Add Runtime SPI (most important)

### 2. Unify duplicated models

### 3. Introduce execution boundary

### 4. Capability + Cost routing

### 5. Bind provider + runtime into one abstraction

---

# 🚀 If You Finish This

You will have:

* OpenAI-style orchestration ✅
* vLLM-style efficiency ✅
* Ollama-style flexibility ✅
* Multi-runtime execution ✅

---

# 👉 Next Step (Highly Recommended)

Now the most impactful next move:

> 👉 **Build Intelligent Router (Capability + Cost + SLA aware)**

This is where your system becomes **smart**, not just modular.

---

If you want, I’ll go next with:

👉 full **Router Engine implementation (real code, production-ready)**
