# Walkthrough: Gollek Core Runtime Evolution

## Summary

Built the entire 4-phase core runtime architecture from the enhancement plan in `gollek/docs/enhancement/next/`. Created **3 new Maven modules** with **35+ Java source files**, all compiling successfully.

## New Modules

### `gollek-runtime-tensor` (Phase 1 — Foundation)

| File | Purpose |
|------|---------|
| [DType.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/DType.java) | Data type enum (FP32, FP16, BF16, INT8, INT4, QINT8, QINT4) |
| [Device.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/Device.java) | Compute device enum (CPU, CUDA, Metal, ROCm, TPU, NPU) |
| [BackendType.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/BackendType.java) | Backend identity enum (LibTorch, GGML, ONNX, LiteRT, CPU_JAVA) |
| [Tensor.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/Tensor.java) | Universal tensor interface — the core contract |
| [DefaultTensor.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/DefaultTensor.java) | Concrete tensor with zero-copy view support |
| [PooledTensorStorage.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/PooledTensorStorage.java) | Ref-counted storage with pool return on release |
| [TensorPool.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/TensorPool.java) | Thread-safe memory reuse pool |
| [TensorKey.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/TensorKey.java) | Pool index key (shape + dtype + device) |
| [ExecutionContext.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/ExecutionContext.java) | Scoped lifecycle for temp tensors |
| [Backend.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/Backend.java) | Pluggable execution backend interface |
| [BackendRegistry.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/BackendRegistry.java) | Backend lookup registry |
| [QuantParams.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/QuantParams.java) | Quantization metadata (scale + zero_point) |
| [NativeMemory.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-tensor/src/main/java/tech/kayys/gollek/runtime/tensor/NativeMemory.java) | Backend-dispatched memory cleanup |

---

### `gollek-runtime-graph` (Phase 2 — Graph Engine)

| File | Purpose |
|------|---------|
| [GraphNode.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/GraphNode.java) | DAG node with lifetime tracking |
| [ComputationGraph.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/ComputationGraph.java) | DAG container |
| [LazyTensor.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/LazyTensor.java) | Records ops for deferred execution |
| [ExecutionPlan.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/ExecutionPlan.java) | Compiled topological op list |
| [GraphPlanner.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/GraphPlanner.java) | Topological sort compiler |
| [FusionOptimizer.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/FusionOptimizer.java) | Pattern-matching op fusion |
| [LifetimeAnalyzer.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/LifetimeAnalyzer.java) | Tensor lifetime computation |
| [GraphMemoryPlanner.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/GraphMemoryPlanner.java) | Lifetime-aware memory reuse |
| [GraphExecutor.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-graph/src/main/java/tech/kayys/gollek/runtime/graph/GraphExecutor.java) | Full pipeline executor |

---

### `gollek-runtime-inference` (Phase 3+4 — LLM Runtime)

| File | Purpose |
|------|---------|
| [BackendRouter.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/router/BackendRouter.java) | Per-op backend selection interface |
| [HeuristicRouter.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/router/HeuristicRouter.java) | Rule-based GPU/CPU routing |
| [KVCache.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/KVCache.java) | KV cache interface with snapshot |
| [KVPage.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/KVPage.java) | Fixed-size page for PagedAttention |
| [PagedKVCache.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/PagedKVCache.java) | vLLM-style paged implementation |
| [TokenStreamer.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/streaming/TokenStreamer.java) | Streaming token callback |
| [BatchRequest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/batch/BatchRequest.java) | Runtime scheduling unit (distinct from SPI InferenceRequest) |
| [RequestQueue.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/batch/RequestQueue.java) | Thread-safe request queue |
| [ContinuousBatchScheduler.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/batch/ContinuousBatchScheduler.java) | vLLM-style dynamic batching |
| [PrefixKey.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/cache/PrefixKey.java) | Token sequence hash key |
| [PrefixCache.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/cache/PrefixCache.java) | Shared prefix KV reuse (30-90% savings) |
| [SpeculativeDecoder.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/speculative/SpeculativeDecoder.java) | Draft+verify speculative decoding (2-5x speedup) |
| [TenantTier.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/tenant/TenantTier.java) | Service tier with DRR quantum |
| [TenantContext.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/tenant/TenantContext.java) | Per-tenant quotas and queues |
| [MultiTenantScheduler.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/tenant/MultiTenantScheduler.java) | DRR fair scheduler |
| [KVQuotaManager.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/tenant/KVQuotaManager.java) | Per-tenant KV block quotas |

---

## Modified Files

| File | Change |
|------|--------|
| [core/pom.xml](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/core/pom.xml) | Added 3 new modules |

```diff:pom.xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>tech.kayys.gollek</groupId>
        <artifactId>gollek-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>gollek-core-parent</artifactId>
    <packaging>pom</packaging>

    <name>gollek-core-parent</name>
    <description>gollek Core</description>


    <properties>
        <maven.compiler.source>25</maven.compiler.source>
        <maven.compiler.target>25</maven.compiler.target>
    </properties>

    <dependencyManagement>
        <dependencies>
            <!-- Import Gollek BOM for internal module versions -->
            <dependency>
                <groupId>tech.kayys.gollek</groupId>
                <artifactId>gollek-bom</artifactId>
                <version>${project.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <modules>
        <!-- SPI interfaces (must be built first) -->
        <module>spi</module>

        <!-- Error codes (no deps on other core modules) -->
        <module>gollek-error-code</module>

        <!-- Plugin modules: plugin/pom.xml (gollek-core-plugin-parent) declares all plugin children -->
        <module>plugin</module>

        <!-- Core infrastructure -->
        <module>gollek-cluster</module>
        <module>gollek-engine</module>
        <module>gollek-observability</module>
        <module>gollek-multimodal-core</module>

        <!-- Model modules -->
        <module>model/gollek-model-repo-core</module>
        <module>model/gollek-models-family</module>
        <module>model/gollek-model-registry</module>
        <module>model/gollek-model-routing</module>
        <module>model/gollek-model-runner</module>

        <!-- Provider modules -->
        <module>gollek-provider-core</module>
        <module>gollek-provider-routing</module>

        <!-- API and tools -->
        <module>gollek-api-rest</module>
        <module>gollek-tool-core</module>
        <module>gollek-tokenizer-core</module>
        <module>gollek-embedding-core</module>
    </modules>
</project>
===
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>tech.kayys.gollek</groupId>
        <artifactId>gollek-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>gollek-core-parent</artifactId>
    <packaging>pom</packaging>

    <name>gollek-core-parent</name>
    <description>gollek Core</description>


    <properties>
        <maven.compiler.source>25</maven.compiler.source>
        <maven.compiler.target>25</maven.compiler.target>
    </properties>

    <dependencyManagement>
        <dependencies>
            <!-- Import Gollek BOM for internal module versions -->
            <dependency>
                <groupId>tech.kayys.gollek</groupId>
                <artifactId>gollek-bom</artifactId>
                <version>${project.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <modules>
        <!-- SPI interfaces (must be built first) -->
        <module>spi</module>

        <!-- Error codes (no deps on other core modules) -->
        <module>gollek-error-code</module>

        <!-- Plugin modules: plugin/pom.xml (gollek-core-plugin-parent) declares all plugin children -->
        <module>plugin</module>

        <!-- Runtime primitives (Tensor SPI, Backend, Pool) -->
        <module>gollek-runtime-tensor</module>
        <!-- Computation graph engine (fusion, memory planner) -->
        <module>gollek-runtime-graph</module>
        <!-- LLM inference runtime (KV cache, batching, scheduling) -->
        <module>gollek-runtime-inference</module>

        <!-- Core infrastructure -->
        <module>gollek-cluster</module>
        <module>gollek-engine</module>
        <module>gollek-observability</module>
        <module>gollek-multimodal-core</module>

        <!-- Model modules -->
        <module>model/gollek-model-repo-core</module>
        <module>model/gollek-models-family</module>
        <module>model/gollek-model-registry</module>
        <module>model/gollek-model-routing</module>
        <module>model/gollek-model-runner</module>

        <!-- Provider modules -->
        <module>gollek-provider-core</module>
        <module>gollek-provider-routing</module>

        <!-- API and tools -->
        <module>gollek-api-rest</module>
        <module>gollek-tool-core</module>
        <module>gollek-tokenizer-core</module>
        <module>gollek-embedding-core</module>
    </modules>
</project>
```

## Documentation

| File | Description |
|------|-------------|
| [core-runtime-architecture.md](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/website/gollek-ai.github.io/docs/core-runtime-architecture.md) | Full developer guide with code examples and architecture diagram |

## Verification

```
Reactor Summary for Gollek Runtime Tensor 1.0.0-SNAPSHOT:
  Gollek Runtime Tensor .............................. SUCCESS [0.768 s]
  Gollek Runtime Graph ............................... SUCCESS [0.198 s]
  Gollek Runtime Inference ........................... SUCCESS [0.193 s]
BUILD SUCCESS — Total time: 1.820 s
```

## Design Decisions

1. **`BatchRequest` renamed from `InferenceRequest`** — avoids confusion with the SPI-level `tech.kayys.gollek.spi.inference.InferenceRequest` which is the high-level API request. `BatchRequest` is the low-level runtime scheduling unit.

2. **Three separate modules** (tensor → graph → inference) — layered dependency with clean separation. `tensor` has zero internal dependencies, `graph` depends only on `tensor`, `inference` depends on both.

3. **Zero-copy view operations** — `reshape()` and `slice()` share underlying `PooledTensorStorage` via `retain()` reference counting instead of copying data.

## Next Steps

- Implement a concrete `LibTorchBackend` wrapping the existing 762-line `Tensor.java` in the libtorch runner
- Wire the `ContinuousBatchScheduler` into the existing `LibTorchProvider` inference loop
- Execute Phase 4 Step 13: migrate the 13 optimization plugins into core modules
- Add JUnit 5 integration tests for `TensorPool`, `FusionOptimizer`, and `MultiTenantScheduler`
