# Gollek Framework Enhancement Plan v1.0 — Part 2

**Document Version:** 1.0  
**Date:** April 7, 2026  
**Continues From:** `framework-enhancement-plan-01.md`  
**Status:** 🎯 Planning Phase

---

## Overview

Part 1 covered the high-level 4-phase roadmap. This document dives deeper into:

1. **SDK Unification** — clean developer surface across all runtimes
2. **NLP Ecosystem** — tokenization, training, fine-tuning
3. **Hardware Acceleration** — GPU/Metal binding strategy
4. **Implementation Details** — concrete class designs and APIs
5. **Integration with Wayang Platform** — agent, RAG, workflow hooks

---

## 5. SDK Unification (Critical Gap)

### Problem

Currently the SDK has fragmented surfaces:
- Different runners (GGUF, ONNX, LiteRT, LibTorch) expose different APIs
- No unified session abstraction
- KV cache / streaming / batching not unified
- Hard to plug into agent framework

### Target: Unified Client API

```java
// Single entry point regardless of backend
GollekClient client = GollekClient.builder()
    .endpoint("http://localhost:8080")
    .model("qwen2-7b")
    .build();

// Streaming-first
client.generateStream(Prompt.text("Explain attention"))
      .onToken(System.out::print)
      .onComplete(result -> log(result.metrics()));

// Batch inference
List<GenerationResult> results = client.generateBatch(prompts);

// Embeddings
float[] embedding = client.embed("Hello world");
```

### New Module: `gollek-sdk-unified`

```
gollek/sdk/lib/gollek-sdk-unified/
├── GollekClient.java              # Main entry point
├── GollekClientBuilder.java
├── session/
│   ├── InferenceSession.java      # Session abstraction
│   ├── TrainingSession.java
│   └── SessionConfig.java
├── api/
│   ├── GenerationRequest.java
│   ├── GenerationResult.java
│   ├── GenerationStream.java      # Streaming response
│   ├── EmbeddingRequest.java
│   └── BatchRequest.java
├── routing/
│   ├── BackendRouter.java         # Routes to GGUF/ONNX/etc.
│   ├── FeatureNegotiator.java     # Detects backend capabilities
│   └── LoadBalancer.java
└── README.md
```

### Core Interfaces

**`GollekClient.java`**
```java
public interface GollekClient {
    GenerationResult generate(GenerationRequest request);
    GenerationStream generateStream(GenerationRequest request);
    List<GenerationResult> generateBatch(List<GenerationRequest> requests);
    float[] embed(String text);
    ModelInfo modelInfo();
    void close();

    static GollekClientBuilder builder() { ... }
}
```

**`InferenceSession.java`**
```java
public interface InferenceSession extends AutoCloseable {
    GenerationResult run(GenerationRequest request);
    GenerationStream stream(GenerationRequest request);
    SessionMetrics metrics();
    boolean supportsFeature(Feature feature);  // KV_CACHE, SPECULATIVE, MULTIMODAL
}
```

**`GenerationStream.java`**
```java
public interface GenerationStream {
    GenerationStream onToken(Consumer<String> handler);
    GenerationStream onComplete(Consumer<GenerationResult> handler);
    GenerationStream onError(Consumer<Throwable> handler);
    CompletableFuture<GenerationResult> toFuture();
    Stream<String> toStream();  // Java Stream API
}
```

### Feature Negotiation

```java
public enum Feature {
    KV_CACHE,
    SPECULATIVE_DECODING,
    MULTIMODAL,
    FLASH_ATTENTION,
    QUANTIZATION_INT8,
    STREAMING,
    BATCH_INFERENCE
}

// Auto-detected at session creation
FeatureSet features = session.supportedFeatures();
if (features.has(Feature.KV_CACHE)) {
    session.enableKVCache(KVCacheConfig.defaults());
}
```

**Deliverables:**
- [ ] `GollekClient` unified interface
- [ ] `InferenceSession` abstraction
- [ ] `GenerationStream` reactive API
- [ ] Backend router (GGUF/ONNX/LiteRT/LibTorch)
- [ ] Feature negotiation
- [ ] Migration guide from old APIs
- [ ] Examples for each backend

**Timeline:** Phase 1, Weeks 1-6 (parallel with CNN/RNN)

---

## 6. NLP Ecosystem (Critical Gap)

### 6.1 Tokenizer Framework

**Problem:** No public tokenization API for training. Inference works but training requires tokenizing datasets natively.

**New Module:** `gollek-sdk-tokenizer` (enhance existing `gollek-tokenizer-core`)

```
gollek/sdk/lib/gollek-sdk-tokenizer/
├── Tokenizer.java                 # Core interface
├── TokenizerConfig.java
├── impl/
│   ├── BPETokenizer.java          # Byte-Pair Encoding
│   ├── WordPieceTokenizer.java    # BERT-style
│   ├── SentencePieceTokenizer.java
│   └── TikTokenTokenizer.java     # GPT-style
├── loader/
│   ├── HuggingFaceTokenizerLoader.java  # Load tokenizer.json
│   └── VocabLoader.java
└── README.md
```

**Core Interface:**
```java
public interface Tokenizer {
    List<Integer> encode(String text);
    String decode(List<Integer> ids);
    BatchEncoding batchEncode(List<String> texts, int maxLength);
    int vocabSize();
    int padTokenId();
    int eosTokenId();
    int bosTokenId();
}
```

**HuggingFace Compatibility:**
```java
// Load any HuggingFace tokenizer
Tokenizer tokenizer = HuggingFaceTokenizerLoader
    .from(Path.of("tokenizer.json"));

// Or from Hub
Tokenizer tokenizer = ModelHub.loadTokenizer("bert-base-uncased");

// Batch encode for training
BatchEncoding batch = tokenizer.batchEncode(
    texts,
    maxLength = 512,
    padding = true,
    truncation = true
);
```

**Deliverables:**
- [ ] `Tokenizer` interface
- [ ] BPE, WordPiece, SentencePiece implementations
- [ ] HuggingFace `tokenizer.json` loader
- [ ] Padding/truncation/attention masks
- [ ] Example: BERT fine-tuning pipeline
- [ ] Unit tests (>95% coverage)

**Timeline:** Phase 1, Weeks 5-10

---

### 6.2 Model StateDict & Serialization

**Problem:** Can load SafeTensors but cannot save trained models.

**Enhance:** `gollek-sdk-nn`

```java
// Save
model.saveStateDict(Path.of("model.safetensors"));
model.save(Path.of("checkpoint/"));  // Full checkpoint

// Load
model.loadStateDict(Path.of("model.safetensors"));
Module loaded = Module.load(Path.of("checkpoint/"));

// Partial loading (transfer learning)
model.loadStateDict(pretrained, strict = false);

// Inspect
Map<String, Tensor> stateDict = model.stateDict();
stateDict.forEach((name, param) -> 
    System.out.printf("%s: %s%n", name, param.shape()));
```

**StateDict API:**
```java
public interface StateDictCapable {
    Map<String, GradTensor> stateDict();
    void loadStateDict(Map<String, GradTensor> state);
    void loadStateDict(Map<String, GradTensor> state, boolean strict);
    void saveStateDict(Path path);
    void save(Path checkpointDir);
    static Module load(Path checkpointDir);
}
```

**Deliverables:**
- [ ] `stateDict()` / `loadStateDict()` on all modules
- [ ] SafeTensors save/load
- [ ] GGUF save/load
- [ ] Checkpoint with optimizer state
- [ ] Strict/non-strict loading
- [ ] Example: Fine-tune and save BERT
- [ ] Unit tests

**Timeline:** Phase 1, Weeks 8-12

---

### 6.3 NLP Training Pipeline

**Enhance:** `gollek-sdk-nlp`

```
gollek/sdk/lib/gollek-sdk-nlp/
├── TextDataset.java               # Text dataset with tokenization
├── TextClassificationTrainer.java
├── SequenceToSequenceTrainer.java
├── LanguageModelTrainer.java      # CLM / MLM
├── tasks/
│   ├── TextClassification.java
│   ├── TokenClassification.java   # NER
│   ├── QuestionAnswering.java
│   └── Summarization.java
└── README.md
```

**Example: Fine-tune for classification:**
```java
// Dataset
TextDataset dataset = TextDataset.builder()
    .tokenizer(tokenizer)
    .texts(texts)
    .labels(labels)
    .maxLength(512)
    .build();

// Model
BertForSequenceClassification model =
    BertForSequenceClassification.pretrained("bert-base-uncased", numLabels=2);

// Trainer
TextClassificationTrainer trainer = TextClassificationTrainer.builder()
    .model(model)
    .optimizer(AdamW.create(model.parameters(), lr=2e-5, weightDecay=0.01))
    .epochs(3)
    .warmupSteps(100)
    .build();

trainer.fit(dataset.split(0.8));
```

**Deliverables:**
- [ ] `TextDataset` with tokenization
- [ ] Fine-tuning trainers for common tasks
- [ ] Warmup LR scheduler
- [ ] Example: Sentiment analysis fine-tuning
- [ ] Example: NER fine-tuning
- [ ] Documentation

**Timeline:** Phase 2, Months 7-9

---

## 7. Hardware Acceleration Strategy

### 7.1 Device Abstraction

**Problem:** `GradTensor` uses `float[]` — all ops on CPU. Need transparent GPU support.

**Design:**
```java
// Device placement (PyTorch-style)
GradTensor tensor = GradTensor.of(data).to(Device.CUDA);
GradTensor tensor = GradTensor.of(data).to(Device.METAL);  // Apple
GradTensor tensor = GradTensor.of(data).to(Device.CPU);

// Model to device
model.to(Device.CUDA);

// Auto device selection
Device device = Device.best();  // CUDA > Metal > CPU
```

**`Device.java`:**
```java
public sealed interface Device permits Device.CPU, Device.CUDA, Device.Metal, Device.ROCm {
    record CPU() implements Device {}
    record CUDA(int index) implements Device {
        static CUDA of(int index) { return new CUDA(index); }
    }
    record Metal() implements Device {}

    static Device best() {
        if (CUDARuntime.isAvailable()) return new CUDA(0);
        if (MetalRuntime.isAvailable()) return new Metal();
        return new CPU();
    }
}
```

### 7.2 Backend Binding Strategy

**Approach:** Bind `GradTensor` storage to existing Gollek C++ core via JNI/FFM.

```
gollek/sdk/lib/gollek-sdk-tensor/
├── storage/
│   ├── TensorStorage.java         # Interface
│   ├── HeapStorage.java           # float[] (current)
│   ├── CUDAStorage.java           # CUDA device memory
│   └── MetalStorage.java          # Apple Metal
├── ops/
│   ├── TensorOps.java             # Interface
│   ├── CPUOps.java                # Java implementation
│   ├── CUDAOps.java               # cuBLAS/cuDNN via JNI
│   └── MetalOps.java              # Metal Performance Shaders
└── Device.java
```

**JNI Bridge:**
```java
// CUDAOps.java
public class CUDAOps implements TensorOps {
    static { System.loadLibrary("gollek-cuda"); }

    @Override
    public native float[] matmul(float[] a, float[] b, int[] shapeA, int[] shapeB);

    @Override
    public native float[] conv2d(float[] input, float[] weight, int[] config);
}
```

### 7.3 Phased GPU Rollout

**Phase 2 (Months 7-9): Foundation**
- Device abstraction API
- CPU/GPU memory management
- JNI bridge to existing C++ kernels
- `tensor.to(device)` semantics

**Phase 3 (Months 13-15): Acceleration**
- cuBLAS for matmul
- cuDNN for Conv2d, BatchNorm
- Flash Attention CUDA kernel
- Apple Metal for macOS

**Phase 4 (Months 19-21): Advanced**
- TensorRT integration
- ROCm for AMD
- Custom CUDA kernels
- Multi-GPU memory management

**Deliverables (Phase 2):**
- [ ] `Device` sealed interface
- [ ] `TensorStorage` abstraction
- [ ] JNI bridge to C++ core
- [ ] `tensor.to(Device.CUDA)` working
- [ ] Benchmark: CPU vs GPU matmul
- [ ] Documentation

---

## 8. Tensor Operations — Concrete Implementation

### 8.1 Missing Operations in `GradTensor`

**Slicing & Indexing:**
```java
// Slice along dimension
GradTensor seq = tensor.slice(dim=0, start=0, end=10);

// Boolean masking
GradTensor masked = tensor.maskedSelect(mask);
tensor.maskedFill_(mask, value=0.0f);

// Gather/scatter (critical for attention, loss functions)
GradTensor gathered = tensor.gather(dim=1, index);
tensor.scatter_(dim=1, index, src);

// Advanced indexing
GradTensor selected = tensor.index(new int[]{0, 2, 4});
```

**Shape Operations:**
```java
// Concatenate and stack
GradTensor cat = GradTensor.cat(List.of(a, b, c), dim=0);
GradTensor stacked = GradTensor.stack(List.of(a, b, c), dim=0);

// Split
List<GradTensor> chunks = tensor.split(chunkSize=64, dim=0);
List<GradTensor> parts = tensor.chunk(numChunks=4, dim=0);

// Squeeze/unsqueeze
GradTensor expanded = tensor.unsqueeze(dim=0);
GradTensor squeezed = tensor.squeeze(dim=0);
```

**Reduction Operations:**
```java
tensor.sum(dim=1, keepdim=true)
tensor.mean(dim=1)
tensor.max(dim=1)
tensor.min(dim=1)
tensor.argmax(dim=1)
tensor.std(dim=1)
tensor.var(dim=1)
tensor.norm(p=2, dim=1)
```

**Einstein Summation:**
```java
// Batch matmul
GradTensor result = GradTensor.einsum("bij,bjk->bik", a, b);

// Attention scores
GradTensor scores = GradTensor.einsum("bhid,bhjd->bhij", q, k);

// Outer product
GradTensor outer = GradTensor.einsum("i,j->ij", a, b);
```

**Deliverables:**
- [ ] `slice`, `gather`, `scatter`, `maskedSelect`
- [ ] `cat`, `stack`, `split`, `chunk`
- [ ] Full reduction ops with `dim` and `keepdim`
- [ ] `einsum` with common patterns
- [ ] Backward pass for all new ops
- [ ] Unit tests (>95% coverage)

**Timeline:** Phase 1, Weeks 17-22

---

## 9. Integration with Wayang Platform

### 9.1 Agent Integration

The Gollek SDK should integrate cleanly with `wayang-gollek/agent`:

```java
// Agent using Gollek SDK for inference
@Agent
public class InferenceAgent {

    @Inject
    GollekClient client;

    @Skill("generate")
    public String generate(String prompt) {
        return client.generate(
            GenerationRequest.of(prompt)
        ).text();
    }

    @Skill("embed")
    public float[] embed(String text) {
        return client.embed(text);
    }
}
```

**New Module:** `gollek-sdk-agent-integration`

```
gollek/sdk/lib/gollek-sdk-agent-integration/
├── AgentGollekClient.java         # Agent-aware client
├── SkillInferenceAdapter.java     # Skill → inference bridge
├── MemoryAwareSession.java        # Session with agent memory
└── README.md
```

### 9.2 RAG Integration

```java
// RAG pipeline using SDK
RAGPipeline pipeline = RAGPipeline.builder()
    .embedder(client.asEmbedder())
    .vectorStore(vectorStore)
    .generator(client.asGenerator())
    .build();

String answer = pipeline.query("What is attention mechanism?");
```

### 9.3 Gamelan Workflow Integration

```java
// Workflow node using Gollek SDK
@WorkflowNode("inference")
public class InferenceNode implements Node {

    @Override
    public NodeResult execute(NodeContext ctx) {
        String prompt = ctx.input("prompt");
        GenerationResult result = client.generate(
            GenerationRequest.of(prompt)
                .maxTokens(512)
                .temperature(0.7f)
        );
        return NodeResult.of("output", result.text());
    }
}
```

**Deliverables:**
- [ ] Agent integration module
- [ ] RAG pipeline adapter
- [ ] Gamelan workflow node
- [ ] Examples for each integration
- [ ] Documentation

**Timeline:** Phase 2, Months 10-12

---

## 10. Developer Experience (DX) Improvements

### 10.1 Fluent Builder APIs

All SDK components should use consistent fluent builders:

```java
// Consistent pattern across all modules
Model model = Sequential.builder()
    .add(Linear.of(784, 256))
    .add(ReLU.instance())
    .add(Dropout.rate(0.2f))
    .add(Linear.of(256, 10))
    .build();

Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(Adam.lr(0.001))
    .loss(CrossEntropyLoss.instance())
    .epochs(100)
    .callbacks(EarlyStopping.patience(5), ModelCheckpoint.best())
    .build();
```

### 10.2 JBang Examples

Provide runnable JBang scripts for quick experimentation:

```java
///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.2.0
//DEPS tech.kayys.gollek:gollek-sdk-train:0.2.0

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.sdk.train.*;

public class QuickStart {
    public static void main(String[] args) {
        var model = new Sequential(new Linear(2, 4), new ReLU(), new Linear(4, 1));
        var trainer = Trainer.builder().model(model).epochs(100).build();
        trainer.fit(xorDataset());
    }
}
```

### 10.3 Annotation-Based Model Definition

```java
// Declarative model definition
@Model
public class MyClassifier extends Module {

    @Layer
    private final Linear fc1 = Linear.of(784, 256);

    @Layer
    private final ReLU relu = ReLU.instance();

    @Layer
    private final Linear fc2 = Linear.of(256, 10);

    @Override
    public GradTensor forward(GradTensor x) {
        return fc2.forward(relu.forward(fc1.forward(x)));
    }
}
```

### 10.4 Python Interop

Bridge for teams migrating from Python:

```java
// Load PyTorch model weights
Module model = PytorchBridge.loadModel("model.pt");

// Convert numpy arrays
GradTensor tensor = NumpyBridge.fromNpy("data.npy");

// Export to Python-compatible format
model.saveStateDict("weights.safetensors");  // Loadable in Python
```

**Deliverables:**
- [ ] Consistent builder APIs across all modules
- [ ] 20+ JBang runnable examples
- [ ] Annotation-based model definition
- [ ] PyTorch weight loading
- [ ] NumPy array bridge
- [ ] Interactive REPL support

**Timeline:** Phase 1-2 (continuous)

---

## 11. Testing Strategy

### 11.1 Test Pyramid

```
gollek/sdk/lib/*/src/test/
├── unit/           # Fast, isolated, >95% coverage
├── integration/    # Cross-module, medium speed
├── benchmark/      # JMH benchmarks
└── e2e/            # Full pipeline tests
```

### 11.2 Numerical Correctness Tests

All autograd operations must be verified against PyTorch:

```java
@Test
void testLinearBackward() {
    // Compare gradient with PyTorch reference values
    GradTensor x = GradTensor.of(new float[]{1, 2, 3, 4}, new int[]{2, 2});
    GradTensor w = GradTensor.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new int[]{2, 2});

    GradTensor out = x.matmul(w);
    out.sum().backward();

    // Reference gradients computed in PyTorch
    assertArrayEquals(new float[]{0.3f, 0.7f, 0.3f, 0.7f}, x.grad(), 1e-5f);
}
```

### 11.3 Performance Benchmarks (JMH)

```java
@Benchmark
@BenchmarkMode(Mode.Throughput)
public void matmulBenchmark(Blackhole bh) {
    GradTensor a = GradTensor.randn(512, 512);
    GradTensor b = GradTensor.randn(512, 512);
    bh.consume(a.matmul(b));
}
```

### 11.4 Regression Tests

- Track performance across releases
- Alert on >5% regression
- Compare with PyTorch baseline

**Deliverables:**
- [ ] Unit test suite per module (>90% coverage)
- [ ] Numerical correctness tests vs PyTorch
- [ ] JMH benchmark suite
- [ ] CI/CD integration
- [ ] Performance dashboard

---

## 12. Release & Versioning Strategy

### 12.1 Semantic Versioning

```
0.x.y  — Current (breaking changes allowed)
1.0.0  — Phase 1 complete (stable API)
2.0.0  — Phase 2 complete (production ready)
3.0.0  — Phase 3 complete (advanced features)
```

### 12.2 Module Versioning

All SDK modules versioned together via BOM:

```xml
<!-- Single BOM import -->
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<!-- Then use without versions -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-nn</artifactId>
</dependency>
```

### 12.3 Deprecation Policy

- Deprecated APIs kept for 2 major versions
- Migration guides for every breaking change
- `@Deprecated` + `@since` + `@see` Javadoc

---

## 13. Updated Module Map

```
gollek/sdk/lib/
├── gollek-sdk-tensor/         ← Enhance: advanced ops, device abstraction
├── gollek-sdk-autograd/       ← Enhance: higher-order grads, hooks
├── gollek-sdk-nn/             ← Enhance: statedict, more layers
├── gollek-sdk-cnn/            ← NEW: Conv2d, MaxPool, etc.
├── gollek-sdk-rnn/            ← NEW: LSTM, GRU
├── gollek-sdk-transformers/   ← NEW: Transformer blocks
├── gollek-sdk-train/          ← Enhance: distributed, gradient accum
├── gollek-sdk-data/           ← Enhance: augmentation, streaming
├── gollek-sdk-tokenizer/      ← NEW: BPE, WordPiece, HF loader
├── gollek-sdk-nlp/            ← Enhance: fine-tuning tasks
├── gollek-sdk-vision/         ← Enhance: image transforms
├── gollek-sdk-multimodal/     ← Enhance: vision-language
├── gollek-sdk-optimize/       ← Enhance: quantization, pruning
├── gollek-sdk-distributed/    ← NEW: DDP, multi-node
├── gollek-sdk-viz/            ← NEW: TensorBoard, W&B
├── gollek-sdk-models/         ← NEW: pre-trained zoo
├── gollek-sdk-unified/        ← NEW: unified client API
├── gollek-sdk-agent-integration/ ← NEW: agent/RAG/workflow hooks
├── gollek-sdk-ml/             ← Enhance: sklearn-style API
├── gollek-sdk-hub/            ← Enhance: model registry
├── gollek-sdk-export/         ← Enhance: ONNX, mobile
├── gollek-sdk-litert/         ← Existing: LiteRT/TFLite
└── gollek-sdk-onnx/           ← Existing: ONNX runtime
```

---

## 14. Immediate Next Steps

### Week 1-2 Actions

1. **Tensor ops gap analysis** — list every missing op vs PyTorch
2. **Conv2d design doc** — im2col vs direct convolution decision
3. **Tokenizer loader** — prototype HuggingFace `tokenizer.json` parser
4. **StateDict API** — design and implement `Module.save/load`
5. **Device abstraction** — design `Device` sealed interface

### Milestone: v0.2.0 (End of Phase 1)

- [ ] Conv2d + MaxPool2d working
- [ ] LSTM working
- [ ] Tokenizer loading HuggingFace models
- [ ] `model.save()` / `model.load()` working
- [ ] Advanced tensor ops (slice, gather, cat, stack)
- [ ] Unified client API
- [ ] 15+ JBang examples
- [ ] All tests passing

---

**Document Owner:** Gollek Core Team  
**Last Updated:** April 7, 2026  
**Next:** `framework-enhancement-plan-03.md` — Implementation walkthroughs
