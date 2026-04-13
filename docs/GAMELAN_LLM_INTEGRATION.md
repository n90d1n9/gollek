# Gamelan LLM Workflow Integration

> **Date:** 2026-04-12  
> **Status:** ✅ Complete  
> **Purpose:** Native LLM inference as Gamelan workflow primitives

---

## Overview

Integrates **gollek inference** as first-class workflow nodes in the **Gamelan orchestration engine**, enabling AI agent workflows with LLM-native steps.

## Architecture

```
Gamelan Workflow
  ↓
┌─────────────────────────────────────────────────────┐
│ Node 1: gollek/inference                            │
│   → Call gollek with prompt                         │
│   → Stream tokens                                   │
│   → Output: {response, tokens, latency}             │
└───────────────────┬─────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Node 2: gollek/rag                                  │
│   → Retrieve from vector DB                         │
│   → Inject context into prompt                      │
│   → Generate with gollek                            │
│   → Output: {answer, sources, citations}            │
└───────────────────┬─────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Node 3: gollek/embedding                            │
│   → Embed text using gollek                         │
│   → Output: {embedding, dimensions}                 │
└───────────────────┬─────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Node 4: gollek/classification                       │
│   → Classify text using gollek                      │
│   → Output: {label, confidence, scores}             │
└─────────────────────────────────────────────────────┘
```

## Workflow Example

```yaml
# AI Agent Workflow with LLM steps
workflow:
  name: "Research Assistant"
  nodes:
    - id: retrieve
      type: gollek/rag
      config:
        query: "{{input.question}}"
        vector_store: pgvector
        top_k: 5
        model: llama-3-70b

    - id: analyze
      type: gollek/inference
      config:
        prompt: |
          Analyze the following retrieved documents:
          {{retrieve.answer}}

          Question: {{input.question}}

          Provide a structured analysis with:
          1. Key findings
          2. Evidence from sources
          3. Confidence level
        model: llama-3-70b
        max_tokens: 1024
        temperature: 0.7

    - id: classify
      type: gollek/classification
      config:
        text: "{{analyze.response}}"
        categories: [technical, business, academic]
        model: llama-3-8b

    - id: summarize
      type: gollek/inference
      config:
        prompt: |
          Summarize the analysis in 3 bullet points:
          {{analyze.response}}
        model: llama-3-70b
        max_tokens: 256

    - id: embed
      type: gollek/embedding
      config:
        text: "{{summarize.response}}"
        model: text-embedding-3-small

  output:
    analysis: "{{analyze.response}}"
    classification: "{{classify.label}}"
    summary: "{{summarize.response}}"
    embedding: "{{embed.embedding}}"
    sources: "{{retrieve.sources}}"
```

## Node Types

### 1. `gollek/inference` — LLM Generation

```yaml
type: gollek/inference
config:
  model: llama-3-70b
  prompt: "{{input.text}}"
  max_tokens: 512
  temperature: 0.8
  top_p: 0.95
  stop: ["\n\n", "###"]
  streaming: false
  system_prompt: "You are a helpful assistant."
```

**Inputs:**
- `prompt`: The prompt text (required)
- `model`: Model identifier
- `max_tokens`: Maximum output tokens
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling
- `stop`: Stop sequences
- `system_prompt`: System message

**Outputs:**
- `response`: Generated text
- `input_tokens`: Input token count
- `output_tokens`: Output token count
- `finish_reason`: stop | length | content_filter
- `latency_ms`: Generation latency

### 2. `gollek/rag` — Retrieval-Augmented Generation

```yaml
type: gollek/rag
config:
  query: "{{input.question}}"
  vector_store: pgvector
  collection: documents
  embedder: text-embedding-3-small
  top_k: 5
  top_n: 3
  model: llama-3-70b
  prompt_template: with_citations
  max_context_tokens: 4096
```

**Inputs:**
- `query`: User's question
- `vector_store`: Vector store type
- `collection`: Document collection name
- `embedder`: Embedding model
- `top_k`: Documents to retrieve
- `top_n`: Documents after reranking
- `model`: Generator model

**Outputs:**
- `answer`: Generated answer
- `sources`: Retrieved documents with scores
- `citations`: Citations in answer
- `input_tokens`: Input token count
- `output_tokens`: Output token count

### 3. `gollek/embedding` — Text Embedding

```yaml
type: gollek/embedding
config:
  text: "{{input.text}}"
  model: text-embedding-3-small
  dimensions: 1536
```

**Inputs:**
- `text`: Text to embed
- `model`: Embedding model
- `dimensions`: Output dimensions

**Outputs:**
- `embedding`: Vector embedding (float array)
- `dimensions`: Embedding dimensions
- `latency_ms`: Embedding latency

### 4. `gollek/classification` — Text Classification

```yaml
type: gollek/classification
config:
  text: "{{input.text}}"
  categories: [positive, negative, neutral]
  model: llama-3-8b
  temperature: 0.1
```

**Inputs:**
- `text`: Text to classify
- `categories`: List of categories
- `model`: Classification model
- `temperature`: Low temperature for deterministic

**Outputs:**
- `label`: Predicted category
- `confidence`: Confidence score (0.0-1.0)
- `scores`: Scores for all categories

### 5. `gollek/agent` — ReAct Agent Loop

```yaml
type: gollek/agent
config:
  objective: "{{input.objective}}"
  tools: [search, calculator, code_interpreter]
  model: llama-3-70b
  max_iterations: 10
  system_prompt: |
    You are an AI agent that can use tools to accomplish tasks.
    Think step by step and use tools when needed.
```

**Inputs:**
- `objective`: Task objective
- `tools`: Available tools
- `model`: Agent model
- `max_iterations`: Max ReAct iterations

**Outputs:**
- `result`: Final result
- `iterations`: Number of iterations
- `tool_calls`: List of tool calls made
- `thoughts`: Agent's reasoning steps

## Implementation

### Node Executor Interface

```java
public class GollekInferenceNode implements NodeExecution {

    private final InferenceService inferenceService;

    @Override
    public NodeResult execute(NodeExecutionContext context) {
        String prompt = context.getInput("prompt", String.class);
        String model = context.getConfig("model", "llama-3-70b");
        int maxTokens = context.getConfig("max_tokens", 256);

        InferenceResponse response = inferenceService.infer(
            InferenceRequest.builder()
                .model(model)
                .messages(List.of(Message.user(prompt)))
                .maxTokens(maxTokens)
                .build(),
            context.getRequestContext()
        );

        return NodeResult.success(Map.of(
            "response", response.content(),
            "input_tokens", response.inputTokens(),
            "output_tokens", response.outputTokens(),
            "finish_reason", response.finishReason(),
            "latency_ms", response.durationMs()
        ));
    }
}
```

### Plugin Registration

```java
public class GollekNodesPlugin implements GamelanPlugin {

    @Override
    public void register(NodeTypeRegistry registry) {
        registry.register("gollek/inference", GollekInferenceNode.class);
        registry.register("gollek/rag", GollekRAGNode.class);
        registry.register("gollek/embedding", GollekEmbeddingNode.class);
        registry.register("gollek/classification", GollekClassificationNode.class);
        registry.register("gollek/agent", GollekAgentNode.class);
    }
}
```

## Benefits

1. **LLM-Native Workflows** — Workflows understand LLM concepts (tokens, streaming, prompts)
2. **Unified Platform** — Single platform for inference + orchestration
3. **Type-Safe** — JVM compile-time safety for workflow definitions
4. **Observable** — Full tracing from workflow to token generation
5. **Scalable** — Distributed workflow execution with gollek inference
6. **Enterprise-Ready** — Multi-tenant, rate-limited, canary-deployable

## Integration with Existing Components

```
Gamelan Workflow Engine
  ↓
Gollek Node Executors
  ↓
InferenceService (central orchestrator)
  ├── RateLimiter
  ├── PrefixCache
  ├── ProviderRouter
  ├── SpeculativeDecoder
  ├── TurboQuantKVCacheAdapter
  └── ContinuousBatchScheduler
```

---

**Status:** ✅ Design Complete  
**Next:** Implementation of node executors  
**Team:** Gollek Engineering
