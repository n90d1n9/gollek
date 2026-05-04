# Gollek Inference Orchestrator Deep Dive

The **Inference Orchestrator** is a critical subsystem within the Gollek Engine (`gollek-engine`). It is responsible for managing the complete lifecycle of inference requests, acting as the operational "brain" that coordinates routing, execution states, disaggregated architectures, and metric collection.

## 1. Orchestrator Component Architecture

The orchestrator abstracts the complexity of different execution strategies (combined vs. disaggregated) and provides a unified interface for the `InferenceEngine` to execute requests.

```mermaid
classDiagram
    class InferenceOrchestrator {
        <<interface>>
        +executeAsync(modelId, request)
        +streamExecute(modelId, request)
        +executeEmbedding(modelId, request)
    }

    class DefaultInferenceOrchestrator {
        -modelRouter: ModelRouterService
        -metrics: InferenceMetrics
        -pdProvider: DisaggregatedLLMProvider
        +executeAsync(modelId, request)
    }

    class StageAwareOrchestrator {
        -stateMachine: ExecutionStateMachine
        +executeStage(request, stage)
    }

    class ExecutionStateMachine {
        -currentState: ExecutionState
        -context: RequestContext
        +transition(newState)
        +getCurrentState()
    }

    class DisaggregatedLLMProvider {
        <<interface>>
        +prefill(request)
        +decode(request, kvCacheId)
    }

    class ExecutionState {
        <<enumeration>>
        PRE_PROCESSING
        PREFILL
        DECODE
        POST_PROCESSING
        COMPLETED
        FAILED
    }

    InferenceOrchestrator <|-- DefaultInferenceOrchestrator
    InferenceOrchestrator <|-- StageAwareOrchestrator
    DefaultInferenceOrchestrator --> StageAwareOrchestrator : delegates complex flows
    StageAwareOrchestrator --> ExecutionStateMachine : manages state
    StageAwareOrchestrator --> DisaggregatedLLMProvider : executes phases
    ExecutionStateMachine --> ExecutionState : maintains
```

## 2. Execution State Machine

The `ExecutionStateMachine` strictly controls the transitions between different phases of an inference request, preventing invalid operational sequences.

```mermaid
stateDiagram-v2
    [*] --> PRE_PROCESSING : Request Received
    
    state "Pre-Processing Phase" as PreProcessing {
        PRE_PROCESSING
        note right of PRE_PROCESSING
            Tokenization, 
            Constraint Validation,
            Cache Lookups
        end note
    }

    PreProcessing --> DisaggregatedRoute : Is Disaggregated?
    PreProcessing --> CombinedRoute : Is Combined?

    state "Disaggregated Execution" as Disaggregated {
        PREFILL
        DECODE
        
        PREFILL --> DECODE : KV Cache & Prompt Embeddings Ready
        DECODE --> DECODE : Autoregressive Token Generation
    }

    state "Combined Execution" as Combined {
        COMBINED_INFERENCE
        note right of COMBINED_INFERENCE
            Standard single-pass
            Runner execution
        end note
    }

    DisaggregatedRoute --> PREFILL
    CombinedRoute --> COMBINED_INFERENCE

    DECODE --> POST_PROCESSING : End of Sequence (EOS) Reached
    COMBINED_INFERENCE --> POST_PROCESSING : Generation Complete

    state "Post-Processing Phase" as PostProcessing {
        POST_PROCESSING
        note right of POST_PROCESSING
            Detokenization, 
            Output Formatting,
            Metrics Aggregation
        end note
    }

    POST_PROCESSING --> COMPLETED : Success
    
    PRE_PROCESSING --> FAILED : Error
    PREFILL --> FAILED : Error
    DECODE --> FAILED : Error
    COMBINED_INFERENCE --> FAILED : Error
    POST_PROCESSING --> FAILED : Error
    
    COMPLETED --> [*]
    FAILED --> [*]
```

## 3. Disaggregated Inference Sequence

Disaggregated inference separates the highly parallel "Prefill" phase (processing the prompt) from the memory-bandwidth-bound "Decode" phase (generating tokens one-by-one). The Orchestrator manages this handoff seamlessly.

```mermaid
sequenceDiagram
    participant Eng as InferenceEngine
    participant Orch as StageAwareOrchestrator
    participant SM as ExecutionStateMachine
    participant Router as ModelRouterService
    participant PD as DisaggregatedLLMProvider
    participant Cache as KVCacheManager

    Eng->>Orch: executeAsync(request)
    Orch->>SM: transition(PRE_PROCESSING)
    
    Orch->>Router: resolveRouting(request)
    Router-->>Orch: Disaggregated Topology Detected
    
    %% Prefill Stage
    Orch->>SM: transition(PREFILL)
    Orch->>PD: prefill(request)
    note over PD: Processes all prompt tokens in parallel
    PD->>Cache: store(kv_states)
    Cache-->>PD: kvCacheId
    PD-->>Orch: PrefillResponse(kvCacheId)
    
    %% Decode Stage
    Orch->>SM: transition(DECODE)
    
    loop Until EOS Token
        Orch->>PD: decode(request, kvCacheId)
        PD->>Cache: fetch(kvCacheId)
        Cache-->>PD: kv_states
        note over PD: Generates 1 token
        PD->>Cache: update(kv_states)
        PD-->>Orch: DecodeChunk(token)
        Orch-->>Eng: stream(token)
    end
    
    %% Post Processing
    Orch->>SM: transition(POST_PROCESSING)
    Orch->>Cache: release(kvCacheId)
    Orch->>SM: transition(COMPLETED)
    Orch-->>Eng: Response Completed
```

## 4. Error Handling and Resilience Flow

The Orchestrator incorporates self-healing and resilient execution pathways. If a primary provider or stage fails, the orchestrator handles fallback routing.

```mermaid
sequenceDiagram
    participant Orch as DefaultInferenceOrchestrator
    participant Router as ModelRouterService
    participant Primary as Primary Provider (e.g., Metal)
    participant Fallback as Fallback Provider (e.g., CPU)

    Orch->>Router: route(request)
    Router-->>Orch: PrimaryProvider
    
    Orch->>Primary: infer(request)
    Primary--xOrch: ResourceExhaustedException (OOM)
    
    note over Orch: Error Caught
    Orch->>Orch: Increment Failure Metric
    
    Orch->>Router: route(request, exclude=Primary)
    Router-->>Orch: FallbackProvider
    
    Orch->>Fallback: infer(request)
    Fallback-->>Orch: InferenceResponse (Success)
    Orch-->>Orch: Record Recovery Metric
```
