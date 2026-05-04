# Gollek Architecture & Request Flow

This document illustrates the internal architecture and flow of the Gollek Inference Engine across various scenarios, using Mermaid diagrams.

## 1. High-Level Architecture

The Gollek platform is built around a pluggable architecture, isolating the CLI and routing layers from the underlying inference formats and native kernels.

```mermaid
flowchart TD
    subgraph CLI ["CLI Layer (gollek-cli)"]
        ChatCmd[chat command]
        RunCmd[run command]
        UI[Terminal UI Renderer]
    end

    subgraph API ["API & Engine Layer (gollek-engine)"]
        Router[ModelRouterService]
        Registry[Model Registry]
        Metrics[Metrics & Cache]
        Engine[InferenceEngine]
    end

    subgraph PluginSystem ["Plugin System Integrator"]
        RunnerManager[Runner Plugin Manager]
        KernelManager[Kernel Plugin Manager]
    end

    subgraph Runners ["Level 1: Runner Plugins"]
        RunnerJlama["Jlama Runner\n(safetensors)"]
        RunnerLlama["LlamaCpp Runner\n(gguf)"]
        RunnerONNX["ONNX Runner\n(onnx)"]
        RunnerDiffuser["Diffuser Runner\n(stable-diffusion)"]
        RunnerLiteRT["LiteRT Runner\n(tflite)"]
    end

    subgraph Hardware ["Native execution FFM / JNI"]
        Metal[Metal]
        CUDA[CUDA]
        CPU[CPU]
    end

    ChatCmd --> Router
    RunCmd --> Router
    Router <--> Registry
    Router --> Engine
    Engine --> RunnerManager
    RunnerManager --> RunnerJlama
    RunnerManager --> RunnerONNX
    RunnerManager --> RunnerDiffuser
    
    RunnerJlama --> Hardware
    RunnerONNX --> Hardware
    RunnerDiffuser --> Hardware
```

---

## 2. Request Flows

### Scenario A: Interactive Chat with a Text Model
**Command:** `gollek chat --model 4b05be` (Model: Qwen2.5-0.5B-Instruct, safetensors)

In this scenario, the CLI establishes an interactive streaming session. The engine routes the `safetensors` model to the corresponding text runner (e.g., Jlama or HuggingFace runner).

```mermaid
sequenceDiagram
    actor User
    participant CLI as Gollek CLI
    participant Router as ModelRouterService
    participant Reg as ModelRegistry
    participant Engine as InferenceEngine
    participant Plugin as RunnerPluginRegistry
    participant Runner as Text Runner (safetensors)

    User->>CLI: `gollek chat --model 4b05be`
    CLI->>Router: stream(model=4b05be)
    Router->>Reg: resolve("4b05be")
    Reg-->>Router: ModelManifest (format: safetensors, text-generation)
    Router->>Engine: route request
    Engine->>Plugin: createSession(manifest)
    Plugin-->>Engine: RunnerSession (Jlama/Safetensors)
    
    loop Chat Session
        User->>CLI: "Hello, how are you?"
        CLI->>Engine: streamExecute(prompt)
        Engine->>Runner: stream(prompt)
        
        loop Stream Chunks
            Runner-->>CLI: StreamingInferenceChunk
            CLI->>User: (Updates Terminal UI)
        end
    end
```

### Scenario B: Text-to-Image Generation
**Command:** `gollek run --model 551fc4 --prompt "draw an old car" --output ~/Downloads/old-car.png` (Model: Stable Diffusion v1.4, ONNX)

This is a single-shot execution. The engine identifies that the model requires the ONNX Diffuser runner.

```mermaid
sequenceDiagram
    actor User
    participant CLI as Gollek CLI
    participant Router as ModelRouterService
    participant Reg as ModelRegistry
    participant Engine as InferenceEngine
    participant Plugin as RunnerPluginRegistry
    participant Runner as Diffuser/ONNX Runner

    User->>CLI: `gollek run --model 551fc4 --prompt "..."`
    CLI->>Router: execute(model=551fc4, prompt)
    Router->>Reg: resolve("551fc4")
    Reg-->>Router: ModelManifest (format: onnx, text-to-image)
    Router->>Engine: executeAsync(request)
    Engine->>Plugin: createSession(manifest)
    Plugin-->>Engine: RunnerSession (Diffuser/ONNX)
    
    Engine->>Runner: infer(prompt)
    Note over Runner: Tokenize Prompt<br/>UNet Denoising (ONNX/FFM)<br/>VAE Decode
    Runner-->>Engine: InferenceResponse (Image Buffer)
    Engine-->>CLI: Response
    CLI->>CLI: Save to ~/Downloads/old-car.png
    CLI-->>User: "Image generated successfully"
```

### Scenario C: Single-Shot Text Inference
**Command:** `gollek run --model 1a008d --prompt "who are you"` (Model: Gemma-4-E2B-it, safetensors)

This flow is similar to the chat scenario, but it executes as a single, synchronous operation rather than setting up an interactive loop.

```mermaid
sequenceDiagram
    actor User
    participant CLI as Gollek CLI
    participant Router as ModelRouterService
    participant Reg as ModelRegistry
    participant Engine as InferenceEngine
    participant Plugin as RunnerPluginRegistry
    participant Runner as Text Runner (safetensors)

    User->>CLI: `gollek run --model 1a008d --prompt "who are you"`
    CLI->>Router: execute(model=1a008d, prompt)
    Router->>Reg: resolve("1a008d")
    Reg-->>Router: ModelManifest (format: safetensors, text-generation)
    Router->>Engine: execute(request)
    Engine->>Plugin: createSession(manifest)
    Plugin-->>Engine: RunnerSession
    
    Engine->>Runner: infer(prompt)
    Note over Runner: Tokenize<br/>Forward Pass<br/>Sample Output
    Runner-->>Engine: InferenceResponse
    Engine-->>CLI: Response
    CLI-->>User: Prints output to stdout
```

## 3. Deep Dive: Inference Engine & Routing Core

This section drills down into the core components of `gollek-engine`, illustrating how requests are handled internally, routed, and orchestrated.

### Component Architecture

```mermaid
classDiagram
    class InferenceEngine {
        <<interface>>
        +initialize()
        +infer(request)
        +stream(request)
    }

    class DefaultInferenceEngine {
        -initialized: boolean
        +infer(request)
    }
    
    class InferenceOrchestrator {
        +executeAsync()
        +streamExecute()
    }
    
    class ModelRouterService {
        +route(modelId, request)
        +stream(modelId, request)
    }
    
    class FormatAwareProviderRouter {
        +resolveFormat(modelId)
    }
    
    class DevicePreferenceResolver {
        +apply(context, request)
    }

    class PluginSystemIntegrator {
        +initialize()
        +getKernelPluginManager()
        +getRunnerPluginManager()
    }

    InferenceEngine <|-- DefaultInferenceEngine
    DefaultInferenceEngine --> InferenceOrchestrator : uses
    DefaultInferenceEngine --> PluginSystemIntegrator : initializes
    InferenceOrchestrator --> ModelRouterService : delegates routing
    ModelRouterService --> FormatAwareProviderRouter : filters by format
    ModelRouterService --> DevicePreferenceResolver : applies context
    ModelRouterService --> ProviderRegistry : queries capabilities
```

### Deep Routing Sequence (`ModelRouterService`)

When the Engine receives a request, the `ModelRouterService` performs a complex resolution process to find the correct `LLMProvider` or `RunnerPlugin`.

```mermaid
sequenceDiagram
    participant Orch as InferenceOrchestrator
    participant Router as ModelRouterService
    participant DevRes as DevicePreferenceResolver
    participant FormatRes as FormatAwareProviderRouter
    participant Reg as ProviderRegistry
    participant Provider as LLMProvider

    Orch->>Router: route("model-id", Request)
    
    %% 1. Resolve Device Preferences
    Router->>DevRes: apply(RequestContext, Request)
    DevRes-->>Router: Updated RequestContext (e.g., preferredDevice=METAL)
    
    %% 2. Resolve Model Format
    Router->>FormatRes: resolveFormat("model-id")
    FormatRes-->>Router: Format (e.g., safetensors, gguf, onnx)
    
    %% 3. Filter Providers
    Router->>Reg: getAllProviders()
    Reg-->>Router: List<LLMProvider>
    
    loop For each Provider
        Router->>Provider: capabilities()
        Provider-->>Router: ProviderCapabilities
        Router->>Router: Check if Format matches
        
        alt Format Matches & Features Supported
            Router->>Provider: supports("model-id", ProviderRequest)
            Provider-->>Router: true
        end
    end
    
    %% 4. Execute
    alt Compatible Provider Found
        Router->>Provider: infer(ProviderRequest)
        Provider-->>Router: InferenceResponse
        Router-->>Orch: InferenceResponse
    else No Provider Found
        Router-->>Orch: throws NoCompatibleProviderException
    end
```

### Inference Orchestration (`InferenceOrchestrator`)

The `InferenceOrchestrator` manages the lifecycle of the inference execution, dealing with disaggregated inference stages (e.g., separated prefill and decode stages) and recording metrics.

```mermaid
stateDiagram-v2
    [*] --> PRE_PROCESSING
    
    PRE_PROCESSING --> ROUTING : Validate Request
    ROUTING --> EXECUTION : Find Compatible Provider
    
    state EXECUTION {
        [*] --> DISAGGREGATED_CHECK
        DISAGGREGATED_CHECK --> COMBINED_INFERENCE : Single Stage
        DISAGGREGATED_CHECK --> PREFILL_STAGE : Multi-Stage
        
        PREFILL_STAGE --> DECODE_STAGE : KV Cache Ready
        DECODE_STAGE --> [*] : Generation Complete
        COMBINED_INFERENCE --> [*] : Generation Complete
    }
    
    EXECUTION --> POST_PROCESSING
    POST_PROCESSING --> METRICS_RECORDING
    METRICS_RECORDING --> [*]
```
