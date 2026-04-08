
# Multimodal Inference Integration Guide

## Overview

The improved multimodal inference system provides unified support for processing multiple input modalities (text, images, audio, video, documents) and generating multimodal outputs through the Gollek inference engine.

**Integrated Runners:**
- ✅ **GGUF/llama.cpp** - LLaVA and vision-language models
- ✅ **ONNX Runtime** - CLIP, ViT, BLIP, and other vision models
- ✅ **LibTorch** - Stable Diffusion, DALL-E, and PyTorch vision models

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│              Gollek Inference Engine                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │     MultimodalEngineIntegration                   │  │
│  │  (Converts standard ↔ multimodal requests)       │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                   │
│         ┌───────────▼───────────┐                       │
│         │ MultimodalInference    │                       │
│         │ Service                │                       │
│         └───────────┬───────────┘                       │
│                     │                                   │
│         ┌───────────▼───────────┐                       │
│         │ MultimodalProcessors   │                       │
│         │ ┌─────┐ ┌─────┐ ┌───┐ │                       │
│         │ │Vision│ │Audio│ │...│ │                       │
│         │ └─────┘ └─────┘ └───┘ │                       │
│         └───────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### Supported Modalities

**Input:**
- ✅ Text
- ✅ Images (JPEG, PNG, WebP, GIF)
- ✅ Audio (WAV, MP3, FLAC)
- ✅ Video (MP4, WebM)
- ✅ Documents (PDF, DOCX, HTML)
- ✅ Embeddings
- ✅ Time-series data

**Output:**
- ✅ Text
- ✅ Images (generation)
- ✅ Audio (TTS)
- ✅ Structured data

### Integration Points

1. **Standard Inference API**: Use multimodal through existing `InferenceRequest`
2. **Native Multimodal API**: Direct `MultimodalRequest` for advanced use cases
3. **Streaming Support**: Real-time multimodal streaming
4. **Multi-tenant**: Tenant-aware multimodal processing

## Usage Examples

### Example 1: Image + Text (Vision QA)

```java
@Inject
MultimodalInferenceService multimodalService;

// Create multimodal request
MultimodalContent text = MultimodalContent.ofText(
    "What is shown in this image?"
);

MultimodalContent image = MultimodalContent.ofBase64Image(
    imageBytes,
    "image/jpeg"
);

MultimodalRequest request = MultimodalRequest.builder()
    .model("gpt-4-vision")
    .inputs(text, image)
    .outputConfig(MultimodalRequest.OutputConfig.builder()
        .maxTokens(500)
        .temperature(0.7)
        .build())
    .build();

// Execute
Uni<MultimodalResponse> response = multimodalService.infer(request);

response.subscribe().with(resp -> {
    System.out.println("Answer: " + resp.getOutputs()[0].getText());
    System.out.println("Tokens used: " + resp.getUsage().getTotalTokens());
});
```

### Example 2: Document Analysis

```java
// Upload PDF for analysis
MultimodalContent document = MultimodalContent.ofDocument(
    pdfBytes,
    "pdf",
    "application/pdf"
);

MultimodalContent query = MultimodalContent.ofText(
    "Summarize the key points from this document"
);

MultimodalRequest request = MultimodalRequest.builder()
    .model("claude-3-opus")
    .inputs(query, document)
    .outputConfig(MultimodalRequest.OutputConfig.builder()
        .maxTokens(2000)
        .build())
    .build();

multimodalService.infer(request)
    .subscribe().with(resp -> {
        String summary = resp.getOutputs()[0].getText();
        System.out.println("Summary: " + summary);
    });
```

### Example 3: Audio Transcription

```java
// Transcribe audio
MultimodalContent audio = MultimodalContent.ofAudio(
    audioBytes,
    "audio/wav"
);

MultimodalContent instruction = MultimodalContent.ofText(
    "Transcribe this audio to text"
);

MultimodalRequest request = MultimodalRequest.builder()
    .model("whisper-large")
    .inputs(instruction, audio)
    .outputConfig(MultimodalRequest.OutputConfig.builder()
        .maxTokens(1000)
        .build())
    .build();

multimodalService.infer(request)
    .subscribe().with(resp -> {
        String transcription = resp.getOutputs()[0].getText();
        System.out.println("Transcription: " + transcription);
    });
```

### Example 4: Integration with Standard Inference

```java
@Inject
MultimodalEngineIntegration integration;

// Standard inference request with images
InferenceRequest request = InferenceRequest.builder()
    .requestId("req-123")
    .model("gpt-4-vision")
    .addMessage("user", "What's in this image?")
    .parameter("images", List.of("data:image/jpeg;base64,..."))
    .parameter("max_tokens", 500)
    .build();

// Execute multimodal inference
Uni<InferenceResponse> response = integration.executeMultimodal(request);

response.subscribe().with(resp -> {
    System.out.println("Response: " + resp.getContent());
});
```

### Example 5: Streaming Multimodal

```java
// Streaming image analysis
MultimodalRequest request = MultimodalRequest.builder()
    .model("gpt-4-vision")
    .inputs(
        MultimodalContent.ofText("Describe this image in detail"),
        MultimodalContent.ofBase64Image(imageBytes, "image/jpeg")
    )
    .outputConfig(MultimodalRequest.OutputConfig.builder()
        .stream(true)
        .build())
    .build();

multimodalService.inferStream(request)
    .subscribe().with(
        chunk -> System.out.print(chunk.getText()),
        error -> error.printStackTrace(),
        () -> System.out.println("\nDone!")
    );
```

## Configuration

### Basic Configuration

```properties
# Enable multimodal inference
gollek.multimodal.enabled=true

# Default model for multimodal
gollek.multimodal.default-model=gpt-4-vision

# Timeout settings
gollek.multimodal.timeout-ms=30000

# Max concurrent requests
gollek.multimodal.max-concurrent=10
```

### Provider-Specific Configuration

```properties
# OpenAI Vision
gollek.multimodal.openai.api-key=sk-...
gollek.multimodal.openai.base-url=https://api.openai.com

# Google Gemini
gollek.multimodal.gemini.api-key=...
gollek.multimodal.gemini.base-url=https://generativelanguage.googleapis.com

# Anthropic Claude
gollek.multimodal.anthropic.api-key=sk-ant-...
gollek.multimodal.anthropic.base-url=https://api.anthropic.com
```

## Processor Implementation

### Creating a Custom Processor

```java
@ApplicationScoped
public class CustomVisionProcessor implements MultimodalProcessor {

    @Override
    public String getProcessorId() {
        return "custom-vision";
    }

    @Override
    public boolean isAvailable() {
        // Check if backend is reachable
        return true;
    }

    @Override
    public Uni<MultimodalResponse> process(MultimodalRequest request) {
        return Uni.createFrom().item(() -> {
            // Process image + text
            MultimodalContent output = MultimodalContent.ofText(
                analyzeImage(request.getInputs())
            );
            
            return MultimodalResponse.builder()
                .requestId(request.getRequestId())
                .outputs(output)
                .usage(new MultimodalResponse.Usage(100, 50))
                .durationMs(System.currentTimeMillis() - startTime)
                .build();
        });
    }

    @Override
    public Multi<MultimodalResponse> processStream(MultimodalRequest request) {
        return process(request).onItem().transformToMulti(Multi.createFrom()::just);
    }

    private String analyzeImage(MultimodalRequest request) {
        // Implement vision processing
        return "Image analysis result";
    }
}
```

## Performance Optimization

### Batching

```java
// Batch multiple requests
List<MultimodalRequest> requests = getRequests();

Uni<List<MultimodalResponse>> responses = Multi.createFrom().iterable(requests)
    .onItem().transformToUniAndConcatenate(req -> multimodalService.infer(req))
    .collect().asList();
```

### Caching

```java
@Inject
SemanticResponseCache cache;

public Uni<MultimodalResponse> inferWithCache(MultimodalRequest request) {
    String cacheKey = generateCacheKey(request);
    
    return cache.get(cacheKey)
        .onItem().transformToUni(cached -> {
            if (cached != null) {
                return Uni.createFrom().item(cached);
            }
            return multimodalService.infer(request)
                .onItem().invoke(resp -> cache.put(cacheKey, resp));
        });
}
```

### Load Balancing

```java
@Inject
AdaptiveLoadBalancer loadBalancer;

public MultimodalProcessor selectProcessor(MultimodalRequest request) {
    return loadBalancer.selectLeastLoaded(
        processors.stream()
            .filter(p -> p.isAvailable())
            .collect(Collectors.toList())
    );
}
```

## Troubleshooting

### Common Issues

#### "No processor available for model"

**Solution**: Ensure the model name matches a registered processor ID or add pattern matching in `selectProcessor()`.

#### "Timeout during multimodal inference"

**Solutions**:
1. Increase timeout: `request.setTimeoutMs(60000)`
2. Reduce image resolution
3. Use streaming for large outputs

#### "Out of memory with large images"

**Solutions**:
1. Resize images before sending
2. Use URI references instead of base64
3. Implement chunked processing

## Resources

- [Multimodal API Reference](./api/multimodal)
- [Supported Models](./models/multimodal)
- [Processor Implementation Guide](./guides/processor-implementation)

---

[Back to Inference Engine](../inference-engine) &nbsp; [View Executors](../executors)
