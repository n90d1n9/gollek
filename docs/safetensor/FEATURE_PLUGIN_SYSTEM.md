# Safetensor Feature Plugin System

Domain-specific feature plugins for the Safetensor runner, enabling modular deployment of audio, vision, and text processing capabilities.

## Overview

The Safetensor Feature Plugin system provides a two-level plugin architecture:

```
SafetensorRunnerPlugin (Level 1)
    ├── AudioFeaturePlugin (Level 2)
    ├── VisionFeaturePlugin (Level 2)
    └── TextFeaturePlugin (Level 2)
```

This allows for:
- **Selective Deployment**: Deploy only needed features
- **Fine-grained Control**: Enable/disable specific capabilities
- **Resource Optimization**: Load only required models
- **Independent Updates**: Update features independently

## Feature Plugins

### 1. Audio Feature Plugin

**Artifact**: `gollek-plugin-feature-audio`

**Capabilities**:
- ✅ Speech-to-text (Whisper)
- ✅ Text-to-speech (SpeechT5)
- ✅ Audio feature extraction
- ✅ Voice activity detection

**Supported Models**:
- Whisper (tiny, base, small, medium, large, large-v3)
- SpeechT5 (TTS, ASR)
- Wav2Vec2
- Hubert

**Configuration**:
```yaml
gollek:
  runners:
    safetensor-runner:
      features:
        audio:
          enabled: true
          default_model: whisper-large-v3
          language: en
          task: transcribe  # or translate
```

**Usage**:
```java
@Inject
RunnerPluginRegistry runnerRegistry;

// Create session with audio feature
Map<String, Object> config = Map.of(
    "features", Map.of(
        "audio", Map.of(
            "enabled", true,
            "task", "transcribe"
        )
    )
);

Optional<RunnerSession> session = runnerRegistry.createSession(
    "whisper-large-v3.safetensors",
    config
);

// Process audio
byte[] audioData = loadAudio("speech.wav");
AudioFeaturePlugin.AudioInput input = AudioFeaturePlugin.AudioInput.builder()
    .audioData(audioData)
    .task("transcribe")
    .language("en")
    .build();

Object result = session.get().infer(input).await();
```

---

### 2. Vision Feature Plugin

**Artifact**: `gollek-plugin-feature-vision`

**Capabilities**:
- ✅ Image classification
- ✅ Object detection
- ✅ Image segmentation
- ✅ Visual question answering
- ✅ Image captioning

**Supported Models**:
- CLIP (ViT-B/32, ViT-L/14)
- ViT (base, large)
- DETR (ResNet-50, ResNet-101)
- LLaVA (7B, 13B)
- YOLO (v5, v8)

**Configuration**:
```yaml
gollek:
  runners:
    safetensor-runner:
      features:
        vision:
          enabled: true
          default_model: clip-vit-large
          max_image_size: 2048
```

**Usage**:
```java
// Create session with vision feature
Map<String, Object> config = Map.of(
    "features", Map.of(
        "vision", Map.of(
            "enabled", true,
            "task", "classification"
        )
    )
);

Optional<RunnerSession> session = runnerRegistry.createSession(
    "clip-vit-large.safetensors",
    config
);

// Process image
byte[] imageData = loadImage("photo.jpg");
Object result = session.get().infer(imageData).await();
```

---

### 3. Text Feature Plugin

**Artifact**: `gollek-plugin-feature-text`

**Capabilities**:
- ✅ Text generation (LLMs)
- ✅ Text classification
- ✅ Named entity recognition
- ✅ Question answering
- ✅ Text embedding

**Supported Models**:
- Llama 2/3 (7B, 13B, 70B)
- Mistral (7B)
- Mixtral MoE (8x7B)
- BERT (base, large)
- RoBERTa (base, large)

**Configuration**:
```yaml
gollek:
  runners:
    safetensor-runner:
      features:
        text:
          enabled: true
          default_model: llama-3-8b
          max_context: 4096
          temperature: 0.7
```

**Usage**:
```java
// Create session with text feature
Map<String, Object> config = Map.of(
    "features", Map.of(
        "text", Map.of(
            "enabled", true,
            "temperature", 0.7
        )
    )
);

Optional<RunnerSession> session = runnerRegistry.createSession(
    "llama-3-8b.safetensors",
    config
);

// Process text
InferenceRequest request = InferenceRequest.builder()
    .model("llama-3-8b")
    .message(Message.user("Explain quantum computing"))
    .build();

InferenceResponse response = session.get().infer(request).await();
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           SafetensorRunnerPlugin                        │
│  - Manages feature plugins                              │
│  - Routes requests to appropriate feature               │
│  - Handles cross-feature coordination                   │
└─────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ AudioFeature     │ │ VisionFeature    │ │ TextFeature      │
│ Plugin           │ │ Plugin           │ │ Plugin           │
│                  │ │                  │ │                  │
│ - Whisper        │ │ - CLIP           │ │ - Llama          │
│ - SpeechT5       │ │ - ViT            │ │ - Mistral        │
│ - Wav2Vec2       │ │ - DETR           │ │ - BERT           │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## Deployment Scenarios

### Scenario 1: Audio-Only Deployment

**Use Case**: Speech transcription service

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      features:
        audio:
          enabled: true
        vision:
          enabled: false  # Not deployed
        text:
          enabled: false  # Not deployed
```

**Size**: ~500 MB (Whisper models only)

---

### Scenario 2: Vision-Only Deployment

**Use Case**: Image classification service

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      features:
        audio:
          enabled: false
        vision:
          enabled: true
        text:
          enabled: false
```

**Size**: ~800 MB (ViT, CLIP models)

---

### Scenario 3: Multi-Modal Deployment

**Use Case**: LLaVA visual question answering

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      features:
        audio:
          enabled: false
        vision:
          enabled: true
        text:
          enabled: true  # For LLM component
```

**Size**: ~15 GB (LLaVA + vision encoder)

---

### Scenario 4: Full Deployment

**Use Case**: Development/testing environment

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      features:
        audio:
          enabled: true
        vision:
          enabled: true
        text:
          enabled: true
```

**Size**: ~20 GB (all features)

---

## Feature Plugin SPI

### SafetensorFeaturePlugin Interface

```java
public interface SafetensorFeaturePlugin {
    String id();
    String name();
    String version();
    String description();
    void initialize(Map<String, Object> config);
    boolean isAvailable();
    int priority();
    Set<String> supportedModels();
    Set<String> supportedInputTypes();
    Set<String> supportedOutputTypes();
    Object process(Object input);
    Map<String, Object> metadata();
    void shutdown();
    boolean isHealthy();
}
```

### Creating a Custom Feature Plugin

```java
public class CustomFeaturePlugin implements SafetensorFeaturePlugin {
    
    @Override
    public String id() {
        return "custom-feature";
    }
    
    @Override
    public String name() {
        return "Custom Processing";
    }
    
    @Override
    public Set<String> supportedModels() {
        return Set.of("custom-model-v1", "custom-model-v2");
    }
    
    @Override
    public Object process(Object input) {
        // Custom processing logic
        return processCustom(input);
    }
    
    // ... other methods
}
```

---

## Configuration Reference

### Full Configuration Example

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      backend: direct
      device: cuda
      dtype: f16
      features:
        audio:
          enabled: true
          default_model: whisper-large-v3
          language: en
          task: transcribe
          whisper:
            beam_size: 5
            best_of: 5
            temperature: 0.0
          speecht5:
            speaker_id: default
            speed: 1.0
        vision:
          enabled: true
          default_model: clip-vit-large
          max_image_size: 2048
          clip:
            projection_dim: 768
          detr:
            score_threshold: 0.5
        text:
          enabled: true
          default_model: llama-3-8b
          max_context: 4096
          temperature: 0.7
          top_p: 0.9
          llama:
            n_gpu_layers: -1
            flash_attention: true
```

---

## Monitoring

### Health Checks

```bash
# Get feature health status
curl http://localhost:8080/api/v1/runners/safetensor-runner/features/health

# Response:
{
  "audio-feature": {
    "healthy": true,
    "whisper_available": true,
    "speecht5_available": true
  },
  "vision-feature": {
    "healthy": true,
    "clip_available": true
  },
  "text-feature": {
    "healthy": true,
    "llama_available": true
  }
}
```

### Metrics

```bash
# Get feature metrics
curl http://localhost:8080/api/v1/runners/safetensor-runner/features/metrics

# Response:
{
  "audio-feature": {
    "requests_total": 1000,
    "avg_latency_ms": 250,
    "whisper_requests": 800,
    "speecht5_requests": 200
  },
  "vision-feature": {
    "requests_total": 500,
    "avg_latency_ms": 150
  },
  "text-feature": {
    "requests_total": 2000,
    "avg_latency_ms": 500
  }
}
```

---

## Troubleshooting

### Feature Not Available

```
Error: Audio feature is not available
```

**Solution**:
1. Check if feature is enabled in config
2. Verify model files exist
3. Check backend health

### Model Not Found

```
Error: Model whisper-large-v3 not found
```

**Solution**:
1. Download model: `gollek-cli model download whisper-large-v3`
2. Verify model path in config
3. Check model file permissions

### Out of Memory

```
Error: CUDA out of memory
```

**Solution**:
1. Disable unused features
2. Reduce model size
3. Enable quantization
4. Reduce batch size

---

## Resources

- **Feature Plugin SPI**: `plugins/runner/safetensor/gollek-plugin-runner-safetensor/src/main/java/.../SafetensorFeaturePlugin.java`
- **Audio Feature**: `plugins/runner/safetensor/gollek-plugin-feature-audio/`
- **Vision Feature**: `plugins/runner/safetensor/gollek-plugin-feature-vision/`
- **Text Feature**: `plugins/runner/safetensor/gollek-plugin-feature-text/`
- [Safetensor Runner Plugin](../gollek-plugin-runner-safetensor/README.md)
- [Runner Plugin System](../../RUNNER_PLUGIN_SYSTEM.md)

---

## License

MIT License - See LICENSE file for details.
