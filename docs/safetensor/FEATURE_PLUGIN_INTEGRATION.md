# Safetensor Feature Plugin Integration Guide

## Overview

The Safetensor Feature Plugin system integrates with existing Safetensor modules to provide domain-specific capabilities. This guide shows how the integration works.

## Integration Architecture

```
Existing Safetensor Modules          Feature Plugins
┌────────────────────────┐          ┌────────────────────────┐
│ WhisperEngine          │─────────▶│ AudioFeaturePlugin     │
│ (gollek-safetensor-    │          │ (gollek-plugin-        │
│  audio)                │          │  feature-audio)        │
│                        │          └────────────────────────┘
│ SpeechT5Engine         │─────────▶
│ (gollek-safetensor-    │
│  audio)                │          ┌────────────────────────┐
│                        │─────────▶│ VisionFeaturePlugin    │
│ AudioProcessor         │          │ (gollek-plugin-        │
│ (gollek-safetensor-    │          │  feature-vision)       │
│  audio)                │          └────────────────────────┘
└────────────────────────┘
                                ┌────────────────────────┐
                                │ TextFeaturePlugin      │
                                │ (gollek-plugin-        │
                                │  feature-text)         │
                                └────────────────────────┘
```

## CDI Integration

### FeaturePluginProducer

The `FeaturePluginProducer` class wires existing engines with feature plugins:

```java
@ApplicationScoped
public class FeaturePluginProducer {

    @Inject
    WhisperEngine whisperEngine;

    @Inject
    SpeechT5Engine speechT5Engine;

    @Produces
    @Singleton
    public AudioFeaturePlugin produceAudioFeaturePlugin() {
        return new AudioFeaturePlugin(whisperEngine, speechT5Engine);
    }

    @Produces
    @Singleton
    public VisionFeaturePlugin produceVisionFeaturePlugin() {
        return new VisionFeaturePlugin();
    }

    @Produces
    @Singleton
    public TextFeaturePlugin produceTextFeaturePlugin() {
        return new TextFeaturePlugin();
    }
}
```

## Usage

### Inject Feature Plugin

```java
@Inject
AudioFeaturePlugin audioFeature;

// Use audio feature
byte[] audio = loadAudio("speech.wav");
Uni<String> transcription = audioFeature.transcribeAudio(audio);
```

### Via Safetensor Runner

```java
@Inject
RunnerPluginRegistry runnerRegistry;

// Create session with audio feature enabled
Map<String, Object> config = Map.of(
    "features", Map.of(
        "audio", Map.of("enabled", true)
    )
);

Optional<RunnerSession> session = runnerRegistry.createSession(
    "whisper-large-v3.safetensors",
    config
);

// Process audio through session
```

## Existing Module Integration

### WhisperEngine Integration

The `AudioFeaturePlugin` uses the existing `WhisperEngine`:

```java
public class AudioFeaturePlugin implements SafetensorFeaturePlugin {
    private final WhisperEngine whisperEngine;  // Injected
    
    private Uni<String> transcribeAudio(byte[] audioData) {
        // Delegates to existing WhisperEngine
        return whisperEngine.transcribe(audioData, language, task);
    }
}
```

**Existing WhisperEngine features used**:
- ✅ Multi-format audio decoding
- ✅ Automatic resampling to 16kHz
- ✅ Voice activity detection
- ✅ Streaming transcription
- ✅ Word-level timestamps
- ✅ Language detection

### SpeechT5Engine Integration

```java
public class AudioFeaturePlugin implements SafetensorFeaturePlugin {
    private final SpeechT5Engine speechT5Engine;  // Injected
    
    private Uni<byte[]> synthesizeSpeech(String text) {
        // Delegates to existing SpeechT5Engine
        AudioConfig config = new AudioConfig();
        return speechT5Engine.synthesize(text, defaultModel, Path.of("models"), config);
    }
}
```

**Existing SpeechT5Engine features used**:
- ✅ HiFi-GAN vocoder
- ✅ Multiple voice presets
- ✅ Speaker embedding support
- ✅ Prosody control

## Configuration

### Application Properties

```yaml
# Existing Safetensor configuration
gollek:
  audio:
    whisper:
      beam-size: 5
      temperature: 0.0
      language: en
      task: transcribe
      vad-enabled: true
    tts:
      default-voice: alloy
      speed: 1.0
  
  # Feature plugin configuration
  runners:
    safetensor-runner:
      features:
        audio:
          enabled: true
          default_model: whisper-large-v3
```

## Testing

### Unit Test Example

```java
@QuarkusTest
class AudioFeaturePluginTest {

    @Inject
    AudioFeaturePlugin audioFeature;

    @Test
    void shouldTranscribeAudio() {
        // Arrange
        byte[] audio = loadTestAudio();
        
        // Act
        Object result = audioFeature.process(audio);
        
        // Assert
        assertNotNull(result);
        assertTrue(result instanceof Uni);
    }

    @Test
    void shouldSynthesizeSpeech() {
        // Arrange
        String text = "Hello world";
        
        // Act
        Object result = audioFeature.process(text);
        
        // Assert
        assertNotNull(result);
    }
}
```

### Integration Test Example

```java
@QuarkusTest
class FeaturePluginIntegrationTest {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Test
    void shouldCreateSessionWithAudioFeature() {
        // Arrange
        Map<String, Object> config = Map.of(
            "features", Map.of("audio", Map.of("enabled", true))
        );
        
        // Act
        Optional<RunnerSession> session = runnerRegistry.createSession(
            "whisper-large-v3.safetensors",
            config
        );
        
        // Assert
        assertTrue(session.isPresent());
    }
}
```

## Deployment

### Maven Dependencies

```xml
<!-- Safetensor Runner -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-runner-safetensor</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Feature Plugins -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-feature-audio</artifactId>
    <version>${project.version}</version>
</dependency>

<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-feature-vision</artifactId>
    <version>${project.version}</version>
</dependency>

<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-feature-text</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Existing Safetensor Modules (transitive) -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-audio</artifactId>
    <version>${project.version}</version>
</dependency>
```

## Troubleshooting

### Feature Plugin Not Found

```
Error: No producer found for AudioFeaturePlugin
```

**Solution**:
1. Ensure `FeaturePluginProducer` is in classpath
2. Check CDI bean discovery mode
3. Verify `beans.xml` exists

### Engine Not Injected

```
Error: Unsatisfied dependency for type WhisperEngine
```

**Solution**:
1. Ensure `gollek-safetensor-audio` module is in classpath
2. Check that WhisperEngine is a CDI bean (@ApplicationScoped)
3. Verify module dependencies in POM

### Feature Not Available

```
Error: Audio feature is not available
```

**Solution**:
1. Check if feature is enabled in config
2. Verify WhisperEngine/SpeechT5Engine are initialized
3. Check logs for initialization errors

## Resources

- **Feature Plugin SPI**: `plugins/runner/safetensor/gollek-plugin-runner-safetensor/src/main/java/.../SafetensorFeaturePlugin.java`
- **Feature Plugin Producer**: `plugins/runner/safetensor/gollek-plugin-runner-safetensor/src/main/java/.../FeaturePluginProducer.java`
- **Existing WhisperEngine**: `plugins/runner/safetensor/gollek-safetensor-audio/src/main/java/.../WhisperEngine.java`
- **Existing SpeechT5Engine**: `plugins/runner/safetensor/gollek-safetensor-audio/src/main/java/.../SpeechT5Engine.java`
- [FEATURE_PLUGIN_SYSTEM.md](FEATURE_PLUGIN_SYSTEM.md) - Complete usage guide
