# Gollek Plugin Developer Guide

## Creating a Standalone Gollek Plugin

This guide shows you how to create independent Gollek plugins that work with the Gollek inference engine without requiring the full Gollek platform build.

## Quick Start

### 1. Create Your Plugin Project

```bash
mkdir gollek-plugin-myprovider
cd gollek-plugin-myprovider
mvn archetype:generate \
  -DgroupId=com.example \
  -DartifactId=gollek-plugin-myprovider \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DinteractiveMode=false
```

### 2. Configure pom.xml

Copy the `pom-standalone.xml` template and customize:

```xml
<project>
    <groupId>com.example</groupId>
    <artifactId>gollek-plugin-myprovider</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <gollek.version>1.0.0-SNAPSHOT</gollek.version>
        <maven.compiler.release>25</maven.compiler.release>
    </properties>
    
    <!-- Add dependencies as shown in pom-standalone.xml -->
</project>
```

### 3. Implement Your Provider

```java
package com.example.plugin;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.exception.ProviderException;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;

public class MyProvider implements LLMProvider, StreamingProvider {
    
    @Override
    public String id() {
        return "my-provider";
    }
    
    @Override
    public String name() {
        return "My Provider";
    }
    
    @Override
    public String version() {
        return "1.0.0";
    }
    
    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
            .providerId(id())
            .name(name())
            .description("My custom provider")
            .version(version())
            .vendor("My Company")
            .homepage("https://example.com")
            .build();
    }
    
    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(true)
            .functionCalling(true)
            .multimodal(false)
            .maxContextTokens(8192)
            .supportedModels(Set.of("model-v1", "model-v2"))
            .build();
    }
    
    @Override
    public void initialize(ProviderConfig config) {
        // Initialize your provider with configuration
        String apiKey = config.getSecret("apiKey").orElse(null);
        String baseUrl = config.getString("baseUrl", "https://api.example.com");
        // ... setup your client
    }
    
    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return Set.of("model-v1", "model-v2").contains(modelId);
    }
    
    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        // Implement non-streaming inference
        return Uni.createFrom().item(InferenceResponse.builder()
            .requestId(request.getRequestId())
            .content("Response from my provider")
            .model(request.getModel())
            .build());
    }
    
    @Override
    public Multi<InferenceChunk> inferStream(ProviderRequest request) {
        // Implement streaming inference
        return Multi.createFrom().items(
            InferenceChunk.of(request.getRequestId(), 0, "Hello "),
            InferenceChunk.of(request.getRequestId(), 1, "World!"),
            InferenceChunk.finalChunk(request.getRequestId(), 2, "")
        );
    }
    
    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("Provider is ready"));
    }
    
    @Override
    public void shutdown() {
        // Cleanup resources
    }
}
```

### 4. Build Your Plugin

```bash
# Standard JAR (SPI dependencies provided by runtime)
mvn clean package

# Fat JAR with all dependencies (for standalone deployment)
mvn clean package -Pfat-jar

# Install to local Gollek plugins directory
mvn clean install -Pinstall-plugin
```

### 5. Deploy Your Plugin

#### Option A: Local Development
```bash
mvn install -Pinstall-plugin
# Plugin JAR copied to ~/.gollek/plugins/
```

#### Option B: Publish to Maven Repository
```bash
# Deploy to your Nexus/Artifactory
mvn deploy

# Or publish to Maven Central (requires signing)
mvn deploy -Prelease
```

#### Option C: Manual Installation
Copy the JAR to Gollek's plugins directory:
```bash
cp target/gollek-plugin-myprovider-1.0.0.jar ~/.gollek/plugins/
```

## Plugin Manifest

Your plugin JAR must include these manifest entries:

```
Plugin-Id: my-provider
Plugin-Version: 1.0.0
Plugin-Class: com.example.plugin.MyProvider
Plugin-Provider: com.example
Plugin-Description: My custom provider for Gollek
```

The `maven-jar-plugin` configuration in `pom-standalone.xml` adds these automatically.

## Configuration

Users configure your plugin via Gollek's configuration:

```yaml
gollek:
  providers:
    my-provider:
      enabled: true
      apiKey: "your-api-key"
      baseUrl: "https://api.example.com"
      timeout: 30s
```

Or via environment variables:
```bash
export GOLLEK_PROVIDERS_MY_PROVIDER_API_KEY="your-api-key"
export GOLLEK_PROVIDERS_MY_PROVIDER_BASE_URL="https://api.example.com"
```

## Best Practices

### 1. Dependency Management
- Mark Gollek SPI dependencies as `provided`
- Keep your plugin lightweight
- Avoid bundling Gollek libraries in your JAR

### 2. Error Handling
```java
@Override
public Uni<InferenceResponse> infer(ProviderRequest request) {
    try {
        // Your implementation
    } catch (AuthenticationException e) {
        return Uni.createFrom().failure(
            new ProviderException.ProviderAuthenticationException(
                id(), "Authentication failed: " + e.getMessage()));
    } catch (Exception e) {
        return Uni.createFrom().failure(
            new ProviderException.ProviderExecutionException(
                id(), "Execution failed: " + e.getMessage()));
    }
}
```

### 3. Health Checks
```java
@Override
public Uni<ProviderHealth> health() {
    if (!initialized) {
        return Uni.createFrom().item(ProviderHealth.unhealthy("Not initialized"));
    }
    
    if (!isApiKeyValid()) {
        return Uni.createFrom().item(ProviderHealth.degraded("API key invalid"));
    }
    
    return Uni.createFrom().item(ProviderHealth.healthy("Ready"));
}
```

### 4. Logging
```java
private static final Logger LOG = Logger.getLogger(MyProvider.class);

@Override
public void initialize(ProviderConfig config) {
    LOG.infof("Initializing %s provider (version %s)", name(), version());
    // ...
    LOG.info("Provider initialized successfully");
}
```

### 5. Testing
```java
@Test
void testInference() {
    MyProvider provider = new MyProvider();
    provider.initialize(testConfig());
    
    ProviderRequest request = ProviderRequest.builder()
        .model("model-v1")
        .message(Message.user("Hello"))
        .build();
    
    InferenceResponse response = provider.inferBlocking(request);
    
    assertNotNull(response);
    assertNotNull(response.getContent());
}
```

## Plugin Types

### LLM Provider (Text Generation)
Implement `LLMProvider` for standard text generation.

### Streaming Provider
Implement `StreamingProvider` for streaming responses (recommended).

### Embedding Provider
Implement embedding generation (future SPI extension).

## Version Compatibility

Specify compatible Gollek versions in your plugin metadata:

```java
@PluginMetadata(
    id = "my-provider",
    version = "1.0.0",
    gollekVersion = ">=1.0.0 <2.0.0"
)
```

## Publishing to Maven Central

1. **Get a Sonatype account**: https://issues.sonatype.org
2. **Create a JIRA ticket** for your namespace
3. **Configure GPG signing** in `pom.xml`
4. **Deploy**:
   ```bash
   mvn deploy -Prelease
   ```

## Example Plugins

- **gollek-plugin-openai**: OpenAI GPT models
- **gollek-plugin-anthropic**: Anthropic Claude models
- **gollek-plugin-google**: Google Gemini models
- **gollek-plugin-ollama**: Local Ollama models

## Troubleshooting

### Plugin Not Loading
- Check manifest entries are correct
- Verify plugin class is public and has no-arg constructor
- Check Gollek logs for loading errors

### Class Not Found Errors
- Ensure SPI dependencies are marked as `provided`
- Don't bundle Gollek libraries in your JAR

### Configuration Not Applied
- Use `ProviderConfig` methods to read configuration
- Check environment variable naming convention

## Support

- **Documentation**: https://wayang.ai/docs/plugins
- **Issues**: https://github.com/wayang-ai/gollek/issues
- **Discussions**: https://github.com/wayang-ai/gollek/discussions

## License

MIT License - See LICENSE file for details.
