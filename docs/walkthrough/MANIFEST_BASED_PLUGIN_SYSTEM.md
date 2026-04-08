# Manifest-Based Plugin System Guide

**Date**: 2026-03-25  
**Version**: 2.0.0  
**Status**: ✅ **IMPLEMENTED**

---

## Overview

The Gollek platform uses a **manifest-based plugin system** that is:

- ✅ **Flexible** - Plug in any plugin needed without hardcoded mappings
- ✅ **Portable** - Deploy as standalone (all-in-one JAR) or microservice
- ✅ **Agnostic** - No hardcoded plugin IDs or capabilities
- ✅ **Extensible** - Add new plugins without modifying core code

Plugins declare their capabilities, dependencies, and metadata via JAR manifest entries, enabling dynamic discovery and capability-based routing.

---

## Plugin Manifest Format

### Required Entries

```manifest
Plugin-Id: gguf-runner
Plugin-Type: runner
Plugin-Version: 2.0.0
Plugin-Name: GGUF Runner
Plugin-Provider: tech.kayys.gollek.plugin.runner.gguf.GGUFRunnerPlugin
```

### Optional Entries

```manifest
# Capabilities provided by this plugin
Plugin-Capabilities: gguf-inference, llama-architecture, mistral-architecture

# Plugin dependencies
Plugin-Dependencies: kernel-plugin

# Supported deployment modes
Plugin-Deployment: standalone,microservice

# Author and vendor information
Plugin-Author: Gollek Team
Plugin-Vendor: Kayys.tech
Plugin-License: MIT
Plugin-Homepage: https://gollek.ai
Plugin-Documentation: https://gollek.ai/docs/gguf-runner
Plugin-Repository: https://github.com/gollek-ai/gollek

# GPU requirements
Plugin-GPU-Requirement: CUDA 11.0+
Plugin-Minimum-Compute-Capability: 6.0
Plugin-Minimum-Memory: 4GB

# Performance metadata
Plugin-Performance-Speedup: 2-3x
Plugin-Performance-Memory-Overhead: 100MB
```

---

## Example Plugin Manifests

### GGUF Runner Plugin

```manifest
Manifest-Version: 1.0
Plugin-Id: gguf-runner
Plugin-Type: runner
Plugin-Version: 2.0.0
Plugin-Name: GGUF Runner
Plugin-Description: GGUF format support using llama.cpp
Plugin-Provider: tech.kayys.gollek.plugin.runner.gguf.GGUFRunnerPlugin
Plugin-Capabilities: gguf-inference, llama-architecture, mistral-architecture, qwen-architecture, gemma-architecture
Plugin-Dependencies: 
Plugin-Deployment: standalone,microservice
Plugin-Author: Gollek Team
Plugin-Vendor: Kayys.tech
Plugin-License: MIT
Plugin-Performance-Speedup: 1x (baseline)
Plugin-Performance-Memory-Overhead: 50-100MB
```

### CUDA Kernel Plugin

```manifest
Manifest-Version: 1.0
Plugin-Id: cuda-kernel
Plugin-Type: kernel
Plugin-Version: 2.0.0
Plugin-Name: CUDA Kernel
Plugin-Description: NVIDIA CUDA kernel implementations
Plugin-Provider: tech.kayys.gollek.plugin.kernel.cuda.CudaKernelPlugin
Plugin-Capabilities: cuda-acceleration, flash-attention-2, flash-attention-3, paged-attention
Plugin-Dependencies: 
Plugin-Deployment: microservice,hybrid
Plugin-GPU-Requirement: NVIDIA GPU, CUDA 11.0+
Plugin-Minimum-Compute-Capability: 6.0
Plugin-Minimum-Memory: 4GB
Plugin-Performance-Speedup: 5-10x (vs CPU)
Plugin-Performance-Memory-Overhead: 50-100MB
```

### FlashAttention-3 Optimization Plugin

```manifest
Manifest-Version: 1.0
Plugin-Id: flash-attention-3
Plugin-Type: optimization
Plugin-Version: 2.0.0
Plugin-Name: FlashAttention-3
Plugin-Description: FlashAttention-3 kernel for Hopper+ GPUs
Plugin-Provider: tech.kayys.gollek.plugin.optimization.fa3.FlashAttention3Plugin
Plugin-Capabilities: flash-attention-3, optimized-attention
Plugin-Dependencies: cuda-kernel
Plugin-Deployment: microservice,hybrid
Plugin-GPU-Requirement: NVIDIA Hopper+ (H100), CUDA 12.0+
Plugin-Minimum-Compute-Capability: 9.0
Plugin-Minimum-Memory: 16GB
Plugin-Performance-Speedup: 2-3x (vs standard attention)
Plugin-Performance-Memory-Overhead: 100-200MB
```

### OpenAI Provider Plugin

```manifest
Manifest-Version: 1.0
Plugin-Id: openai-provider
Plugin-Type: provider
Plugin-Version: 2.0.0
Plugin-Name: OpenAI Provider
Plugin-Description: OpenAI API provider for GPT models
Plugin-Provider: tech.kayys.gollek.plugin.provider.openai.OpenAiProvider
Plugin-Capabilities: openai-api, gpt-4, gpt-3.5-turbo, embeddings
Plugin-Dependencies: 
Plugin-Deployment: standalone,microservice,hybrid
Plugin-Author: Gollek Team
Plugin-Vendor: Kayys.tech
Plugin-License: MIT
```

---

## Building Plugins with Maven

### POM Configuration

```xml
<project>
    ...
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.5.0</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addDefaultImplementationEntries>true</addDefaultImplementationEntries>
                        </manifest>
                        <manifestEntries>
                            <!-- Required -->
                            <Plugin-Id>gguf-runner</Plugin-Id>
                            <Plugin-Type>runner</Plugin-Type>
                            <Plugin-Version>${project.version}</Plugin-Version>
                            <Plugin-Name>GGUF Runner</Plugin-Name>
                            <Plugin-Provider>tech.kayys.gollek.plugin.runner.gguf.GGUFRunnerPlugin</Plugin-Provider>
                            
                            <!-- Optional -->
                            <Plugin-Capabilities>gguf-inference, llama-architecture, mistral-architecture</Plugin-Capabilities>
                            <Plugin-Deployment>standalone,microservice</Plugin-Deployment>
                            <Plugin-Author>Gollek Team</Plugin-Author>
                            <Plugin-Vendor>Kayys.tech</Plugin-Vendor>
                            <Plugin-License>MIT</Plugin-License>
                            <Plugin-Performance-Speedup>1x (baseline)</Plugin-Performance-Speedup>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

---

## Capability Discovery

### How It Works

1. **Plugin Loading**: Plugins are loaded via ServiceLoader or from plugin directory
2. **Manifest Parsing**: JAR manifest is parsed to extract PluginDescriptor
3. **Capability Indexing**: Capabilities are indexed for fast lookup
4. **Capability Checking**: Applications check for capabilities, not specific plugins

### Example Usage

```java
@Inject
PluginAvailabilityChecker pluginChecker;

// Check if capability is available
if (pluginChecker.hasCapability("gguf-inference")) {
    System.out.println("GGUF inference is available");
}

// Get plugins that provide capability
List<PluginDescriptor> plugins = pluginChecker.getPluginsForCapability("cuda-acceleration");
for (PluginDescriptor plugin : plugins) {
    System.out.printf("Plugin: %s, Version: %s, Speedup: %s\n",
        plugin.getId(),
        plugin.getVersion(),
        plugin.getMetadata("speedup"));
}

// Get required plugins for capability
String required = pluginChecker.getRequiredPluginsForCapability("flash-attention-3");
System.out.println("Required plugins:\n" + required);

// List all available capabilities
Set<String> capabilities = pluginChecker.getAvailableCapabilities();
System.out.println("Available capabilities: " + capabilities);
```

---

## Deployment Modes

### STANDALONE Mode

**Description**: All plugins built-in, compiled in one JAR

**Use Case**: Simple deployment, no dynamic loading

**Configuration**:
```bash
-Dgollek.deployment.mode=standalone
```

**Plugin Loading**:
- Only ServiceLoader discovery
- No plugin directory scanning
- All plugins must be included at build time

### MICROSERVICE Mode

**Description**: Dynamic plugin loading from plugin directory

**Use Case**: Flexible deployment, hot-reload support

**Configuration**:
```bash
-Dgollek.deployment.mode=microservice
-Dgollek.plugin.directory=~/.gollek/plugins
```

**Plugin Loading**:
- ServiceLoader discovery
- Plugin directory scanning
- Hot-reload support

### HYBRID Mode (Default)

**Description**: Built-in + dynamic plugins

**Use Case**: Best of both worlds

**Configuration**:
```bash
-Dgollek.deployment.mode=hybrid
```

**Plugin Loading**:
- ServiceLoader discovery (built-in)
- Plugin directory scanning (dynamic)
- Maximum flexibility

---

## Error Handling

### No Plugins Available

```java
try {
    if (!pluginChecker.hasProviders() && !pluginChecker.hasRunnerPlugins()) {
        throw new NoPluginsAvailableException(
            pluginChecker.getPluginDirectory(),
            pluginChecker.getDeploymentMode().toString());
    }
} catch (NoPluginsAvailableException e) {
    System.err.println(e.getMessage());
    System.err.println(e.getInstallationInstructions());
    System.exit(1);
}
```

### Plugin Not Available

```java
try {
    if (!pluginChecker.hasProvider("openai")) {
        throw new PluginNotAvailableException(
            "provider",
            "openai",
            pluginChecker.getPluginDirectory());
    }
} catch (PluginNotAvailableException e) {
    System.err.println(e.getMessage());
    System.err.println(e.getInstallationInstructions());
    System.exit(1);
}
```

### Capability Not Available

```java
try {
    if (!pluginChecker.hasCapability("cuda-acceleration")) {
        System.err.println(pluginChecker.getCapabilityNotAvailableError("cuda-acceleration"));
        System.exit(1);
    }
} catch (Exception e) {
    // Handle error
}
```

---

## Best Practices

### 1. Declare All Capabilities

```manifest
# Good - comprehensive capabilities
Plugin-Capabilities: gguf-inference, llama-architecture, mistral-architecture, qwen-architecture

# Bad - limited capabilities
Plugin-Capabilities: gguf-inference
```

### 2. Specify Dependencies

```manifest
# Good - declares dependencies
Plugin-Id: flash-attention-3
Plugin-Dependencies: cuda-kernel

# Bad - no dependencies declared
Plugin-Id: flash-attention-3
Plugin-Dependencies:
```

### 3. Include Performance Metadata

```manifest
# Good - includes performance info
Plugin-Performance-Speedup: 2-3x
Plugin-Performance-Memory-Overhead: 100-200MB

# Bad - no performance info
```

### 4. Support Multiple Deployment Modes

```manifest
# Good - supports all modes
Plugin-Deployment: standalone,microservice,hybrid

# Bad - limited deployment
Plugin-Deployment: microservice
```

### 5. Provide GPU Requirements

```manifest
# Good - clear GPU requirements
Plugin-GPU-Requirement: NVIDIA Hopper+ (H100), CUDA 12.0+
Plugin-Minimum-Compute-Capability: 9.0
Plugin-Minimum-Memory: 16GB

# Bad - vague requirements
Plugin-GPU-Requirement: NVIDIA GPU
```

---

## Migration Guide

### From Hardcoded to Manifest-Based

**Before (Hardcoded)**:
```java
// In PluginAvailabilityChecker
private String getRequiredPluginsForCapability(String capability) {
    return switch (capability.toLowerCase()) {
        case "gguf-inference" -> "   • gguf-runner\n";
        case "cuda-acceleration" -> "   • cuda-kernel\n";
        // ... hardcoded mappings
    };
}
```

**After (Manifest-Based)**:
```java
// Plugin declares capabilities in manifest
Plugin-Capabilities: gguf-inference, llama-architecture

// PluginAvailabilityChecker reads from manifest
public String getRequiredPluginsForCapability(String capability) {
    for (PluginDescriptor plugin : discoveredPlugins.values()) {
        if (plugin.hasCapability(capability)) {
            StringBuilder required = new StringBuilder();
            required.append("   • ").append(plugin.getId());
            
            // Add dependencies from manifest
            for (String dep : plugin.getDependencies()) {
                required.append("\n   • ").append(dep);
            }
            
            return required.toString();
        }
    }
    return "";
}
```

---

## Troubleshooting

### Plugin Not Discovered

**Symptom**: Plugin JAR in directory but not discovered

**Solutions**:
1. Check manifest: `jar xf plugin.jar META-INF/MANIFEST.MF && cat META-INF/MANIFEST.MF`
2. Verify required entries: `Plugin-Id`, `Plugin-Type`, `Plugin-Provider`
3. Check deployment mode compatibility
4. Restart application

### Capability Not Found

**Symptom**: `hasCapability()` returns false

**Solutions**:
1. Check plugin manifest for `Plugin-Capabilities` entry
2. Verify capability name matches exactly (case-sensitive)
3. Check if plugin is loaded: `pluginChecker.getDiscoveredPlugins()`
4. Verify deployment mode compatibility

### Dependency Not Resolved

**Symptom**: Plugin fails to load due to missing dependency

**Solutions**:
1. Check `Plugin-Dependencies` in manifest
2. Install required dependencies
3. Verify dependency plugin IDs match exactly
4. Check load order (dependencies load first)

---

## Resources

- **PluginDescriptor API**: [Javadoc Link]
- **PluginAvailabilityChecker API**: [Javadoc Link]
- **Plugin Development Guide**: [Link]
- **Example Plugins**: [GitHub Link]

---

**Status**: ✅ **MANIFEST-BASED PLUGIN SYSTEM IMPLEMENTED**

The plugin system is now fully manifest-based, providing maximum flexibility, portability, and agnosticism without hardcoded mappings.
