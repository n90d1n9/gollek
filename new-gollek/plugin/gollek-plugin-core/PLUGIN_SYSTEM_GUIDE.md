# Gollek Plugin System - Cloud Provider Plugin Architecture

## 🎯 Overview

The Gollek Plugin System enables dynamic loading of cloud provider plugins from JAR files stored in `~/.gollek/plugins/`. This allows you to add/remove cloud providers without recompiling the main application.

## 📁 Architecture

```
~/.gollek/plugins/
├── openai-provider.jar          # OpenAI cloud provider
├── anthropic-provider.jar       # Anthropic cloud provider
├── google-provider.jar          # Google Cloud provider
└── custom-provider.jar          # Your custom provider
```

## 🔧 Components

### 1. JarPluginLoader

Located in: `inference-gollek/core/gollek-plugin-core/src/main/java/tech/kayys/gollek/plugin/core/JarPluginLoader.java`

**Features**:
- ✅ Auto-discovery of JAR files in `~/.gollek/plugins/`
- ✅ Isolated ClassLoader per plugin
- ✅ Hot-reload on file changes
- ✅ Plugin lifecycle management (load/unload)

**Usage**:
```java
JarPluginLoader loader = new JarPluginLoader();
List<GollekPlugin> plugins = loader.loadAll();

// Load specific JAR
GollekPlugin plugin = loader.loadFromJar(Paths.get("plugin.jar"));

// Unload plugin
loader.unload("plugin-id");
```

### 2. PluginManager (Enhanced)

Located in: `inference-gollek/core/gollek-plugin-core/src/main/java/tech/kayys/gollek/plugin/core/PluginManager.java`

**Enhanced with**:
- JAR plugin loading integration
- Hot-reload support
- Unified management of CDI + JAR plugins

**Configuration**:
```bash
# Set custom plugin directory
java -Dgollek.plugin.directory=/path/to/plugins ...
```

### 3. Plugin Descriptor (plugin.json)

Optional manifest file inside JAR:

```json
{
  "id": "openai-cloud-provider",
  "name": "OpenAI Cloud Provider",
  "version": "1.0.0",
  "description": "Cloud provider for OpenAI models",
  "provider": "Kayys.tech",
  "mainClass": "tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProviderPlugin",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming"],
  "config": {
    "apiKey": {"required": true, "type": "string"},
    "baseUrl": {"required": false, "type": "string", "default": "https://api.openai.com/v1"}
  }
}
```

## 📦 Creating a Cloud Provider Plugin

### Step 1: Create Plugin Project

```xml
<!-- pom.xml -->
<project>
    <groupId>tech.kayys.gollek.plugins</groupId>
    <artifactId>openai-cloud-provider</artifactId>
    <version>1.0.0</version>
    
    <dependencies>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-provider</artifactId>
            <version>1.0.0-SNAPSHOT</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
</project>
```

### Step 2: Implement Plugin Class

```java
package tech.kayys.gollek.plugin.cloud.openai;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.plugin.GollekPlugin;

public class OpenAiCloudProviderPlugin implements GollekPlugin, LLMProvider {
    
    @Override
    public String id() {
        return "openai-cloud-provider";
    }
    
    @Override
    public String version() {
        return "1.0.0";
    }
    
    @Override
    public void initialize(PluginContext context) {
        // Initialize provider
    }
    
    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(true)
            .models(List.of("gpt-4", "gpt-3.5-turbo"))
            .build();
    }
    
    // Implement LLMProvider methods...
}
```

### Step 3: Build JAR

```bash
mvn clean package
```

### Step 4: Deploy Plugin

```bash
# Copy JAR to plugin directory
cp target/openai-cloud-provider-1.0.0.jar ~/.gollek/plugins/

# Or use custom directory
cp target/openai-cloud-provider-1.0.0.jar /path/to/plugins/
```

### Step 5: Restart or Hot-Reload

The plugin will be automatically:
- Discovered on next startup
- Hot-reloaded if file changes (watch service enabled)

## 🔄 Plugin Lifecycle

```
LOAD → INITIALIZE → START → RUNNING → STOP → UNLOAD
```

### Lifecycle Methods

```java
public interface GollekPlugin {
    void initialize(PluginContext context);
    void start();
    void stop();
    void shutdown();
    
    String id();
    String version();
    int order(); // Loading order
}
```

## 🔒 Plugin Isolation

Each plugin JAR is loaded with its own `PluginClassLoader`:

```
System ClassLoader
    └── Plugin ClassLoader (isolated)
        └── Plugin classes
        └── Plugin dependencies
```

**Benefits**:
- No classpath conflicts
- Independent dependency versions
- Safe unloading

## 📊 Plugin Directory Structure

```
~/.gollek/
└── plugins/
    ├── openai-provider.jar
    ├── openai-provider.jar.bak    # Backup before hot-reload
    ├── anthropic-provider.jar
    └── plugin.json                # Optional global config
```

## ⚙️ Configuration

### System Properties

| Property | Default | Description |
|----------|---------|-------------|
| `gollek.plugin.directory` | `~/.gollek/plugins` | Plugin directory |
| `gollek.plugin.hotreload` | `true` | Enable hot-reload |
| `gollek.plugin.watch.interval` | `1000` | Watch interval (ms) |

### Example

```bash
java -Dgollek.plugin.directory=/opt/gollek/plugins \
     -Dgollek.plugin.hotreload=true \
     -jar gollek-runtime.jar
```

## 🎯 Use Cases

### 1. Add New Cloud Provider

```bash
# Download plugin JAR
wget https://example.com/mistral-provider.jar

# Copy to plugin directory
cp mistral-provider.jar ~/.gollek/plugins/

# Restart or wait for hot-reload
```

### 2. Update Existing Provider

```bash
# New version will replace old on hot-reload
cp mistral-provider-2.0.0.jar ~/.gollek/plugins/mistral-provider.jar
```

### 3. Remove Provider

```bash
# Remove JAR - plugin will be unloaded
rm ~/.gollek/plugins/mistral-provider.jar
```

### 4. Development Workflow

```bash
# Watch directory during development
mvn clean package && cp target/*.jar ~/.gollek/plugins/

# Plugin auto-reloads on file change
```

## 📝 Plugin Descriptor Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique plugin identifier |
| `name` | ✅ | Human-readable name |
| `version` | ✅ | SemVer version |
| `description` | ❌ | Plugin description |
| `provider` | ❌ | Author/provider |
| `mainClass` | ✅ | Main plugin class |
| `dependencies` | ❌ | Maven dependencies |
| `capabilities` | ❌ | Feature tags |
| `config` | ❌ | Configuration schema |

## 🔍 Plugin Discovery

Plugins are discovered via:

1. **JAR Scanning** - All `.jar` files in plugin directory
2. **ServiceLoader** - `META-INF/services/tech.kayys.gollek.spi.plugin.GollekPlugin`
3. **plugin.json** - Manifest file inside JAR
4. **CDI** - For embedded plugins

## 🚀 Best Practices

### 1. Version Your Plugins

```
openai-provider-1.0.0.jar  # Good
openai-provider.jar        # Avoid (no version)
```

### 2. Use Capabilities

```java
@Capabilities({"cloud-provider", "streaming", "vision"})
```

### 3. Handle Configuration

```java
@ConfigProperty(name = "api.key", required = true)
private String apiKey;
```

### 4. Implement Health Checks

```java
@Override
public PluginHealth health() {
    return PluginHealth.healthy()
        .withDetail("apiStatus", "connected");
}
```

### 5. Log Appropriately

```java
private static final Logger LOG = Logger.getLogger(MyPlugin.class);

@Override
public void initialize(PluginContext context) {
    LOG.infof("Initializing %s v%s", id(), version());
}
```

## 📚 API Reference

### JarPluginLoader

```java
// Load all plugins from directory
List<GollekPlugin> loadAll()

// Load specific JAR
GollekPlugin loadFromJar(Path jarPath)

// Unload plugin
boolean unload(String pluginId)

// Start/stop watching
void startWatching()
void stopWatching()

// Get loaded plugin IDs
Set<String> getLoadedPluginIds()
```

### PluginManager

```java
// Initialize all plugins
void initialize()

// Start all plugins
void start()

// Stop all plugins
void stop()

// Get all plugins
Collection<GollekPlugin> all()

// Get plugin by ID
Optional<GollekPlugin> byId(String id)

// Get plugins by capability
List<GollekPlugin> byCapability(String capability)

// Register plugin dynamically
void registerPlugin(GollekPlugin plugin)

// Unregister plugin
void unregisterPlugin(String id)
```

## 🐛 Troubleshooting

### Plugin Not Loading

**Check**:
1. JAR is in correct directory
2. Main class exists and implements `GollekPlugin`
3. No `ClassNotFoundException` in logs
4. Dependencies are available

### Hot-Reload Not Working

**Check**:
1. File watcher started (check logs)
2. JAR file not locked by another process
3. File permissions allow reading

### ClassLoader Issues

**Symptoms**: `ClassCastException` between plugin and core

**Solution**: Ensure plugin uses SPI interfaces, not implementations

## 📖 Examples

See example plugins in:
- `inference-gollek/plugins/gollek-plugin-openai/`
- `inference-gollek/plugins/gollek-plugin-anthropic/`

---

**Version**: 2.1.0  
**Status**: ✅ Implementation Complete  
**Location**: `inference-gollek/core/gollek-plugin-core/`
