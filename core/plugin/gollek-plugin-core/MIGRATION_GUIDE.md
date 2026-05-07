# Cloud Provider Migration Guide to Plugin System

## 🎯 Overview

Migrate existing cloud provider modules from static Maven dependencies to dynamic plugin system with automatic dependency resolution.

---

## 📋 Migration Checklist

- [ ] Update main class to implement `GollekPlugin`
- [ ] Create `plugin.json` descriptor
- [ ] Update POM for plugin packaging
- [ ] Move configuration to plugin.json
- [ ] Build and test plugin JAR
- [ ] Deploy to `~/.gollek/plugins/`
- [ ] Verify auto-loading

---

## 🔄 Migration Steps

### Step 1: Update Main Class

**Before:**
```java
package tech.kayys.gollek.provider.openai;

import tech.kayys.gollek.spi.provider.LLMProvider;

public class OpenAiProvider implements LLMProvider {
    // Existing implementation
}
```

**After:**
```java
package tech.kayys.gollek.plugin.cloud.openai;

import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.provider.LLMProvider;

public class OpenAiCloudProvider implements GollekPlugin, LLMProvider {
    
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
        // Load configuration from context
        Map<String, Object> config = context.configuration();
        String apiKey = (String) config.get("apiKey");
        // Initialize provider
    }
    
    @Override
    public void start() {
        // Start provider
    }
    
    @Override
    public void stop() {
        // Stop provider
    }
    
    @Override
    public void shutdown() {
        // Cleanup
    }
    
    // Existing LLMProvider methods remain unchanged
}
```

### Step 2: Create plugin.json

**Location**: `src/main/resources/plugin.json`

```json
{
  "id": "openai-cloud-provider",
  "name": "OpenAI Cloud Provider",
  "version": "1.0.0",
  "description": "Cloud provider for OpenAI models (GPT-4, GPT-3.5-turbo)",
  "provider": "Kayys.tech",
  "mainClass": "tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProvider",
  
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  
  "capabilities": ["cloud-provider", "streaming", "function-calling"],
  
  "config": {
    "apiKey": {
      "type": "string",
      "required": true,
      "description": "OpenAI API key",
      "sensitive": true
    },
    "baseUrl": {
      "type": "string",
      "required": false,
      "default": "https://api.openai.com/v1"
    },
    "organization": {
      "type": "string",
      "required": false,
      "description": "OpenAI organization ID"
    }
  },
  
  "models": [
    {
      "id": "gpt-4",
      "name": "GPT-4",
      "contextLength": 8192
    },
    {
      "id": "gpt-4-turbo",
      "name": "GPT-4 Turbo",
      "contextLength": 128000
    },
    {
      "id": "gpt-3.5-turbo",
      "name": "GPT-3.5 Turbo",
      "contextLength": 4096
    }
  ]
}
```

### Step 3: Update POM

**Location**: `pom.xml`

**Changes:**

```xml
<project>
    <!-- Update artifact ID -->
    <artifactId>gollek-plugin-cloud-openai</artifactId>
    <name>Gollek Plugin :: OpenAI Cloud Provider</name>
    
    <properties>
        <!-- Mark as plugin -->
        <plugin.type>cloud-provider</plugin.type>
    </properties>
    
    <dependencies>
        <!-- Change scope to 'provided' for Gollek APIs -->
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-provider</artifactId>
            <version>${project.version}</version>
            <scope>provided</scope>
        </dependency>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-spi-inference</artifactId>
            <version>${project.version}</version>
            <scope>provided</scope>
        </dependency>
        
        <!-- Keep runtime dependencies -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-vertx</artifactId>
            <scope>provided</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <!-- Configure JAR plugin for plugin packaging -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.4.2</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addDefaultImplementationEntries>true</addDefaultImplementationEntries>
                        </manifest>
                        <manifestEntries>
                            <Plugin-Id>openai-cloud-provider</Plugin-Id>
                            <Plugin-Version>${project.version}</Plugin-Version>
                            <Plugin-Class>tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProvider</Plugin-Class>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
            
            <!-- Auto-deploy to plugin directory -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-antrun-plugin</artifactId>
                <version>3.1.0</version>
                <executions>
                    <execution>
                        <id>deploy-plugin</id>
                        <phase>install</phase>
                        <goals>
                            <goal>run</goal>
                        </goals>
                        <configuration>
                            <target>
                                <mkdir dir="${user.home}/.gollek/plugins"/>
                                <copy file="${project.build.directory}/${project.build.finalName}.jar"
                                      todir="${user.home}/.gollek/plugins"/>
                                <echo message="Plugin deployed to ~/.gollek/plugins/"/>
                            </target>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

### Step 4: Update Configuration Loading

**Before:**
```java
@ConfigProperty(name = "gollek.providers.openai.api-key")
String apiKey;

@ConfigProperty(name = "gollek.providers.openai.base-url")
String baseUrl;
```

**After:**
```java
private Map<String, Object> config;

@Override
public void initialize(PluginContext context) {
    this.config = context.configuration();
    
    // Load from plugin config
    String apiKey = (String) config.get("apiKey");
    String baseUrl = (String) config.getOrDefault("baseUrl", "https://api.openai.com/v1");
}

@Override
public void onConfigUpdate(Map<String, Object> newConfig) {
    this.config.putAll(newConfig);
}
```

### Step 5: Build and Deploy

```bash
# Build plugin
cd inference-gollek/plugins/gollek-plugin-cloud-openai
mvn clean install

# Auto-deploys to ~/.gollek/plugins/

# Or manually copy
cp target/gollek-plugin-cloud-openai-1.0.0.jar ~/.gollek/plugins/
```

### Step 6: Verify Loading

```bash
# Check logs
tail -f ~/.gollek/logs/gollek.log | grep openai

# Should see:
# "Loading plugin from: ~/.gollek/plugins/gollek-plugin-cloud-openai-1.0.0.jar"
# "Resolving 2 Maven dependencies for plugin: openai-cloud-provider"
# "Successfully loaded plugin: openai-cloud-provider (version 1.0.0)"

# Check via REST API
curl http://localhost:8080/api/v1/plugins/openai-cloud-provider
```

---

## 📦 Provider-Specific Guides

### OpenAI Provider

**Module**: `gollek-ext-cloud-openai` → `gollek-plugin-cloud-openai`

**Main Class**: `OpenAiCloudProvider`

**plugin.json**:
```json
{
  "id": "openai-cloud-provider",
  "name": "OpenAI Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming", "function-calling", "vision"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true},
    "baseUrl": {"type": "string", "required": false, "default": "https://api.openai.com/v1"},
    "organization": {"type": "string", "required": false}
  }
}
```

### Anthropic Provider

**Module**: `gollek-ext-cloud-anthropic` → `gollek-plugin-cloud-anthropic`

**Main Class**: `AnthropicCloudProvider`

**plugin.json**:
```json
{
  "id": "anthropic-cloud-provider",
  "name": "Anthropic Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.anthropic.AnthropicCloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming", "function-calling", "vision"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true},
    "baseUrl": {"type": "string", "required": false, "default": "https://api.anthropic.com"}
  }
}
```

### Google Gemini Provider

**Module**: `gollek-ext-cloud-gemini` → `gollek-plugin-cloud-gemini`

**Main Class**: `GeminiCloudProvider`

**plugin.json**:
```json
{
  "id": "gemini-cloud-provider",
  "name": "Google Gemini Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.gemini.GeminiCloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming", "multimodal"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true}
  }
}
```

### Mistral Provider

**Module**: `gollek-ext-cloud-mistral` → `gollek-plugin-cloud-mistral`

**Main Class**: `MistralCloudProvider`

**plugin.json**:
```json
{
  "id": "mistral-cloud-provider",
  "name": "Mistral Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.mistral.MistralCloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming", "function-calling"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true}
  }
}
```

### Cerebras Provider

**Module**: `gollek-ext-cloud-cerebras` → `gollek-plugin-cloud-cerebras`

**Main Class**: `CerebrasCloudProvider`

**plugin.json**:
```json
{
  "id": "cerebras-cloud-provider",
  "name": "Cerebras Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.cerebras.CerebrasCloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming", "high-throughput"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true}
  }
}
```

---

## 🎯 Migration Script

Automated migration script:

```bash
#!/bin/bash
# migrate-cloud-providers.sh

PROVIDERS=("openai" "anthropic" "gemini" "mistral" "cerebras")

for provider in "${PROVIDERS[@]}"; do
    echo "Migrating $provider provider..."
    
    # Create plugin directory
    mkdir -p inference-gollek/plugins/gollek-plugin-cloud-$provider/src/main/java/tech/kayys/gollek/plugin/cloud/$provider
    
    # Copy existing implementation
    cp -r inference-gollek/extension/cloud/gollek-ext-cloud-$provider/src/main/java/tech/kayys/gollek/provider/$provider/* \
        inference-gollek/plugins/gollek-plugin-cloud-$provider/src/main/java/tech/kayys/gollek/plugin/cloud/$provider/
    
    # Create plugin.json
    cat > inference-gollek/plugins/gollek-plugin-cloud-$provider/src/main/resources/plugin.json << EOF
{
  "id": "$provider-cloud-provider",
  "name": "${provider^} Cloud Provider",
  "version": "1.0.0",
  "mainClass": "tech.kayys.gollek.plugin.cloud.$provider.${provider^}CloudProvider",
  "dependencies": [
    "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
    "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT"
  ],
  "capabilities": ["cloud-provider", "streaming"],
  "config": {
    "apiKey": {"type": "string", "required": true, "sensitive": true}
  }
}
EOF
    
    echo "✓ Migrated $provider provider"
done

echo "Migration complete! Build each plugin with: mvn clean install"
```

---

## ✅ Verification

After migration, verify each plugin:

```bash
# 1. Build all plugins
cd inference-gollek/plugins
for dir in gollek-plugin-cloud-*/; do
    cd $dir && mvn clean install && cd ..
done

# 2. Check plugin directory
ls -la ~/.gollek/plugins/*.jar

# 3. Start runtime
java -jar gollek-runtime.jar

# 4. Check logs
tail -f ~/.gollek/logs/gollek.log | grep -E "(openai|anthropic|gemini|mistral|cerebras)"

# 5. Verify via REST API
curl http://localhost:8080/api/v1/plugins | jq .

# Expected output:
# {
#   "plugins": [
#     {"id": "openai-cloud-provider", "status": "running"},
#     {"id": "anthropic-cloud-provider", "status": "running"},
#     {"id": "gemini-cloud-provider", "status": "running"},
#     {"id": "mistral-cloud-provider", "status": "running"},
#     {"id": "cerebras-cloud-provider", "status": "running"}
#   ]
# }
```

---

## 🐛 Troubleshooting

### Plugin Not Loading

**Check:**
1. JAR exists in `~/.gollek/plugins/`
2. `plugin.json` is in JAR (`jar tf plugin.jar | grep plugin.json`)
3. Main class exists and implements `GollekPlugin`
4. Dependencies are resolvable

### Dependency Resolution Fails

**Check:**
1. Maven coordinates are correct (`groupId:artifactId:version`)
2. Artifacts exist in Maven Central or configured repos
3. Network connectivity to repos
4. Local repo permissions (`~/.m2/repository`)

### ClassCastException

**Cause:** Same class loaded by different ClassLoaders

**Solution:**
- Ensure Gollek SPI modules are `provided` scope
- Don't bundle Gollek APIs in plugin JAR
- Use plugin ClassLoader isolation

### Configuration Not Loaded

**Check:**
1. Config in `plugin.json` matches expected keys
2. Configuration provided via `~/.gollek/plugins/plugin.json` or system properties
3. Plugin reads config in `initialize()` method

---

## 📊 Migration Timeline

| Phase | Providers | Duration |
|-------|-----------|----------|
| **Phase 1** | OpenAI | 1 day |
| **Phase 2** | Anthropic, Gemini | 2 days |
| **Phase 3** | Mistral, Cerebras | 2 days |
| **Phase 4** | Testing & docs | 2 days |
| **Total** | All 5 providers | **1 week** |

---

## 🎯 Post-Migration Benefits

| Benefit | Before | After |
|---------|--------|-------|
| **Deployment** | Recompile runtime | Drop JAR in plugins/ |
| **Updates** | Restart required | Hot-reload |
| **Dependencies** | Static in POM | Dynamic resolution |
| **Isolation** | Shared ClassLoader | Isolated per plugin |
| **Configuration** | Properties files | plugin.json + API |
| **Management** | Manual | REST API |

---

## 📚 Resources

- [Plugin System Guide](/docs/plugin-system)
- [Maven Dependency Resolver](/docs/maven-dependency-resolver)
- [Creating Plugins Tutorial](/docs/creating-plugins)
- [Cloud Providers Documentation](/docs/cloud-providers)

---

**Version**: 2.1.0  
**Last Updated**: 2026-03-22  
**Status**: ✅ Ready for Migration
