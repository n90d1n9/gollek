# Gollek SPI (Service Provider Interfaces)

Centralized location for all Gollek Service Provider Interfaces (SPIs).

## 📦 Module Structure

```
inference-gollek/spi/
├── pom.xml                       # Parent POM
├── gollek-spi-plugin/            # Plugin lifecycle and extension points
├── gollek-spi-model/             # Model loading and metadata
├── gollek-spi-inference/         # Inference execution contracts
└── gollek-spi-provider/          # Provider discovery and management
```

## 🎯 Purpose

Each SPI module defines a specific contract:

### gollek-spi-plugin
- Plugin lifecycle management
- Extension points
- Plugin registry
- Plugin health monitoring
- Plugin state management

### gollek-spi-model  
- Model loading interface
- Model metadata
- Model format abstraction
- Model repository contracts
- Model routing

### gollek-spi-inference
- Inference session management
- Input/output processing
- Execution contracts
- Result handling
- Device management

### gollek-spi-provider
- Provider discovery
- Provider configuration
- Backend selection
- Provider lifecycle
- Provider routing

## 🔧 Usage

Add dependency to your module:

```xml
<!-- Plugin SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-plugin</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Model SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-model</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Inference SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-inference</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Provider SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-provider</artifactId>
    <version>${project.version}</version>
</dependency>
```

## 🏗️ Architecture Principles

The SPI modules follow these principles:

1. **Interface-only**: Define contracts, not implementations
2. **Independent**: Each SPI can evolve independently  
3. **Minimal dependencies**: Only depend on what's necessary
4. **Stable contracts**: Backward-compatible changes only
5. **Well-documented**: Clear JavaDoc for all interfaces
6. **Centralized location**: All SPIs in one place for easy discovery

## 📝 Implementation Guide

To implement a SPI:

1. Add the SPI dependency to your module
2. Implement the interface(s)
3. Register implementation via `META-INF/services`
4. Use appropriate annotations

Example:

```java
// Implement Model SPI
public class MyModelLoader implements ModelLoader {
    @Override
    public Model load(Path path) {
        // Implementation
    }
}

// Register in META-INF/services
// META-INF/services/tech.kayys.gollek.spi.model.ModelLoader
tech.kayys.gollek.spi.model.MyModelLoader
```

## 📁 Location

**Old Structure** (deprecated):
```
inference-gollek/core/
├── gollek-spi-plugin/
├── gollek-spi-model/
├── gollek-spi-inference/
└── gollek-spi-provider/
```

**New Structure** (current):
```
inference-gollek/spi/
├── gollek-spi-plugin/
├── gollek-spi-model/
├── gollek-spi-inference/
└── gollek-spi-provider/
```

## 🔗 Related Modules

### Implementations (in `core/`)
- `gollek-plugin-core` - Plugin implementation
- `gollek-model-registry` - Model registry implementation
- `gollek-provider-core` - Provider implementation

### Runtime (in `runtime/`)
- `gollek-runtime` - Main runtime
- `gollek-engine` - Engine implementation

## 🚀 Benefits of Centralized SPI Location

1. **Easy Discovery**: All SPIs in one place
2. **Clear Separation**: SPIs separate from implementations
3. **Better Organization**: Logical grouping
4. **Simplified Dependencies**: Single parent for all SPIs
5. **Independent Evolution**: SPIs can change without affecting core

---

**Version**: 1.0.0-SNAPSHOT  
**Parent**: gollek-parent  
**Location**: `inference-gollek/spi/`
