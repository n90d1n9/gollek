# Optimization Core Module Migration

**Date**: 2026-03-23
**Status**: ✅ Complete

---

## Migration Summary

Successfully moved the optimization plugin core module from:
- **Old**: `inference-gollek/plugins/gollek-plugin-optimization/gollek-plugin-optimization-core/`
- **New**: `inference-gollek/core/gollek-plugin-optimization-core/`

### Rationale

The optimization core module is better positioned in the `core/` directory because:

1. **Core Infrastructure**: It provides fundamental SPI interfaces used by all optimization plugins
2. **Shared Dependency**: Both built-in and external optimization plugins depend on it
3. **Architectural Clarity**: Separates core SPI from plugin implementations
4. **Consistency**: Aligns with other core modules like `gollek-plugin-core`

---

## Files Created/Moved

### Core Module Structure

```
inference-gollek/core/gollek-plugin-optimization-core/
├── pom.xml                                    # Maven POM (parent: gollek-core-parent)
├── README.md                                  # Complete documentation
└── src/main/java/tech/kayys/gollek/plugin/optimization/
    ├── OptimizationPlugin.java               # Plugin SPI interface
    ├── OptimizationPluginManager.java        # Plugin lifecycle manager
    └── ExecutionContext.java                 # Execution context SPI
```

### Key Components

#### 1. OptimizationPlugin.java
Main interface for all optimization plugins:
- `id()`, `name()`, `description()` - Plugin metadata
- `isAvailable()` - Hardware compatibility check
- `priority()` - Execution priority
- `apply(context)` - Apply optimization
- `shutdown()` - Cleanup resources

#### 2. OptimizationPluginManager.java
Singleton manager for plugin lifecycle:
- Plugin registration/unregistration
- Hardware detection
- Priority-based execution
- Health monitoring
- Configuration management

#### 3. ExecutionContext.java
Context provided to plugins:
- Model parameters access
- Memory buffer access
- GPU stream handles
- Execution phase information

---

## POM Configuration

```xml
<parent>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-core-parent</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</parent>

<artifactId>gollek-plugin-optimization-core</artifactId>
<name>Gollek Core :: Optimization Plugin Core</name>
```

**Dependencies**:
- JNA 5.14.0+ (native library binding)
- JBoss Logging (logging)
- JUnit 5, Mockito (testing)

---

## Documentation Updates

### 1. Website Documentation
**File**: `website/gollek-ai.github.io/docs/optimization-plugins.md`

**Updated Installation Section**:
```bash
# Build optimization core
cd inference-gollek/core/gollek-plugin-optimization-core
mvn clean install

# Build specific optimization plugin
cd inference-gollek/plugins/gollek-plugin-fa3
mvn clean install -Pinstall-plugin
```

### 2. Root README
**File**: `README.md`

**Updated Build Commands**:
```bash
# Build optimization core
cd inference-gollek/core/gollek-plugin-optimization-core
mvn clean install

# Build optimization plugins
cd inference-gollek/plugins/gollek-plugin-fa3
mvn clean install -Pinstall-plugin
```

**Updated Documentation Links**:
```markdown
- [Optimization Plugin Guide](inference-gollek/core/gollek-plugin-optimization-core/README.md)
```

---

## Module Dependencies

### Depends On (Core → Dependencies)
- `net.java.dev.jna:jna` - Native library binding
- `org.jboss.logging:jboss-logging` - Logging

### Used By (Dependencies → Core)
- `gollek-plugin-fa3` - FlashAttention-3 plugin
- `gollek-plugin-fa4` - FlashAttention-4 plugin
- `gollek-plugin-paged-attention` - PagedAttention plugin
- All other optimization plugins

---

## Integration Pattern

### For Plugin Developers

```java
// Import core SPI
import tech.kayys.gollek.plugin.optimization.OptimizationPlugin;
import tech.kayys.gollek.plugin.optimization.ExecutionContext;

// Implement interface
public class MyPlugin implements OptimizationPlugin {
    @Override
    public String id() { return "my-plugin"; }
    
    @Override
    public boolean apply(ExecutionContext context) {
        // Apply optimization
        return true;
    }
}
```

### For Engine Integration

```java
// Get manager instance
OptimizationPluginManager manager = OptimizationPluginManager.getInstance();

// Register plugins
manager.register(new FlashAttention3Plugin());

// Initialize
manager.initialize(config);

// Apply during inference
List<String> applied = manager.applyOptimizations(context);
```

---

## Build Order

```
1. gollek-plugin-optimization-core (core module)
   ↓
2. gollek-plugin-fa3 (depends on core)
   ↓
3. gollek-plugin-fa4 (depends on core)
   ↓
4. Other optimization plugins (depend on core)
```

---

## Testing

### Unit Tests

```java
@Test
void testOptimizationPluginManager() {
    OptimizationPluginManager manager = OptimizationPluginManager.getInstance();
    
    // Test registration
    manager.register(new TestPlugin());
    
    // Test initialization
    manager.initialize(Map.of());
    
    // Test application
    ExecutionContext context = createMockContext();
    List<String> applied = manager.applyOptimizations(context);
    
    assertFalse(applied.isEmpty());
}
```

---

## Migration Checklist

- [x] Create directory structure in `core/`
- [x] Create POM with correct parent
- [x] Move/create SPI interfaces
- [x] Create OptimizationPluginManager
- [x] Create README documentation
- [x] Update parent POM (already listed)
- [x] Update website documentation
- [x] Update root README
- [x] Update cross-references

---

## Benefits of New Location

### Before (in plugins/)
- ❌ Confusing architecture (core SPI in plugins directory)
- ❌ Circular dependency potential
- ❌ Unclear dependency hierarchy

### After (in core/)
- ✅ Clear architectural separation
- ✅ Core SPI separate from plugin implementations
- ✅ Consistent with other core modules
- ✅ Easier to understand dependency graph

---

## Next Steps

### For Optimization Plugin Developers

1. **Update Imports**:
   ```java
   // Old (if it existed)
   import tech.kayys.gollek.plugin.optimization.*; 
   
   // New (same package, different location)
   import tech.kayys.gollek.plugin.optimization.*;
   ```

2. **Update POM Dependencies**:
   ```xml
   <dependency>
       <groupId>tech.kayys.gollek</groupId>
       <artifactId>gollek-plugin-optimization-core</artifactId>
       <version>1.0.0-SNAPSHOT</version>
   </dependency>
   ```

3. **Build Core First**:
   ```bash
   cd inference-gollek/core/gollek-plugin-optimization-core
   mvn clean install
   ```

### For Existing Code

No code changes required! The package structure remains the same:
- `tech.kayys.gollek.plugin.optimization.OptimizationPlugin`
- `tech.kayys.gollek.plugin.optimization.OptimizationPluginManager`
- `tech.kayys.gollek.plugin.optimization.ExecutionContext`

Only the physical location changed, not the package structure.

---

## Resources

- **Module Location**: `inference-gollek/core/gollek-plugin-optimization-core/`
- **Documentation**: `README.md` in module directory
- **Website Docs**: `/docs/optimization-plugins.md`
- **SPI Interfaces**: `src/main/java/tech/kayys/gollek/plugin/optimization/`

---

**Status**: ✅ Migration Complete
**Impact**: No breaking changes (package structure unchanged)
**Action Required**: Update build paths in scripts/documentation
