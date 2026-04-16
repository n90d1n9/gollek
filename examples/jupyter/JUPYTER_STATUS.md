# Jupyter Notebook - Known Issue

## Current Status: ❌ Not Working

The Jupyter kernel for Gollek is experiencing startup failures due to Maven dependency resolution issues.

### Error
```
Kernel does not exist: <kernel-id>
```

### Root Cause
The kernel attempts to load 23+ Maven dependencies at startup, which causes:
- Dependency resolution timeout
- Classpath conflicts
- Kernel process crash

## Working Alternatives

### 1. CLI (Recommended)
```bash
java -jar gollek/ui/gollek-cli/target/gollek-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat" \
  --steps 10 \
  --output cat.png
```

### 2. Shell Script (Easiest)
```bash
cd gollek/examples/jbang/multimodal
./sd_generate.sh "a cat" 10 cat.png
```

### 3. JBang (For non-Quarkus examples)
```bash
cd gollek/examples/jbang/sdk
jbang gollek-quickstart.java
```

## Potential Fix (For Future)

To make Jupyter work, we need to:
1. Create an **uber-JAR** with all Gollek dependencies
2. Or use a **minimal dependency set** (just SDK core)
3. Or switch to a **different Java kernel** (e.g., IJava with classpath config)

### Option 1: Create Uber-JAR
```bash
mvn package -pl gollek/sdk/gollek-sdk-java-local -Puber-jar
# Then configure kernel to use the single JAR
```

### Option 2: Minimal Kernel
```json
{
  "env": {
    "JJAVA_DEPS": "tech.kayys.gollek:gollek-sdk-core:0.1.0-SNAPSHOT"
  }
}
```

### Option 3: Use IJava Kernel
```bash
# Install IJava
jbang app install java-kernel@ivanyu/java-kernel

# Configure with classpath
java -cp "gollek-*.jar" org.dflib.jjava.JJava {connection_file}
```

## Documentation

The notebook (`07-stable-diffusion.ipynb`) is still valuable as:
- **Reference documentation** for the API
- **Template** for when Jupyter is fixed
- **Learning material** with markdown explanations

---

**Last Updated**: April 14, 2026  
**Status**: Blocked on kernel dependency issue  
**Workaround**: Use CLI or shell script
