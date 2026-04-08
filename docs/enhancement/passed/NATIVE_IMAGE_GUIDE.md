# GraalVM Native Image Build Guide

## Overview

Gollek supports building native images using GraalVM for faster startup times and reduced memory footprint. This is particularly beneficial for serverless deployments and containerized environments.

## Prerequisites

### Option 1: Local GraalVM Installation

```bash
# Install GraalVM with Native Image
sdk install java 21.0.2-graalce
sdk use java 21.0.2-graalce

# Verify installation
java -version
# Should show: GraalVM CE ...

# Install native-image component
gu install native-image
```

### Option 2: Container Build (Recommended)

No local GraalVM installation required. Uses Docker/Podman for building.

```bash
# Ensure Docker is running
docker --version
```

## Building Native Image

### Standard Build (Container)

```bash
# From project root
cd inference-gollek/runtime/gollek-runtime-unified

# Build native image using container
mvn package -Pnative -Dquarkus.native.container-build=true
```

### Advanced Build Options

```bash
# Build with specific builder image
mvn package -Pnative \
  -Dquarkus.native.container-build=true \
  -Dquarkus.native.builder-image=quay.io/quarkus/ubi-quarkus-mandrel-builder-image:jdk-21

# Build with optimization for size
mvn package -Pnative \
  -Dquarkus.native.container-build=true \
  -Dquarkus.native.native-image-xmx=4g \
  -Dquarkus.native.enable-vm-inspection=true

# Build with all optimizations
mvn package -Pnative \
  -Dquarkus.native.container-build=true \
  -Dquarkus.native.enable-all-security-services=true \
  -Dquarkus.native.add-all-charsets=true \
  -Dquarkus.native.enable-all-timezones=true
```

### Local Build (Requires GraalVM)

```bash
# Set GRAALVM_HOME
export GRAALVM_HOME=/path/to/graalvm

# Build native image
mvn package -Pnative
```

## Configuration

### Native Image Feature

The `NativeImageFeature` class automatically registers classes for reflection:

- SPI interfaces and implementations
- Mutiny reactive types
- Jackson serialization classes
- Plugin system classes

### Reflection Configuration

Additional reflection configuration is in:
- `src/main/resources/META-INF/native-image/gollek/reflect-config.json`

### Resource Configuration

Resource file patterns are in:
- `src/main/resources/META-INF/native-image/gollek/resource-config.json`

## Testing Native Image

### Run the Native Executable

```bash
# Run the native executable
./target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner

# With configuration
./target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner \
  -Dquarkus.http.port=8080 \
  -Dquarkus.log.level=INFO
```

### Verify Native Image

```bash
# Check if binary is truly native
file target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner

# Should show: ELF 64-bit LSB executable (Linux) or Mach-O 64-bit (macOS)

# Check startup time
time ./target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner
```

## Performance Comparison

### JVM Mode

```bash
# Startup time: ~2-5 seconds
# Memory footprint: ~200-400 MB
java -jar target/quarkus-app/quarkus-run.jar
```

### Native Mode

```bash
# Startup time: ~50-200 ms (10-25x faster)
# Memory footprint: ~80-150 MB (50-60% reduction)
./target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner
```

## Troubleshooting

### Missing Class Registration

If you encounter errors about missing classes:

```java
// Add to NativeImageFeature.java
@RegisterForReflection(targets = {
    YourClass.class
})
```

Or add to `reflect-config.json`:

```json
{
  "name": "com.example.YourClass",
  "allDeclaredConstructors": true,
  "allPublicConstructors": true,
  "allDeclaredMethods": true,
  "allPublicMethods": true,
  "allDeclaredFields": true,
  "allPublicFields": true
}
```

### Resource Not Found

If resources are missing at runtime:

```json
// Add to resource-config.json
{
  "pattern": "path/to/resource/.*"
}
```

### Native Image Build Fails

Common solutions:

```bash
# Increase memory for native-image builder
mvn package -Pnative \
  -Dquarkus.native.native-image-xmx=8g

# Enable verbose output for debugging
mvn package -Pnative \
  -Dquarkus.native.verbose-output=true

# Generate report for analysis
mvn package -Pnative \
  -Dquarkus.native.enable-reports=true
```

### JNI Errors

For JNI-related issues:

```json
// Add to reflect-config.json in "jni" array
{
  "name": "com.example.NativeLibrary"
}
```

## Platform-Specific Notes

### Linux

```bash
# Install required packages
sudo apt-get install build-essential libz-dev zlib1g-dev
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# For Apple Silicon (M1/M2)
# Use Rosetta 2 for x86_64 native images
softwareupdate --install-rosetta
```

### Windows

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

## Docker Deployment

### Multi-Stage Dockerfile

```dockerfile
# Build stage
FROM quay.io/quarkus/ubi-quarkus-maven-builder:3.32 AS builder
WORKDIR /build
COPY . .
RUN mvn package -Pnative -DskipTests -Dquarkus.native.container-build=false

# Runtime stage
FROM gcr.io/distroless/base-debian12
WORKDIR /application
COPY --from=builder /build/runtime/gollek-runtime-unified/target/*-runner ./gollek
COPY --from=builder /build/runtime/gollek-runtime-unified/target/classes/application.properties ./config/

EXPOSE 8080
USER nonroot

ENTRYPOINT ["./gollek"]
```

### Docker Build

```bash
docker build -t gollek-native:latest -f Dockerfile.native .
```

## Benchmarking

### Startup Time

```bash
# Measure startup time
hyperfine --warmup 3 './target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner'
```

### Memory Usage

```bash
# Monitor memory usage
/usr/bin/time -v ./target/gollek-runtime-unified-1.0.0-SNAPSHOT-runner
```

### Throughput

```bash
# Run load tests
wrk -t12 -c400 -d30s http://localhost:8080/api/inference
```

## Known Limitations

1. **Dynamic Class Loading**: Some dynamic class loading features may not work
2. **Reflection**: All reflected classes must be registered
3. **Resources**: All resource files must be explicitly included
4. **Native Libraries**: JNI libraries must be compiled for target platform

## Best Practices

1. **Test Early**: Test native image generation early in development
2. **Minimal Dependencies**: Exclude unnecessary dependencies
3. **Configuration**: Use configuration files for reflection/resource registration
4. **CI/CD**: Automate native image builds in CI/CD pipeline
5. **Monitoring**: Monitor native image performance in production

## References

- [Quarkus Native Image Guide](https://quarkus.io/guides/building-native-image)
- [GraalVM Native Image](https://www.graalvm.org/latest/reference-manual/native-image/)
- [Mandrel Builder Image](https://quay.io/repository/quarkus/ubi-quarkus-mandrel-builder-image)

## Support

For issues related to native image builds:

1. Check the troubleshooting section above
2. Review GraalVM documentation
3. Enable verbose output: `-Dquarkus.native.verbose-output=true`
4. Generate build reports: `-Dquarkus.native.enable-reports=true`
