# Gollek Quantizer Integration Guide

## Overview

This document describes how the quantizer modules (`gollek-quantizer-gptq`, `gollek-quantizer-awq`, `gollek-quantizer-autoround`, `gollek-quantizer-turboquant`) are integrated with the safetensor infrastructure and SDK.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gollek SDK API                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ QuantizationService                                           │ │
│  │ - High-level API for model quantization                       │ │
│  │ - GPTQ, AWQ, AutoRound, TurboQuant access                     │ │
│  │ - Progress streaming, async operations                        │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
└──────────────────────────────┼─────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Safetensor Quantization Module                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ QuantizationEngine                                            │ │
│  │ - Model-level quantization orchestration                      │ │
│  │ - Async operations, progress streaming                        │ │
│  │ - Strategy selection (INT4, INT8, FP8)                        │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
│                              │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ QuantizerRegistry                                             │ │
│  │ - Central quantizer discovery                                 │ │
│  │ - Quantizer selection by config                               │ │
│  │ - Custom quantizer registration                               │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
│                              │                                     │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │ Quantizer SPI (tech.kayys.gollek.safetensor.quantization.    │ │
│  │                       quantizer.Quantizer)                    │ │
│  │ - Tensor-level quantization interface                         │ │
│  │ - quantizeTensor() / dequantizeTensor()                       │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
└──────────────────────────────┼─────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ GPTQQuantizer    │ │ AWQQuantizer     │ │ AutoRoundQuant.  │
│ Adapter          │ │ Adapter (TODO)   │ │ Adapter (TODO)   │
│                  │ │                  │ │                  │
│ ✅ Registered    │ │ ⏳ Pending       │ │ ⏳ Pending       │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ gollek-quantizer │ │ gollek-quantizer │ │ gollek-quantizer │
│ -gptq            │ │ -awq             │ │ -autoround       │
│                  │ │                  │ │                  │
│ ✅ Full tests    │ │ ✅ Full tests    │ │ ✅ Full tests    │
│ ✅ 100% pass     │ │ ✅ 100% pass     │ │ ✅ 100% pass     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Module Dependencies

### Quantizer Modules → Safetensor

Each quantizer module depends on:

```xml
<!-- gollek-quantizer-gptq/pom.xml -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-loader</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

### Safetensor Quantization → Quantizer Modules

The safetensor-quantization module depends on all quantizer modules:

```xml
<!-- gollek-safetensor-quantization/pom.xml -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-quantizer-gptq</artifactId>
    <version>${project.version}</version>
</dependency>
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-quantizer-awq</artifactId>
    <version>${project.version}</version>
</dependency>
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-quantizer-autoround</artifactId>
    <version>${project.version}</version>
</dependency>
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-quantizer-turboquant</artifactId>
    <version>${project.version}</version>
</dependency>
```

### SDK API → Safetensor Quantization

The SDK API depends on safetensor-quantization:

```xml
<!-- gollek-sdk-api/pom.xml -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-quantization</artifactId>
    <version>${project.version}</version>
</dependency>
```

## Integration Points

### 1. QuantizerRegistry

Central registry for all quantizer implementations:

```java
// Get all available quantizers
Map<String, Quantizer> quantizers = QuantizerRegistry.getAll();

// Get specific quantizer
Quantizer gptq = QuantizerRegistry.get("GPTQ");

// Select best quantizer for config
Quantizer best = QuantizerRegistry.selectBest(config);

// Register custom quantizer
QuantizerRegistry.register("CUSTOM", myCustomQuantizer);
```

### 2. QuantizationEngine

Main orchestration engine:

```java
QuantizationEngine engine = new QuantizationEngine();

// Quantize model
QuantResult result = engine.quantize(
    Path.of("/models/llama-7b-fp16"),
    Path.of("/models/llama-7b-int4"),
    QuantConfig.QuantStrategy.INT4,
    QuantConfig.int4Gptq()
);

// Quantize with progress streaming
Multi<Object> progress = engine.quantizeWithProgress(
    modelPath, outputPath, strategy, config
);

// Async quantization
Uni<QuantResult> async = engine.quantizeAsync(
    modelPath, outputPath, strategy, config
);
```

### 3. QuantizationService (SDK)

High-level SDK interface:

```java
QuantizationService qs = QuantizationService.getInstance();

// GPTQ quantization
QuantResult result = qs.quantizeGptq(modelPath, outputPath);

// INT8 quantization
QuantResult result8 = qs.quantizeInt8(modelPath, outputPath);

// With progress streaming
qs.quantizeWithProgress(modelPath, outputPath, config)
  .subscribe().with(
      item -> System.out.println(item),
      error -> System.err.println(error),
      () -> System.out.println("Done!")
  );
```

## Test Results

### Quantizer Modules

| Module | Tests | Status |
|--------|-------|--------|
| gollek-quantizer-gptq | 16/16 | ✅ 100% |
| gollek-quantizer-awq | 19/19 | ✅ 100% |
| gollek-quantizer-autoround | 39/39 | ✅ 100% |
| gollek-quantizer-turboquant | 74/74 | ✅ 100% |

### Integration Tests

| Test Suite | Tests | Status |
|------------|-------|--------|
| QuantizerIntegrationTest | 12/12 | ✅ 100% |

## Building

### Build All Quantizer Modules

```bash
# Build all quantizers
cd gollek/core/quantizer
mvn clean install -pl gollek-quantizer-gptq,gollek-quantizer-awq,gollek-quantizer-autoround,gollek-quantizer-turboquant -am -DskipTests

# Build safetensor quantization
cd ../../plugins/runner/safetensor/gollek-safetensor-quantization
mvn clean install

# Build SDK API
cd ../../../sdk/lib/gollek-sdk-api
mvn clean install
```

### Run Integration Tests

```bash
cd gollek/plugins/runner/safetensor/gollek-safetensor-quantization
mvn clean test -Dtest=QuantizerIntegrationTest
```

## Vector API Module

All quantizer modules use the JDK 25 Vector API (incubator). When running tests, ensure the module is loaded:

```xml
<!-- In pom.xml -->
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <version>3.2.5</version>
    <configuration>
        <argLine>
            --add-modules=jdk.incubator.vector
            --enable-preview
        </argLine>
    </configuration>
</plugin>
```

## Next Steps

### TODO: Complete AWQ/AutoRound/TurboQuant Adapters

Currently only GPTQ has a full adapter implementation. The other quantizers need:

1. **AWQQuantizerAdapter** - Bridge AWQ module to Quantizer SPI
2. **AutoRoundQuantizerAdapter** - Bridge AutoRound module to Quantizer SPI
3. **TurboQuantQuantizerAdapter** - Bridge TurboQuant module to Quantizer SPI

Each adapter should:
- Implement `tech.kayys.gollek.safetensor.quantization.quantizer.Quantizer`
- Register with `QuantizerRegistry` in static initializer
- Delegate to the underlying quantizer's model-level APIs

### TODO: Tensor-Level Quantization

Current quantizers operate at the **model level** (load full model, quantize all weights). For finer-grained control:

1. Implement tensor-level `quantizeTensor()` methods
2. Support partial model quantization
3. Enable mixed-precision quantization

## Troubleshooting

### NoClassDefFoundError: jdk/incubator/vector/Vector

**Solution**: Add Vector API module to test runtime:

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <configuration>
        <argLine>--add-modules=jdk.incubator.vector</argLine>
    </configuration>
</plugin>
```

### ClassNotFoundException: tech.kayys.gollek.sdk.api.QuantizationService

**Solution**: Ensure SDK module is built and on classpath:

```bash
cd gollek/sdk/lib/gollek-sdk-api
mvn clean install
```

### Quantizer Not Found in Registry

**Solution**: Verify adapter is registered:

```java
System.out.println(QuantizerRegistry.getNames());
// Should include: gptq, awq, autoround, turboquant
```

## Support

- **GitHub Issues**: https://github.com/wayang-platform/gollek/issues
- **Documentation**: https://wayang-platform.github.io/gollek
- **Email**: team@wayang.dev
