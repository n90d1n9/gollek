# Gollek Unified Runtime

The **Gollek Unified Runtime** is a versatile distribution that packages the entire Gollek ecosystem into a single, high-performance executable. It serves as the primary entry point for developers and operators, offering a seamless transition between interactive experimentation and production-grade hosting.

## Key Features

- **Dual-Mode Execution**: Run as an interactive CLI or a high-performance REST/Web server.
- **Unified Distribution**: Combines CLI, REST API, Web UI, and Inference Engine in one JAR.
- **Native-First Design**: Optimized for GraalVM native image compilation.
- **Enterprise-Ready**: Includes built-in health checks, metrics, and tracing support.
- **Multi-Backend**: Supports GGUF (llama.cpp) and SafeTensors (LibTorch) inference out of the box.

## Architecture

The unified runtime acts as an orchestrator that wires together several core modules:
- `gollek-cli`: Command-line interface for model management and chat.
- `gollek-runtime-standalone`: The core server implementation.
- `gollek-sdk`: The Java API for interacting with the engine.
- `tenant-core`: Multi-tenant management and isolation.

## Getting Started

### Prerequisites

- **JDK 25**: The runtime leverages the latest Java features (Panama, Valhalla).
- **Native Libraries**: Ensure `llama.cpp` and `LibTorch` are installed in `~/.gollek/libs/`.

### Building

```bash
mvn clean package
```

This will produce an executable "runner" JAR in the `target/` directory.

### Running

#### Server Mode (Default)
Starts the REST API, Web UI, and background inference services.
```bash
java -jar target/gollek-runtime-unified-0.1.0-runner.jar
```

#### CLI Mode
Triggered by providing any command (like `chat`, `pull`, `list`).
```bash
java -jar target/gollek-runtime-unified-0.1.0-runner.jar chat --model llama3
```

## Configuration

Configuration is managed via `application.properties`, environment variables, or CLI system properties.

| Property | Description | Default |
| :--- | :--- | :--- |
| `quarkus.http.port` | Port for the REST/Web server | `8080` |
| `gollek.distribution.mode` | Distribution profile (unified, demo, standalone) | `unified` |
| `GOLLEK_METRICS_ENABLED` | Enable Prometheus metrics | `true` |
| `GOLLEK_TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` |

## Observability

- **Status API**: `GET /api/status` - Returns version and health summary.
- **Health Checks**:
  - Liveness: `/health/live`
  - Readiness: `/health/ready`
- **Metrics**: `/metrics` (Prometheus format)
- **API Docs**: `/swagger-ui` (Swagger) or `/openapi` (Raw JSON)

## Development

The project is structured to support rapid iteration:
- `src/main/java`: Core orchestration and bootstrapping logic.
- `src/main/resources`: Web assets and configuration defaults.

---
© 2026 Kayys.tech. All rights reserved.
