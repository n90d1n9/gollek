# Gollek Inference Engine Enhancement Plan - Implementation Roadmap

**Document Version:** 1.0  
**Date:** March 27, 2026  
**Author:** Gollek Inference Expert Agent  
**Status:** In Progress

---

## Executive Summary

This document outlines the comprehensive implementation roadmap for enhancing the Gollek inference platform based on the enhancement plan (plan-20260327.md). The roadmap is structured into three phases: **Short Term** (Stability & Security), **Mid Term** (Performance & Features), and **Long Term** (Scale & Ecosystem).

### Current State Assessment

**Strengths:**
- ✅ Clean SPI architecture with modular design
- ✅ Quarkus-based modular monolith
- ✅ Existing plugin system with phase-bound execution
- ✅ Basic observability (OpenTelemetry, Micrometer)
- ✅ Circuit breaker and reliability mechanisms
- ✅ Multi-format model support (GGUF, SafeTensors, ONNX)
- ✅ 147+ test files with good coverage in some areas

**Gaps to Address:**
- ❌ Vault secrets integration commented out
- ❌ Incomplete GraalVM Native Image configuration
- ❌ No PII redaction/security plugin
- ❌ No semantic caching
- ❌ Limited integration test suites
- ❌ Missing distributed tracing propagation
- ❌ No event-driven audit logging

---

## Phase 1: Short Term (Weeks 1-4)
**Focus:** Stability & Security

### Milestone 1.1: Security Enhancements (Week 1-2)

#### 1.1.1 Vault Secrets Integration
**Priority:** CRITICAL  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/ProviderConfigLoader.java`
- `core/gollek-provider-core/pom.xml`
- `runtime/gollek-runtime-unified/src/main/resources/application.properties`

**Tasks:**
- [ ] Add Quarkus Vault extension dependency
- [ ] Uncomment and implement VaultSecretManager integration
- [ ] Create Vault configuration classes
- [ ] Add integration tests with Testcontainers Vault
- [ ] Update documentation

**Acceptance Criteria:**
- Vault secrets are loaded for provider configurations
- Fallback to environment variables when Vault is unavailable
- Proper error handling and logging
- Integration tests pass with Vault container

#### 1.1.2 PII Redaction Plugin
**Priority:** CRITICAL  
**Status:** 🔴 NOT STARTED  
**Files to Create:**
- `plugins/common/gollek-plugin-pii-redaction/` (new module)
- `core/spi/gollek-spi-inference/src/main/java/tech/kayys/gollek/spi/inference/SecurityPhasePlugin.java` (new SPI)
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/security/PIIRedactionService.java`

**Tasks:**
- [ ] Create new plugin module structure
- [ ] Implement PII detection using regex patterns (email, phone, SSN, credit cards)
- [ ] Integrate with inference pipeline (PRE_PROCESSING phase)
- [ ] Add configuration for custom PII patterns
- [ ] Create unit and integration tests
- [ ] Add audit logging for redaction events

**Acceptance Criteria:**
- PII is detected and redacted before inference
- Configurable redaction patterns
- Audit trail of redaction events
- Performance impact < 5ms per request

#### 1.1.3 RBAC Enhancement
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/spi/gollek-spi/src/main/java/tech/kayys/gollek/spi/context/RequestContext.java`
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/security/EngineQuotaEnforcer.java`

**Tasks:**
- [ ] Add UserContext with roles to RequestContext
- [ ] Integrate Quarkus OIDC extension
- [ ] Implement role-based quota enforcement
- [ ] Add RBAC checks to API endpoints

**Acceptance Criteria:**
- User roles available in request context
- OIDC authentication working
- Role-based access control enforced

### Milestone 1.2: GraalVM Native Image (Week 2-3)

#### 1.2.1 Finalize Native Image Configuration
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/NativeImageFeature.java`
- `runtime/gollek-runtime-unified/pom.xml`
- `runtime/gollek-runtime-standalone/pom.xml`

**Tasks:**
- [ ] Add GraalVM native-image dependencies
- [ ] Implement proper Feature interface
- [ ] Register all SPI classes for reflection
- [ ] Register Mutiny producers and subscribers
- [ ] Add resource bundle registration
- [ ] Create native image build profiles
- [ ] Test native image compilation
- [ ] Document native image limitations and workarounds

**Acceptance Criteria:**
- Native image compiles successfully
- All SPI plugins load correctly in native mode
- Startup time < 100ms
- Memory footprint reduced by 40%

### Milestone 1.3: Test Coverage Improvement (Week 3-4)

#### 1.3.1 Integration Test Suites
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Create:**
- `core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/InferencePipelineIntegrationTest.java`
- `core/gollek-provider-core/src/test/java/tech/kayys/gollek/provider/core/ProviderConfigLoaderIntegrationTest.java`
- `plugins/common/gollek-plugin-pii-redaction/src/test/java/tech/kayys/gollek/plugin/security/PIIRedactionPluginIT.java`

**Tasks:**
- [ ] Create @QuarkusTest integration test base class
- [ ] Add Testcontainers for external dependencies
- [ ] Create end-to-end inference tests
- [ ] Add provider integration tests (mock providers)
- [ ] Implement contract tests for SPI interfaces
- [ ] Add performance benchmark tests

**Acceptance Criteria:**
- Code coverage increased to >80%
- All critical paths have integration tests
- Performance benchmarks established
- CI/CD pipeline includes integration tests

#### 1.3.2 Contract Testing
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Create:**
- `core/spi/gollek-spi-inference/src/test/java/tech/kayys/gollek/spi/inference/InferencePhasePluginContractTest.java`
- `core/spi/gollek-spi-provider/src/test/java/tech/kayys/gollek/spi/provider/ProviderPluginContractTest.java`

**Tasks:**
- [ ] Define contract test base classes for each SPI
- [ ] Implement contract tests for InferencePhasePlugin
- [ ] Implement contract tests for ProviderPlugin
- [ ] Add contract tests for RunnerPlugin
- [ ] Document contract testing guidelines

**Acceptance Criteria:**
- All plugin types have contract tests
- New plugins must pass contract tests
- Contract tests run in CI/CD pipeline

---

## Phase 2: Mid Term (Weeks 5-10)
**Focus:** Performance & Features

### Milestone 2.1: Semantic Caching (Week 5-6)

#### 2.1.1 Semantic Cache Plugin
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Create:**
- `plugins/optimization/gollek-plugin-semantic-cache/` (new module)
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/cache/SemanticCacheService.java`

**Tasks:**
- [ ] Create semantic cache plugin module
- [ ] Implement embedding-based similarity check
- [ ] Integrate with PRE_PROCESSING phase
- [ ] Add cache invalidation strategies
- [ ] Configure TTL and size limits
- [ ] Add metrics for cache hit/miss rates

**Acceptance Criteria:**
- Semantically similar prompts return cached responses
- Cache hit rate > 30% for typical workloads
- Latency overhead < 10ms
- Configurable similarity threshold

### Milestone 2.2: KV Cache Optimization (Week 6-8)

#### 2.2.1 PagedAttention Enhancement
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `plugins/optimization/gollek-plugin-paged-attention/`
- `plugins/optimization/gollek-plugin-kv-cache/`

**Tasks:**
- [ ] Enhance PagedAttention with vLLM-style memory management
- [ ] Implement KV cache offloading to CPU/NVMe
- [ ] Add dynamic batch size adjustment
- [ ] Optimize memory fragmentation

**Acceptance Criteria:**
- 2x throughput improvement for long sequences
- Memory utilization > 90%
- No OOM errors under load

### Milestone 2.3: RAG Integration (Week 7-9)

#### 2.3.1 RAG Plugin Enhancement
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `plugins/common/gollek-plugin-rag/`

**Tasks:**
- [ ] Add support for multiple vector databases (PGVector, Milvus, Qdrant)
- [ ] Implement hybrid search (dense + sparse)
- [ ] Add re-ranking capabilities
- [ ] Create RAG evaluation metrics

**Acceptance Criteria:**
- Multi-vector-store support
- Sub-100ms retrieval latency
- Configurable top-k and re-ranking

### Milestone 2.4: Resilience4j Integration (Week 8-10)

#### 2.4.1 Replace Custom Circuit Breaker
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/model/ReliabilityManager.java`
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/registry/CircuitBreakerRegistry.java`

**Tasks:**
- [ ] Add Resilience4j Quarkus extension
- [ ] Migrate from custom circuit breaker to Resilience4j
- [ ] Implement bulkhead isolation per tenant
- [ ] Add retry with exponential backoff
- [ ] Configure rate limiting

**Acceptance Criteria:**
- All reliability patterns use Resilience4j
- Per-tenant bulkhead isolation
- Standardized metrics and dashboards

### Milestone 2.5: Multi-Modal Support (Week 9-10)

#### 2.5.1 Multi-Modal Pipeline Enhancement
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/spi/gollek-spi-inference/src/main/java/tech/kayys/gollek/spi/inference/InferenceRequest.java`
- `core/gollek-multimodal-core/`

**Tasks:**
- [ ] Extend InferenceRequest with Attachment support
- [ ] Add image/audio/video processing
- [ ] Update ProviderCapabilities for multimodal
- [ ] Create multimodal test datasets

**Acceptance Criteria:**
- Image input support
- Audio transcription support
- Unified API for multimodal requests

---

## Phase 3: Long Term (Weeks 11-16)
**Focus:** Scale & Ecosystem

### Milestone 3.1: Event-Driven Audit Logging (Week 11-12)

#### 3.1.1 Kafka-Based Audit Events
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/audit/AuditLoggingPlugin.java`

**Tasks:**
- [ ] Add SmallRye Reactive Messaging with Kafka
- [ ] Refactor audit logging to emit events
- [ ] Create audit event consumer service
- [ ] Implement event sourcing for audit trail
- [ ] Add audit log retention policies

**Acceptance Criteria:**
- Audit events published to Kafka
- Decoupled from inference path (<1ms overhead)
- Queryable audit log with retention

### Milestone 3.2: Distributed Tracing Enhancement (Week 12-13)

#### 3.2.1 W3C Trace Context Propagation
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/spi/gollek-spi-provider/src/main/java/tech/kayys/gollek/spi/provider/ProviderRequest.java`
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/observability/`

**Tasks:**
- [ ] Add trace context to ProviderRequest metadata
- [ ] Propagate W3C headers to external providers
- [ ] Create distributed tracing dashboards
- [ ] Add trace-based sampling for debugging

**Acceptance Criteria:**
- End-to-end tracing across services
- Trace context visible in Jaeger/Tempo
- Sampling configurable by trace ID

### Milestone 3.3: CLI and SDKs (Week 13-15)

#### 3.3.1 Gollek CLI
**Priority:** MEDIUM  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `ui/gollek-cli/`

**Tasks:**
- [ ] Add Picocli commands for model management
- [ ] Implement model upload/download
- [ ] Add inference testing commands
- [ ] Create plugin management commands
- [ ] Build native CLI binary

**Acceptance Criteria:**
- CLI commands for all major operations
- Native binary for Linux/Mac/Windows
- Comprehensive help documentation

#### 3.3.2 Python/Node.js SDKs
**Priority:** LOW  
**Status:** 🔴 NOT STARTED  
**Files to Create:**
- `sdk/gollek-python-sdk/`
- `sdk/gollek-nodejs-sdk/`

**Tasks:**
- [ ] Generate SDKs from OpenAPI spec
- [ ] Add async support
- [ ] Implement streaming clients
- [ ] Create example applications
- [ ] Publish to PyPI and npm

**Acceptance Criteria:**
- Python SDK on PyPI
- Node.js SDK on npm
- Comprehensive documentation and examples

### Milestone 3.4: Multi-Cluster Federation (Week 15-16)

#### 3.4.1 Geographic Routing
**Priority:** LOW  
**Status:** 🔴 NOT STARTED  
**Files to Modify:**
- `core/gollek-engine/src/main/java/tech/kayys/gollek/engine/routing/ModelRouterService.java`

**Tasks:**
- [ ] Add region awareness to router
- [ ] Implement latency-based routing
- [ ] Create health check federation
- [ ] Add disaster recovery failover

**Acceptance Criteria:**
- Cross-region routing
- Automatic failover
- Latency-aware request distribution

---

## Implementation Progress Tracker

### Short Term Progress

| Task | Owner | Start Date | End Date | Status | Notes |
|------|-------|------------|----------|--------|-------|
| Vault Integration | Agent | 2026-03-27 | 2026-03-28 | 🔴 NOT STARTED | Critical for security compliance |
| PII Redaction Plugin | Agent | 2026-03-27 | 2026-03-28 | 🔴 NOT STARTED | GDPR/HIPAA requirement |
| GraalVM Native Image | Agent | 2026-03-28 | 2026-03-29 | 🔴 NOT STARTED | Serverless optimization |
| Test Coverage | Agent | 2026-03-28 | 2026-03-29 | 🔴 NOT STARTED | Target: >80% coverage |
| Contract Tests | Agent | 2026-03-29 | 2026-03-30 | 🔴 NOT STARTED | Plugin compatibility |

### Mid Term Progress

| Task | Owner | Start Date | End Date | Status | Notes |
|------|-------|------------|----------|--------|-------|
| Semantic Caching | - | TBD | TBD | 🔴 NOT STARTED | Performance optimization |
| KV Cache Enhancement | - | TBD | TBD | 🔴 NOT STARTED | Throughput improvement |
| RAG Plugin | - | TBD | TBD | 🔴 NOT STARTED | Enterprise knowledge base |
| Resilience4j | - | TBD | TBD | 🔴 NOT STARTED | Standardized reliability |
| Multi-Modal | - | TBD | TBD | 🔴 NOT STARTED | Image/audio support |

### Long Term Progress

| Task | Owner | Start Date | End Date | Status | Notes |
|------|-------|------------|----------|--------|-------|
| Event-Driven Audit | - | TBD | TBD | 🔴 NOT STARTED | Kafka integration |
| Distributed Tracing | - | TBD | TBD | 🔴 NOT STARTED | W3C propagation |
| CLI | - | TBD | TBD | 🔴 NOT STARTED | Developer experience |
| SDKs | - | TBD | TBD | 🔴 NOT STARTED | Python/Node.js |
| Multi-Cluster | - | TBD | TBD | 🔴 NOT STARTED | Global scale |

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GraalVM compatibility issues with SPI | HIGH | MEDIUM | Extensive testing, fallback to JVM |
| Vault integration complexity | MEDIUM | LOW | Use Quarkus Vault extension |
| Performance regression from PII redaction | MEDIUM | LOW | Benchmark, optimize regex patterns |
| Native image size too large | LOW | MEDIUM | Optimize dependencies, use shared libraries |

### Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking changes to SPI | HIGH | LOW | Version SPI, maintain backward compatibility |
| Insufficient test coverage | MEDIUM | MEDIUM | Enforce coverage thresholds in CI |
| Documentation lag | LOW | HIGH | Documentation as part of DoD |

---

## Success Metrics

### Short Term (Week 4)
- [ ] Vault integration operational
- [ ] PII redaction plugin deployed
- [ ] Native image builds successfully
- [ ] Test coverage > 80%
- [ ] Zero critical security vulnerabilities

### Mid Term (Week 10)
- [ ] Semantic cache hit rate > 30%
- [ ] 2x throughput improvement
- [ ] RAG latency < 100ms
- [ ] Resilience4j metrics dashboards
- [ ] Multi-modal API stable

### Long Term (Week 16)
- [ ] Audit event latency < 1ms
- [ ] End-to-end tracing operational
- [ ] CLI and SDKs published
- [ ] Multi-cluster failover tested
- [ ] Platform handles 10K RPS

---

## Dependencies

### External Dependencies
- **Quarkus 3.32.2** - Framework
- **HashiCorp Vault** - Secret management
- **GraalVM 21** - Native compilation
- **Testcontainers** - Integration testing
- **Resilience4j** - Reliability patterns
- **Apache Presidio** - PII detection (optional)

### Internal Dependencies
- **wayang-platform** - Parent platform
- **workflow-gamelan** - Workflow orchestration
- **mcp-kulit** - Model Context Protocol

---

## Appendix A: File Structure Changes

### New Modules to Create
```
inference-gollek/
├── plugins/
│   └── common/
│       ├── gollek-plugin-pii-redaction/          # NEW
│       │   ├── src/
│       │   ├── pom.xml
│       │   └── README.md
│       └── gollek-plugin-semantic-cache/         # NEW
│           ├── src/
│           ├── pom.xml
│           └── README.md
└── sdk/
    ├── gollek-python-sdk/                        # NEW (Phase 3)
    └── gollek-nodejs-sdk/                        # NEW (Phase 3)
```

### Modified Files
```
core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/ProviderConfigLoader.java
core/gollek-engine/src/main/java/tech/kayys/gollek/engine/NativeImageFeature.java
core/spi/gollek-spi/src/main/java/tech/kayys/gollek/spi/context/RequestContext.java
runtime/gollek-runtime-unified/src/main/resources/application.properties
```

---

## Appendix B: Configuration Examples

### Vault Configuration
```yaml
quarkus:
  vault:
    auth:
      method: kubernetes
      kubernetes:
        role: gollek-inference
        service-account-token-path: /var/run/secrets/kubernetes.io/serviceaccount/token
    kv:
      secret-engine-path: secret/
      cache:
        enabled: true
        expire-after-write: 5M
```

### PII Redaction Configuration
```yaml
gollek:
  plugins:
    pii-redaction:
      enabled: true
      patterns:
        - type: EMAIL
          regex: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
          replacement: "[REDACTED_EMAIL]"
        - type: PHONE
          regex: "\\+?[0-9]{10,15}"
          replacement: "[REDACTED_PHONE]"
        - type: CREDIT_CARD
          regex: "\\b[0-9]{13,19}\\b"
          replacement: "[REDACTED_CC]"
      audit:
        enabled: true
        log-redacted-fields: true
```

### Native Image Configuration
```xml
<profile>
  <id>native</id>
  <activation>
    <property>
      <name>native</name>
    </property>
  </activation>
  <properties>
    <quarkus.package.type>native</quarkus.package.type>
    <quarkus.native.container-build>true</quarkus.native.container-build>
    <quarkus.native.builder-image>quay.io/quarkus/ubi-quarkus-mandrel-builder-image:jdk-21</quarkus.native.builder-image>
  </properties>
</profile>
```

---

## Review and Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Technical Lead | | | |
| Product Owner | | | |
| Security Review | | | |
| Operations | | | |

---

**Next Review Date:** April 3, 2026  
**Document Owner:** Gollek Core Team
