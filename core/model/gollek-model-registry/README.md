# Gollek Model Registry

Model registry and repository module for the Gollek inference engine.

## Purpose

This module provides:
- Model metadata storage and retrieval
- Model versioning support
- Model registry services
- Repository pattern implementations for model artifacts

## Components

### Entities
- `Model` - Main model entity with metadata
- `ModelVersion` - Version-specific model information

### Repositories
- `ModelRepository` - Panache repository for Model entities
- `ModelVersionRepository` - Panache repository for ModelVersion entities
- `CachedModelRepository` - Cached wrapper for model repositories
- `ModelRepositoryRegistry` - Registry for repository providers

### Services
- `ModelRegistryService` - Main registry service implementing ModelRegistry SPI
- `ModelManagementService` - Model lifecycle management
- `ModelStatsProvider` - Statistics provider interface (implemented by engine)
- `ModelStorageService` - Storage service interface (implemented by engine/repo-core)

### Providers
- `ModelRepositoryProvider` - Provider interface for repository implementations

## Dependencies

- `gollek-spi-provider` - Core provider SPI
- `gollek-spi-model` - Model SPI interfaces
- `gollek-engine-core` - Core engine interfaces
- `gollek-error-code` - Error code definitions
- Quarkus dependencies (Hibernate Reactive, REST, Cache, etc.)

## Usage

The model registry is automatically discovered and used by the Gollek engine for:
- Registering new models
- Retrieving model manifests
- Managing model versions
- Querying model statistics

## Extension

To add custom model repository implementations:
1. Implement the `ModelRepository` interface from `gollek-model-core`
2. Annotate with `@ApplicationScoped`
3. The registry will automatically discover and use it
