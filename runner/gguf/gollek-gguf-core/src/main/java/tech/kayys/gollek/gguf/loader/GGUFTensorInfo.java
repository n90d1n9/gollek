package tech.kayys.gollek.gguf.loader;

import io.quarkus.runtime.annotations.RegisterForReflection;

/**
 * Metadata and offset for a single tensor within a GGUF file.
 */
@RegisterForReflection
public record GGUFTensorInfo(
    String name,
    long[] shape,
    int typeId,
    long offset,
    long sizeInBytes
) {}
