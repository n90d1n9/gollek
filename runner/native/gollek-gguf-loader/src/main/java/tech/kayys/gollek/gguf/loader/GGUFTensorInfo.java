package tech.kayys.gollek.gguf.loader;

/**
 * Metadata and offset for a single tensor within a GGUF file.
 */
public record GGUFTensorInfo(
    String name,
    long[] shape,
    int typeId,
    long offset,
    long sizeInBytes
) {}
