package tech.kayys.gollek.gguf.loader;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

/**
 * Representation of a parsed GGUF model in memory.
 */
public record GGUFModel(
    int version,
    Map<String, Object> metadata,
    List<GGUFTensorInfo> tensors,
    long dataStart,
    MemorySegment segment
) {}
