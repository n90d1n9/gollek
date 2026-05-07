package tech.kayys.gollek.conversion.ggx;

/**
 * Defines the contract for an input/output within a GGX file.
 * This aligns the serialization format with the runtime GData execution context.
 */
public record GGXBinding(
    String name,
    String type,    // e.g. "tensor", "scalar", "handle"
    String shape
) {}
