package tech.kayys.gollek.conversion.ggx;

import java.util.List;

/**
 * Represents the layout of a compiled graph (GGX).
 * Extended to include strictly typed bindings for the new runtime ExecutionContext.
 */
public final class GGXFile {
    public List<GGXBinding> inputs;
    public List<GGXBinding> outputs;
    
    // Existing fields for weights, graph metadata, etc.
}
