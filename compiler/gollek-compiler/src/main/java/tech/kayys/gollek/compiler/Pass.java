package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.GGraph;

/**
 * COMPILER GOAL
 * Transform:
 * High-level IR (attention, ffn, etc.)
 * Optimized IR (fused ops, layout-aware)
 * ↓
 * ↓
 * Executable graph
 */
public interface Pass {
    String name();

    GGraph apply(GGraph graph, PassContext ctx);
}