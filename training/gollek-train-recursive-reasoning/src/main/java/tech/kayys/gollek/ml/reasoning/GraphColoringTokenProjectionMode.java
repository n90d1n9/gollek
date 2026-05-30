package tech.kayys.gollek.ml.reasoning;

/**
 * Strategy used to convert raw per-node logits into graph-coloring tokens.
 */
public enum GraphColoringTokenProjectionMode {
    NODE_ARGMAX,
    NODE_COLOR_ARGMAX,
    NODE_COLOR_ARGMAX_RESPECT_FIXED
}
