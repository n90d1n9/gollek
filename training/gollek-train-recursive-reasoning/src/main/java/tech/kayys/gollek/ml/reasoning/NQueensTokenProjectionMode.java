package tech.kayys.gollek.ml.reasoning;

/**
 * Strategy used to convert raw per-cell logits into board tokens.
 */
public enum NQueensTokenProjectionMode {
    CELL_ARGMAX,
    ROW_QUEEN_ARGMAX,
    ROW_QUEEN_ARGMAX_RESPECT_FIXED
}
