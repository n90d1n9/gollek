package tech.kayys.gollek.ml.reasoning;

/**
 * Pluggable reconstruction/LPRM/ACT loss evaluator for GRAM deep supervision.
 */
@FunctionalInterface
public interface GramTerminalLossEvaluator {
    GramTerminalLossTerms evaluate(GramTerminalLossInput input);
}
