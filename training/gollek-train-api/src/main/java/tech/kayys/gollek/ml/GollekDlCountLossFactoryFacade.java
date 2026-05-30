package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.nn.loss.NegativeBinomialNllLoss;
import tech.kayys.gollek.ml.nn.loss.PoissonNllLoss;
import tech.kayys.gollek.ml.nn.loss.TweedieNllLoss;

/**
 * Count and compound distribution loss helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlCountLossFactoryFacade extends GollekDlRegressionLossFactoryFacade {
    protected GollekDlCountLossFactoryFacade() {
    }

    public static PoissonNllLoss poissonNllLoss() {
        return new PoissonNllLoss();
    }

    public static PoissonNllLoss poissonNllLoss(boolean logInput) {
        return new PoissonNllLoss(logInput);
    }

    public static PoissonNllLoss poissonNllLoss(boolean logInput, boolean full) {
        return new PoissonNllLoss(logInput, full);
    }

    public static PoissonNllLoss poissonNllLoss(boolean logInput, boolean full, double eps) {
        return new PoissonNllLoss(logInput, full, (float) eps);
    }

    public static PoissonNllLoss countNllLoss() {
        return poissonNllLoss();
    }

    public static TweedieNllLoss tweedieNllLoss() {
        return new TweedieNllLoss();
    }

    public static TweedieNllLoss tweedieNllLoss(double power) {
        return new TweedieNllLoss(power);
    }

    public static TweedieNllLoss tweedieNllLoss(double power, boolean logInput) {
        return new TweedieNllLoss(power, logInput);
    }

    public static TweedieNllLoss tweedieNllLoss(double power, boolean logInput, double eps) {
        return new TweedieNllLoss(power, logInput, eps);
    }

    public static TweedieNllLoss compoundPoissonGammaLoss() {
        return tweedieNllLoss();
    }

    public static TweedieNllLoss compoundPoissonGammaLoss(double power) {
        return tweedieNllLoss(power);
    }

    public static NegativeBinomialNllLoss negativeBinomialNllLoss() {
        return new NegativeBinomialNllLoss();
    }

    public static NegativeBinomialNllLoss negativeBinomialNllLoss(boolean logInput) {
        return new NegativeBinomialNllLoss(logInput);
    }

    public static NegativeBinomialNllLoss negativeBinomialNllLoss(
            boolean logInput,
            boolean includeConstant) {
        return new NegativeBinomialNllLoss(logInput, includeConstant);
    }

    public static NegativeBinomialNllLoss negativeBinomialNllLoss(
            boolean logInput,
            boolean includeConstant,
            double eps) {
        return new NegativeBinomialNllLoss(logInput, includeConstant, eps);
    }

    public static NegativeBinomialNllLoss overdispersedCountNllLoss() {
        return negativeBinomialNllLoss();
    }
}
