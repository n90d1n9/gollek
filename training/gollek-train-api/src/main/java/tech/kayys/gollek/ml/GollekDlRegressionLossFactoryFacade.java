package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.nn.loss.GaussianNllLoss;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.L1Loss;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.PinballLoss;
import tech.kayys.gollek.ml.nn.loss.PredictionIntervalLoss;
import tech.kayys.gollek.ml.nn.loss.SmoothL1Loss;

/**
 * Regression and probabilistic regression loss helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlRegressionLossFactoryFacade extends GollekDlClassificationLossFactoryFacade {
    protected GollekDlRegressionLossFactoryFacade() {
    }

    public static MSELoss mseLoss() {
        return new MSELoss();
    }

    public static L1Loss l1Loss() {
        return new L1Loss();
    }

    public static HuberLoss huberLoss() {
        return new HuberLoss();
    }

    public static HuberLoss huberLoss(float delta) {
        return new HuberLoss(delta);
    }

    public static SmoothL1Loss smoothL1Loss() {
        return new SmoothL1Loss();
    }

    public static SmoothL1Loss smoothL1Loss(float beta) {
        return new SmoothL1Loss(beta);
    }

    public static PinballLoss pinballLoss() {
        return new PinballLoss();
    }

    public static PinballLoss pinballLoss(double quantile) {
        return new PinballLoss(quantile);
    }

    public static PinballLoss quantileLoss(double quantile) {
        return pinballLoss(quantile);
    }

    public static PinballLoss quantileLoss(double... quantiles) {
        return new PinballLoss(quantiles);
    }

    public static PinballLoss predictionIntervalLoss(double lowerQuantile, double upperQuantile) {
        return PinballLoss.interval(lowerQuantile, upperQuantile);
    }

    public static PredictionIntervalLoss intervalScoreLoss() {
        return new PredictionIntervalLoss();
    }

    public static PredictionIntervalLoss intervalScoreLoss(double alpha) {
        return new PredictionIntervalLoss(alpha);
    }

    public static PredictionIntervalLoss intervalScoreLoss(double alpha, double crossingPenalty) {
        return new PredictionIntervalLoss(alpha, crossingPenalty);
    }

    public static PredictionIntervalLoss winklerLoss() {
        return intervalScoreLoss();
    }

    public static PredictionIntervalLoss winklerLoss(double alpha) {
        return intervalScoreLoss(alpha);
    }

    public static PredictionIntervalLoss winklerLoss(double alpha, double crossingPenalty) {
        return intervalScoreLoss(alpha, crossingPenalty);
    }

    public static GaussianNllLoss gaussianNllLoss() {
        return new GaussianNllLoss();
    }

    public static GaussianNllLoss gaussianNllLoss(boolean includeConstant) {
        return new GaussianNllLoss(includeConstant);
    }

    public static GaussianNllLoss gaussianNllLoss(
            boolean includeConstant,
            double minLogVariance,
            double maxLogVariance) {
        return new GaussianNllLoss(includeConstant, (float) minLogVariance, (float) maxLogVariance);
    }

    public static GaussianNllLoss heteroscedasticGaussianLoss() {
        return gaussianNllLoss();
    }

    public static GaussianNllLoss heteroscedasticGaussianLoss(boolean includeConstant) {
        return gaussianNllLoss(includeConstant);
    }
}
