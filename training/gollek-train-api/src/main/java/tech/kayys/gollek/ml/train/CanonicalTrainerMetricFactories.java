package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Legacy metric factory catalog inherited by {@link CanonicalTrainer.Metrics}.
 */
@SuppressWarnings("deprecation")
abstract class CanonicalTrainerMetricFactories {

    public static Supplier<CanonicalTrainer.Metric> classificationAccuracy() {
        return legacy(TrainingMetrics.classificationAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> accuracy() {
        return legacy(TrainingMetrics.accuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationConfusionMatrix() {
        return legacy(TrainingMetrics.classificationConfusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> confusionMatrix() {
        return legacy(TrainingMetrics.confusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> topKAccuracy(int k) {
        return legacy(TrainingMetrics.topKAccuracy(k));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracy() {
        return legacy(TrainingMetrics.binaryAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracy(float logitThreshold) {
        return legacy(TrainingMetrics.binaryAccuracy(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrix() {
        return legacy(TrainingMetrics.binaryConfusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrix(float logitThreshold) {
        return legacy(TrainingMetrics.binaryConfusionMatrix(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecision() {
        return legacy(TrainingMetrics.binaryPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecision(float logitThreshold) {
        return legacy(TrainingMetrics.binaryPrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecall() {
        return legacy(TrainingMetrics.binaryRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecall(float logitThreshold) {
        return legacy(TrainingMetrics.binaryRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1() {
        return legacy(TrainingMetrics.binaryF1());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1(float logitThreshold) {
        return legacy(TrainingMetrics.binaryF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRocAuc() {
        return legacy(TrainingMetrics.binaryRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAuroc() {
        return legacy(TrainingMetrics.binaryAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAveragePrecision() {
        return legacy(TrainingMetrics.binaryAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatch() {
        return legacy(TrainingMetrics.multiLabelExactMatch());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatch(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelExactMatch(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLoss() {
        return legacy(TrainingMetrics.multiLabelHammingLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLoss(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelHammingLoss(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecision() {
        return legacy(TrainingMetrics.multiLabelMacroPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecision(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroPrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecall() {
        return legacy(TrainingMetrics.multiLabelMacroRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecall(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1() {
        return legacy(TrainingMetrics.multiLabelMacroF1());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRocAuc() {
        return legacy(TrainingMetrics.multiLabelMacroRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAuroc() {
        return legacy(TrainingMetrics.multiLabelMacroAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAveragePrecision() {
        return legacy(TrainingMetrics.multiLabelMacroAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> precision() {
        return legacy(TrainingMetrics.precision());
    }

    public static Supplier<CanonicalTrainer.Metric> recall() {
        return legacy(TrainingMetrics.recall());
    }

    public static Supplier<CanonicalTrainer.Metric> f1() {
        return legacy(TrainingMetrics.f1());
    }

    public static Supplier<CanonicalTrainer.Metric> macroF1() {
        return legacy(TrainingMetrics.macroF1());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroRocAuc() {
        return legacy(TrainingMetrics.classificationMacroRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAuroc() {
        return legacy(TrainingMetrics.classificationMacroAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAveragePrecision() {
        return legacy(TrainingMetrics.classificationMacroAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> meanAbsoluteError() {
        return legacy(TrainingMetrics.meanAbsoluteError());
    }

    public static Supplier<CanonicalTrainer.Metric> mae() {
        return legacy(TrainingMetrics.mae());
    }

    public static Supplier<CanonicalTrainer.Metric> meanSquaredError() {
        return legacy(TrainingMetrics.meanSquaredError());
    }

    public static Supplier<CanonicalTrainer.Metric> mse() {
        return legacy(TrainingMetrics.mse());
    }

    public static Supplier<CanonicalTrainer.Metric> rootMeanSquaredError() {
        return legacy(TrainingMetrics.rootMeanSquaredError());
    }

    public static Supplier<CanonicalTrainer.Metric> rmse() {
        return legacy(TrainingMetrics.rmse());
    }

    public static Supplier<CanonicalTrainer.Metric> r2Score() {
        return legacy(TrainingMetrics.r2Score());
    }

    public static Supplier<CanonicalTrainer.Metric> r2() {
        return legacy(TrainingMetrics.r2());
    }

    private static Supplier<CanonicalTrainer.Metric> legacy(Supplier<? extends TrainingMetric> factory) {
        return TrainerLegacyMetrics.legacy(factory);
    }
}
