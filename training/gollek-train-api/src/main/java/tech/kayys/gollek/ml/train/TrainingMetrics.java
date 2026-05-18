package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Built-in metric factories decoupled from the trainer type.
 */
public final class TrainingMetrics {
    private TrainingMetrics() {
    }

    public static Supplier<TrainingMetric> classificationAccuracy() {
        return BuiltInTrainingMetrics.classificationAccuracy();
    }

    public static Supplier<TrainingMetric> accuracy() {
        return classificationAccuracy();
    }

    public static Supplier<TrainingMetric> classificationConfusionMatrix() {
        return BuiltInTrainingMetrics.classificationConfusionMatrix();
    }

    public static Supplier<TrainingMetric> confusionMatrix() {
        return classificationConfusionMatrix();
    }

    public static Supplier<TrainingMetric> topKAccuracy(int k) {
        return BuiltInTrainingMetrics.topKAccuracy(k);
    }

    public static Supplier<TrainingMetric> binaryAccuracy() {
        return BuiltInTrainingMetrics.binaryAccuracy();
    }

    public static Supplier<TrainingMetric> binaryAccuracy(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryAccuracy(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryConfusionMatrix() {
        return BuiltInTrainingMetrics.binaryConfusionMatrix();
    }

    public static Supplier<TrainingMetric> binaryConfusionMatrix(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryConfusionMatrix(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryPrecision() {
        return BuiltInTrainingMetrics.binaryPrecision();
    }

    public static Supplier<TrainingMetric> binaryPrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryPrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryRecall() {
        return BuiltInTrainingMetrics.binaryRecall();
    }

    public static Supplier<TrainingMetric> binaryRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryF1() {
        return BuiltInTrainingMetrics.binaryF1();
    }

    public static Supplier<TrainingMetric> binaryF1(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryRocAuc() {
        return BuiltInTrainingMetrics.binaryRocAuc();
    }

    public static Supplier<TrainingMetric> binaryAuroc() {
        return binaryRocAuc();
    }

    public static Supplier<TrainingMetric> binaryAveragePrecision() {
        return BuiltInTrainingMetrics.binaryAveragePrecision();
    }

    public static Supplier<TrainingMetric> multiLabelExactMatch() {
        return BuiltInTrainingMetrics.multiLabelExactMatch();
    }

    public static Supplier<TrainingMetric> multiLabelExactMatch(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelExactMatch(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelHammingLoss() {
        return BuiltInTrainingMetrics.multiLabelHammingLoss();
    }

    public static Supplier<TrainingMetric> multiLabelHammingLoss(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelHammingLoss(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroPrecision() {
        return BuiltInTrainingMetrics.multiLabelMacroPrecision();
    }

    public static Supplier<TrainingMetric> multiLabelMacroPrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroPrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroRecall() {
        return BuiltInTrainingMetrics.multiLabelMacroRecall();
    }

    public static Supplier<TrainingMetric> multiLabelMacroRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroF1() {
        return BuiltInTrainingMetrics.multiLabelMacroF1();
    }

    public static Supplier<TrainingMetric> multiLabelMacroF1(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroRocAuc() {
        return BuiltInTrainingMetrics.multiLabelMacroRocAuc();
    }

    public static Supplier<TrainingMetric> multiLabelMacroAuroc() {
        return multiLabelMacroRocAuc();
    }

    public static Supplier<TrainingMetric> multiLabelMacroAveragePrecision() {
        return BuiltInTrainingMetrics.multiLabelMacroAveragePrecision();
    }

    public static Supplier<TrainingMetric> precision() {
        return BuiltInTrainingMetrics.precision();
    }

    public static Supplier<TrainingMetric> recall() {
        return BuiltInTrainingMetrics.recall();
    }

    public static Supplier<TrainingMetric> f1() {
        return BuiltInTrainingMetrics.f1();
    }

    public static Supplier<TrainingMetric> macroF1() {
        return f1();
    }

    public static Supplier<TrainingMetric> classificationMacroRocAuc() {
        return BuiltInTrainingMetrics.classificationMacroRocAuc();
    }

    public static Supplier<TrainingMetric> classificationMacroAuroc() {
        return classificationMacroRocAuc();
    }

    public static Supplier<TrainingMetric> classificationMacroAveragePrecision() {
        return BuiltInTrainingMetrics.classificationMacroAveragePrecision();
    }

    public static Supplier<TrainingMetric> meanAbsoluteError() {
        return BuiltInTrainingMetrics.meanAbsoluteError();
    }

    public static Supplier<TrainingMetric> mae() {
        return meanAbsoluteError();
    }

    public static Supplier<TrainingMetric> meanSquaredError() {
        return BuiltInTrainingMetrics.meanSquaredError();
    }

    public static Supplier<TrainingMetric> mse() {
        return meanSquaredError();
    }

    public static Supplier<TrainingMetric> rootMeanSquaredError() {
        return BuiltInTrainingMetrics.rootMeanSquaredError();
    }

    public static Supplier<TrainingMetric> rmse() {
        return rootMeanSquaredError();
    }

    public static Supplier<TrainingMetric> r2Score() {
        return BuiltInTrainingMetrics.r2Score();
    }

    public static Supplier<TrainingMetric> r2() {
        return r2Score();
    }
}
