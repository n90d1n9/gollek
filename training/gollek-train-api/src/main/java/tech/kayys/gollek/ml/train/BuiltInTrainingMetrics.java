package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Package-private facade behind the public {@link TrainingMetrics} catalog.
 */
final class BuiltInTrainingMetrics {
    private BuiltInTrainingMetrics() {
    }

    static Supplier<TrainingMetric> classificationAccuracy() {
        return ClassificationTrainingMetrics.classificationAccuracy();
    }

    static Supplier<TrainingMetric> accuracy() {
        return ClassificationTrainingMetrics.accuracy();
    }

    static Supplier<TrainingMetric> classificationConfusionMatrix() {
        return ClassificationTrainingMetrics.classificationConfusionMatrix();
    }

    static Supplier<TrainingMetric> confusionMatrix() {
        return ClassificationTrainingMetrics.confusionMatrix();
    }

    static Supplier<TrainingMetric> topKAccuracy(int k) {
        return ClassificationTrainingMetrics.topKAccuracy(k);
    }

    static Supplier<TrainingMetric> binaryAccuracy() {
        return BinaryTrainingMetrics.binaryAccuracy();
    }

    static Supplier<TrainingMetric> binaryAccuracy(float logitThreshold) {
        return BinaryTrainingMetrics.binaryAccuracy(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix() {
        return BinaryTrainingMetrics.binaryConfusionMatrix();
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix(float logitThreshold) {
        return BinaryTrainingMetrics.binaryConfusionMatrix(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryPrecision() {
        return BinaryTrainingMetrics.binaryPrecision();
    }

    static Supplier<TrainingMetric> binaryPrecision(float logitThreshold) {
        return BinaryTrainingMetrics.binaryPrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryRecall() {
        return BinaryTrainingMetrics.binaryRecall();
    }

    static Supplier<TrainingMetric> binaryRecall(float logitThreshold) {
        return BinaryTrainingMetrics.binaryRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryF1() {
        return BinaryTrainingMetrics.binaryF1();
    }

    static Supplier<TrainingMetric> binaryF1(float logitThreshold) {
        return BinaryTrainingMetrics.binaryF1(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryRocAuc() {
        return BinaryTrainingMetrics.binaryRocAuc();
    }

    static Supplier<TrainingMetric> binaryAuroc() {
        return BinaryTrainingMetrics.binaryAuroc();
    }

    static Supplier<TrainingMetric> binaryAveragePrecision() {
        return BinaryTrainingMetrics.binaryAveragePrecision();
    }

    static Supplier<TrainingMetric> multiLabelExactMatch() {
        return MultiLabelTrainingMetrics.multiLabelExactMatch();
    }

    static Supplier<TrainingMetric> multiLabelExactMatch(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelExactMatch(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss() {
        return MultiLabelTrainingMetrics.multiLabelHammingLoss();
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelHammingLoss(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision() {
        return MultiLabelTrainingMetrics.multiLabelMacroPrecision();
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroPrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall() {
        return MultiLabelTrainingMetrics.multiLabelMacroRecall();
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroF1() {
        return MultiLabelTrainingMetrics.multiLabelMacroF1();
    }

    static Supplier<TrainingMetric> multiLabelMacroF1(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroF1(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRocAuc() {
        return MultiLabelTrainingMetrics.multiLabelMacroRocAuc();
    }

    static Supplier<TrainingMetric> multiLabelMacroAuroc() {
        return MultiLabelTrainingMetrics.multiLabelMacroAuroc();
    }

    static Supplier<TrainingMetric> multiLabelMacroAveragePrecision() {
        return MultiLabelTrainingMetrics.multiLabelMacroAveragePrecision();
    }

    static Supplier<TrainingMetric> precision() {
        return ClassificationTrainingMetrics.precision();
    }

    static Supplier<TrainingMetric> recall() {
        return ClassificationTrainingMetrics.recall();
    }

    static Supplier<TrainingMetric> f1() {
        return ClassificationTrainingMetrics.f1();
    }

    static Supplier<TrainingMetric> macroF1() {
        return ClassificationTrainingMetrics.macroF1();
    }

    static Supplier<TrainingMetric> classificationMacroRocAuc() {
        return ClassificationTrainingMetrics.classificationMacroRocAuc();
    }

    static Supplier<TrainingMetric> classificationMacroAuroc() {
        return ClassificationTrainingMetrics.classificationMacroAuroc();
    }

    static Supplier<TrainingMetric> classificationMacroAveragePrecision() {
        return ClassificationTrainingMetrics.classificationMacroAveragePrecision();
    }

    static Supplier<TrainingMetric> meanAbsoluteError() {
        return RegressionTrainingMetrics.meanAbsoluteError();
    }

    static Supplier<TrainingMetric> mae() {
        return RegressionTrainingMetrics.mae();
    }

    static Supplier<TrainingMetric> meanSquaredError() {
        return RegressionTrainingMetrics.meanSquaredError();
    }

    static Supplier<TrainingMetric> mse() {
        return RegressionTrainingMetrics.mse();
    }

    static Supplier<TrainingMetric> rootMeanSquaredError() {
        return RegressionTrainingMetrics.rootMeanSquaredError();
    }

    static Supplier<TrainingMetric> rmse() {
        return RegressionTrainingMetrics.rmse();
    }

    static Supplier<TrainingMetric> r2Score() {
        return RegressionTrainingMetrics.r2Score();
    }

    static Supplier<TrainingMetric> r2() {
        return RegressionTrainingMetrics.r2();
    }
}
