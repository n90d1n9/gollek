package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.layer.ReLU;
import tech.kayys.gollek.ml.nn.layer.Dropout;
import tech.kayys.gollek.ml.nn.layer.Sequential;
import tech.kayys.gollek.ml.nn.layer.LayerNorm;
import tech.kayys.gollek.ml.nn.layer.Embedding;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.BinaryFocalWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.FocalLoss;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.L1Loss;
import tech.kayys.gollek.ml.nn.loss.SmoothL1Loss;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.ml.multimodal.VisionBuilder;
import tech.kayys.gollek.ml.multimodal.MultimodalBuilder;
import tech.kayys.gollek.ml.multimodal.VideoBuilder;
import tech.kayys.gollek.ml.multimodal.AudioBuilder;
import tech.kayys.gollek.ml.export.Benchmark;
import tech.kayys.gollek.ml.export.ModelExporter;
import tech.kayys.gollek.ml.hub.HubConfig;
import tech.kayys.gollek.ml.hub.ModelHub;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

import tech.kayys.gollek.ml.optim.*;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.svm.*;
import tech.kayys.gollek.ml.naive_bayes.*;
import tech.kayys.gollek.ml.linear_model.*;
import tech.kayys.gollek.ml.clustering.*;
import tech.kayys.gollek.ml.pipeline.*;
import tech.kayys.gollek.ml.feature.PolynomialFeatures;
import tech.kayys.gollek.ml.model_selection.*;
import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.base.BaseTransformer;
import tech.kayys.gollek.ml.util.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch, while {@code Gollek.ML} provides
 * a scikit-learn style API for traditional machine learning.
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.1";

    private Gollek() {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Deep Learning (PyTorch Style)
    // ══════════════════════════════════════════════════════════════════════

    public static class DL {
        public enum TrainingPreset {
            REGRESSION_MSE_ADAMW,
            REGRESSION_MSE_SGD,
            REGRESSION_HUBER_ADAMW,
            REGRESSION_HUBER_SGD,
            CLASSIFICATION_CROSS_ENTROPY_ADAMW,
            CLASSIFICATION_CROSS_ENTROPY_SGD,
            CLASSIFICATION_FOCAL_ADAMW,
            CLASSIFICATION_FOCAL_SGD,
            BINARY_FOCAL_WITH_LOGITS_ADAMW,
            BINARY_FOCAL_WITH_LOGITS_SGD,
            BINARY_BCE_WITH_LOGITS_ADAMW,
            BINARY_BCE_WITH_LOGITS_SGD
        }

        public static TrainingOptions.Builder trainingOptions() {
            return TrainingOptions.builder();
        }

        public static GradTensor tensor(float[] data, long... shape) {
            return Gollek.tensor(data, shape);
        }

        public static NNModule sequential(NNModule... layers) {
            return new Sequential(layers);
        }

        public static Optimizer adamW(List<Parameter> params, float lr) {
            return AdamW.builder(params, lr).build();
        }

        public static WarmupCosineScheduler warmupCosineScheduler(
                Optimizer optimizer,
                int warmupSteps,
                int totalSteps,
                float maxLr,
                float minLr) {
            return new WarmupCosineScheduler(optimizer, warmupSteps, totalSteps, maxLr, minLr);
        }

        public static ReduceLROnPlateau reduceLrOnPlateauScheduler(
                Optimizer optimizer,
                ReduceLROnPlateau.Mode mode,
                float factor,
                int patience,
                double threshold,
                int cooldown,
                float minLr) {
            return new ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, cooldown, minLr);
        }

        public static CrossEntropyLoss crossEntropy() {
            return new CrossEntropyLoss();
        }

        public static CrossEntropyLoss crossEntropy(float[] classWeights) {
            return new CrossEntropyLoss(classWeights);
        }

        public static FocalLoss focalLoss() {
            return new FocalLoss();
        }

        public static FocalLoss focalLoss(float gamma) {
            return new FocalLoss(gamma);
        }

        public static FocalLoss focalLoss(float gamma, float alpha) {
            return new FocalLoss(gamma, alpha);
        }

        public static FocalLoss focalLoss(float gamma, float[] classWeights) {
            return new FocalLoss(gamma, classWeights);
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

        public static BCEWithLogitsLoss bceWithLogitsLoss() {
            return new BCEWithLogitsLoss();
        }

        public static BCEWithLogitsLoss bceWithLogitsLoss(float positiveWeight) {
            return new BCEWithLogitsLoss(positiveWeight);
        }

        public static BCEWithLogitsLoss bceWithLogitsLoss(float[] positiveWeights) {
            return new BCEWithLogitsLoss(positiveWeights);
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits() {
            return bceWithLogitsLoss();
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float positiveWeight) {
            return bceWithLogitsLoss(positiveWeight);
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float[] positiveWeights) {
            return bceWithLogitsLoss(positiveWeights);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss() {
            return new BinaryFocalWithLogitsLoss();
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(float gamma, float alpha) {
            return new BinaryFocalWithLogitsLoss(gamma, alpha);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
                float gamma,
                float alpha,
                float positiveWeight) {
            return new BinaryFocalWithLogitsLoss(gamma, alpha, positiveWeight);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
                float gamma,
                float alpha,
                float[] positiveWeights) {
            return new BinaryFocalWithLogitsLoss(gamma, alpha, positiveWeights);
        }

        public static CanonicalTrainer.Builder trainer() {
            return CanonicalTrainer.builder();
        }

        public static tech.kayys.gollek.train.diffusion.opd.DefaultDiffusionOpdTrainer.Builder diffusionOpdTrainer() {
            return tech.kayys.gollek.train.diffusion.opd.DefaultDiffusionOpdTrainer.builder();
        }

        public static Acceleration.BackendStatus accelerationStatus() {
            return Acceleration.status();
        }

        public static Acceleration.BackendStatus accelerationStatus(String deviceId) {
            return Acceleration.status(deviceId);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> accuracyMetric() {
            return CanonicalTrainer.Metrics.accuracy();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> confusionMatrixMetric() {
            return CanonicalTrainer.Metrics.confusionMatrix();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationConfusionMatrixMetric() {
            return CanonicalTrainer.Metrics.classificationConfusionMatrix();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> topKAccuracyMetric(int k) {
            return CanonicalTrainer.Metrics.topKAccuracy(k);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric() {
            return CanonicalTrainer.Metrics.binaryAccuracy();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric(float logitThreshold) {
            return CanonicalTrainer.Metrics.binaryAccuracy(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric() {
            return CanonicalTrainer.Metrics.binaryConfusionMatrix();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.binaryConfusionMatrix(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric() {
            return CanonicalTrainer.Metrics.binaryPrecision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric(float logitThreshold) {
            return CanonicalTrainer.Metrics.binaryPrecision(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRecallMetric() {
            return CanonicalTrainer.Metrics.binaryRecall();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRecallMetric(float logitThreshold) {
            return CanonicalTrainer.Metrics.binaryRecall(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryF1Metric() {
            return CanonicalTrainer.Metrics.binaryF1();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryF1Metric(float logitThreshold) {
            return CanonicalTrainer.Metrics.binaryF1(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRocAucMetric() {
            return CanonicalTrainer.Metrics.binaryRocAuc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAurocMetric() {
            return CanonicalTrainer.Metrics.binaryAuroc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAveragePrecisionMetric() {
            return CanonicalTrainer.Metrics.binaryAveragePrecision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric() {
            return CanonicalTrainer.Metrics.multiLabelExactMatch();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.multiLabelExactMatch(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric() {
            return CanonicalTrainer.Metrics.multiLabelHammingLoss();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.multiLabelHammingLoss(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric() {
            return CanonicalTrainer.Metrics.multiLabelMacroPrecision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.multiLabelMacroPrecision(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric() {
            return CanonicalTrainer.Metrics.multiLabelMacroRecall();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.multiLabelMacroRecall(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric() {
            return CanonicalTrainer.Metrics.multiLabelMacroF1();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric(
                float logitThreshold) {
            return CanonicalTrainer.Metrics.multiLabelMacroF1(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRocAucMetric() {
            return CanonicalTrainer.Metrics.multiLabelMacroRocAuc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroAurocMetric() {
            return CanonicalTrainer.Metrics.multiLabelMacroAuroc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroAveragePrecisionMetric() {
            return CanonicalTrainer.Metrics.multiLabelMacroAveragePrecision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> precisionMetric() {
            return CanonicalTrainer.Metrics.precision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> recallMetric() {
            return CanonicalTrainer.Metrics.recall();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> f1Metric() {
            return CanonicalTrainer.Metrics.f1();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> macroF1Metric() {
            return CanonicalTrainer.Metrics.macroF1();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroRocAucMetric() {
            return CanonicalTrainer.Metrics.classificationMacroRocAuc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroAurocMetric() {
            return CanonicalTrainer.Metrics.classificationMacroAuroc();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroAveragePrecisionMetric() {
            return CanonicalTrainer.Metrics.classificationMacroAveragePrecision();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> meanAbsoluteErrorMetric() {
            return CanonicalTrainer.Metrics.meanAbsoluteError();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> maeMetric() {
            return CanonicalTrainer.Metrics.mae();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> meanSquaredErrorMetric() {
            return CanonicalTrainer.Metrics.meanSquaredError();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> mseMetric() {
            return CanonicalTrainer.Metrics.mse();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> rootMeanSquaredErrorMetric() {
            return CanonicalTrainer.Metrics.rootMeanSquaredError();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> rmseMetric() {
            return CanonicalTrainer.Metrics.rmse();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> r2ScoreMetric() {
            return CanonicalTrainer.Metrics.r2Score();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> r2Metric() {
            return CanonicalTrainer.Metrics.r2();
        }

        public static DataLoader.TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
            return DataLoader.tensorDataset(inputs, labels);
        }

        public static GradTensor classLabels(int... labels) {
            return DataLoader.classLabels(labels);
        }

        public static float[] classWeights(int... labels) {
            return DataLoader.classWeights(labels);
        }

        public static float[] classWeightsFor(int numClasses, int... labels) {
            return DataLoader.classWeightsFor(numClasses, labels);
        }

        public static GradTensor binaryLabels(int... labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(boolean... labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(float... labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(int[][] labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(boolean[][] labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(float[][] labels) {
            return DataLoader.binaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(int[][] labels) {
            return binaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(boolean[][] labels) {
            return binaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(float[][] labels) {
            return binaryLabels(labels);
        }

        public static float binaryPositiveWeight(int... labels) {
            return DataLoader.binaryPositiveWeight(labels);
        }

        public static float binaryPositiveWeight(boolean... labels) {
            return DataLoader.binaryPositiveWeight(labels);
        }

        public static float binaryPositiveWeight(float... labels) {
            return DataLoader.binaryPositiveWeight(labels);
        }

        public static float[] multiLabelPositiveWeights(int[][] labels) {
            return DataLoader.multiLabelPositiveWeights(labels);
        }

        public static float[] multiLabelPositiveWeights(boolean[][] labels) {
            return DataLoader.multiLabelPositiveWeights(labels);
        }

        public static float[] multiLabelPositiveWeights(float[][] labels) {
            return DataLoader.multiLabelPositiveWeights(labels);
        }

        public static DataLoader.TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
            return DataLoader.classificationDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
            return DataLoader.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
            return DataLoader.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
            return DataLoader.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
            return DataLoader.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, int[][] labels) {
            return binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, boolean[][] labels) {
            return binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, float[][] labels) {
            return binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDatasetSplit split(
                GradTensor inputs,
                GradTensor labels,
                double trainFraction,
                long seed) {
            return DataLoader.split(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return DataLoader.classificationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationStratifiedSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return DataLoader.classificationStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryStratifiedSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return DataLoader.binaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit split(
                DataLoader.TensorDataset dataset,
                double trainFraction,
                long seed) {
            Objects.requireNonNull(dataset, "dataset must not be null");
            return dataset.split(trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit trainValidationSplit(
                GradTensor inputs,
                GradTensor labels,
                double trainFraction,
                long seed) {
            return split(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit trainValidationSplit(
                DataLoader.TensorDataset dataset,
                double trainFraction,
                long seed) {
            return split(dataset, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return classificationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return classificationStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return binaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorBuilder dataLoader(DataLoader.TensorDatasetAdapter dataset) {
            return DataLoader.tensorBuilder(dataset);
        }

        public static DataLoader.TensorDataLoader dataLoader(
                GradTensor inputs,
                GradTensor labels,
                int batchSize) {
            return DataLoader.tensors(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader classificationDataLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return DataLoader.classification(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return DataLoader.binary(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return DataLoader.binary(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return DataLoader.binary(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return DataLoader.binary(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader classificationLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return classificationDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return binaryDataLoader(inputs, labels, batchSize);
        }

        @SafeVarargs
        public static EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                TrainingPreset preset,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return evaluate(model, loader, preset, "auto", metrics);
        }

        @SafeVarargs
        public static EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                TrainingPreset preset,
                String deviceId,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            Objects.requireNonNull(preset, "preset must not be null");
            return evaluate(model, loader, presetLoss(preset), deviceId, metrics);
        }

        @SafeVarargs
        public static EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                CanonicalTrainer.LossFunction loss,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return evaluate(model, loader, loss, "auto", metrics);
        }

        @SafeVarargs
        public static EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                CanonicalTrainer.LossFunction loss,
                String deviceId,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories =
                    metrics == null ? List.of() : Arrays.asList(metrics);
            return evaluate(model, loader, loss, deviceId, metricFactories);
        }

        public static EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                CanonicalTrainer.LossFunction loss,
                String deviceId,
                List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
            Objects.requireNonNull(model, "model must not be null");
            Objects.requireNonNull(loader, "loader must not be null");
            Objects.requireNonNull(loss, "loss must not be null");

            String requestedDevice = deviceId == null || deviceId.isBlank() ? "auto" : deviceId.trim();
            List<CanonicalTrainer.Metric> resolvedMetrics = instantiateEvaluationMetrics(metricFactories);
            boolean restoreTraining = model.isTraining();
            double weightedLoss = 0.0;
            long lossWeight = 0;
            long sampleCount = 0;
            int batchCount = 0;

            try (Acceleration.Scope ignored = Acceleration.prefer(requestedDevice);
                    NoGrad ignoredNoGrad = NoGrad.enter()) {
                Acceleration.BackendStatus startStatus = Acceleration.status(requestedDevice);
                model.eval();
                for (Batch batch : loader) {
                    Objects.requireNonNull(batch, "evaluation loader produced null batch");
                    GradTensor prediction = model.forward(batch.inputs());
                    GradTensor lossTensor = Objects.requireNonNull(
                            loss.compute(prediction, batch.labels()),
                            "loss function returned null");
                    double batchLoss = requireFiniteEvaluationLoss(lossTensor.item());
                    long weight = Math.max(1L, batch.labels().numel());
                    weightedLoss += batchLoss * weight;
                    lossWeight += weight;
                    sampleCount += batchSampleCount(batch);
                    batchCount++;
                    for (CanonicalTrainer.Metric metric : resolvedMetrics) {
                        metric.update(prediction, batch.labels());
                    }
                }

                Map<String, Double> metricValues = new LinkedHashMap<>();
                for (CanonicalTrainer.Metric metric : resolvedMetrics) {
                    metricValues.put(requireEvaluationMetricName(metric.name()), metric.value());
                }
                Map<String, Object> metricDetails = evaluationMetricDetails(resolvedMetrics);
                Acceleration.BackendStatus endStatus = Acceleration.status(requestedDevice);
                Map<String, Object> metadata = new LinkedHashMap<>();
                metadata.put("requestedDevice", requestedDevice);
                metadata.put("executionBackend", endStatus.id());
                metadata.put("executionDeviceName", endStatus.deviceName());
                metadata.put("executionAccelerated", endStatus.accelerated());
                metadata.put("requestedDeviceAvailable", endStatus.available());
                metadata.put("acceleratedMatmulCalls", endStatus.acceleratedMatmulCalls());
                metadata.put("acceleratedMatmulCallsAtStart", startStatus.acceleratedMatmulCalls());
                metadata.put("metricsEnabled", !resolvedMetrics.isEmpty());
                metadata.put("metricDetails", metricDetails);

                double meanLoss = lossWeight == 0 ? Double.NaN : weightedLoss / lossWeight;
                return new EvaluationSummary(
                        meanLoss,
                        batchCount,
                        sampleCount,
                        Collections.unmodifiableMap(metricValues),
                        metricDetails,
                        Collections.unmodifiableMap(metadata));
            } finally {
                if (restoreTraining) {
                    model.train();
                } else {
                    model.eval();
                }
            }
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset) {
            return fit(model, trainLoader, null, epochs, learningRate, preset, 1.0, 0, 0.0);
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                int epochs,
                float learningRate,
                TrainingPreset preset) {
            return fit(
                    model,
                    split,
                    batchSize,
                    true,
                    0L,
                    epochs,
                    learningRate,
                    preset,
                    trainingOptions().build());
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                TrainingOptions options) {
            return fit(model, split, batchSize, true, 0L, epochs, learningRate, preset, options);
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                boolean shuffleTraining,
                long shuffleSeed,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                TrainingOptions options) {
            Objects.requireNonNull(split, "split must not be null");
            DataLoader.TensorDataLoader trainLoader = split.trainLoader(batchSize, shuffleTraining, shuffleSeed);
            DataLoader.TensorDataLoader validationLoader = split.validationLoader(batchSize);
            return fit(model, trainLoader, validationLoader, epochs, learningRate, preset, options);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                String deviceId) {
            return fit(
                    model,
                    trainLoader,
                    null,
                    epochs,
                    learningRate,
                    preset,
                    trainingOptions().device(deviceId).build());
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip) {
            return fit(model, trainLoader, validationLoader, epochs, learningRate, preset, gradientClip, 0, 0.0);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                TrainingOptions options) {
            TrainingOptions resolvedOptions = options == null ? trainingOptions().build() : options;
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    resolvedOptions.gradientClip(),
                    resolvedOptions.earlyStoppingPatience(),
                    resolvedOptions.earlyStoppingMinDelta(),
                    resolvedOptions.earlyStoppingMonitorMetric(),
                    resolvedOptions.earlyStoppingMonitorMode(),
                    resolvedOptions.checkpointDir(),
                    resolvedOptions.resumeFromCheckpoint(),
                    resolvedOptions.failOnCheckpointLoadError(),
                    resolvedOptions.saveBestModelCheckpoint(),
                    resolvedOptions.restoreBestModelAtEnd(),
                    resolvedOptions.bestModelMonitorMetric(),
                    resolvedOptions.bestModelMonitorMode(),
                    resolvedOptions.gradientAccumulationSteps(),
                    resolvedOptions.device(),
                    resolvedOptions.metricFactories(),
                    resolvedOptions.schedulerFactory(),
                    resolvedOptions.schedulerStepUnit(),
                    resolvedOptions.schedulerMonitorMetric(),
                    resolvedOptions.crossEntropyClassWeights(),
                    resolvedOptions.focalGamma(),
                    resolvedOptions.focalAlpha(),
                    resolvedOptions.focalClassWeights(),
                    resolvedOptions.bcePositiveWeights());
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta) {
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    null,
                    false,
                    true,
                    1,
                    "auto");
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                int gradientAccumulationSteps) {
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    null,
                    false,
                    true,
                    gradientAccumulationSteps,
                    "auto");
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError) {
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    checkpointDir,
                    resumeFromCheckpoint,
                    failOnCheckpointLoadError,
                    1,
                    "auto");
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                int gradientAccumulationSteps) {
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    checkpointDir,
                    resumeFromCheckpoint,
                    failOnCheckpointLoadError,
                    gradientAccumulationSteps,
                    "auto");
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                int gradientAccumulationSteps,
                String deviceId) {
            return fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    null,
                    CanonicalTrainer.BestModelMonitorMode.MIN,
                    checkpointDir,
                    resumeFromCheckpoint,
                    failOnCheckpointLoadError,
                    true,
                    false,
                    null,
                    CanonicalTrainer.BestModelMonitorMode.MIN,
                    gradientAccumulationSteps,
                    deviceId,
                    List.of(),
                    null,
                    CanonicalTrainer.SchedulerStepUnit.BATCH,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null);
        }

        private static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                String earlyStoppingMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                boolean saveBestModelCheckpoint,
                boolean restoreBestModelAtEnd,
                String bestModelMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode,
                int gradientAccumulationSteps,
                String deviceId,
                List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories,
                SchedulerFactory schedulerFactory,
                CanonicalTrainer.SchedulerStepUnit schedulerStepUnit,
                String schedulerMonitorMetric,
                float[] crossEntropyClassWeights,
                Float focalGamma,
                Float focalAlpha,
                float[] focalClassWeights,
                float[] bcePositiveWeights) {
            Objects.requireNonNull(model, "model must not be null");
            Objects.requireNonNull(trainLoader, "trainLoader must not be null");
            Objects.requireNonNull(preset, "preset must not be null");

            float lr = learningRate > 0 ? learningRate : 1e-3f;
            Optimizer optimizer = presetOptimizer(model, lr, preset);
            CanonicalTrainer.LossFunction loss = presetLoss(
                    preset,
                    crossEntropyClassWeights,
                    focalGamma,
                    focalAlpha,
                    focalClassWeights,
                    bcePositiveWeights);
            LRScheduler scheduler = schedulerFactory == null
                    ? null
                    : Objects.requireNonNull(schedulerFactory.create(optimizer), "scheduler factory returned null");

            CanonicalTrainer.Builder trainerBuilder = trainer()
                    .model(model)
                    .optimizer(optimizer)
                    .loss(loss)
                    .epochs(epochs)
                    .gradientClip(gradientClip)
                    .gradientAccumulationSteps(gradientAccumulationSteps)
                    .earlyStopping(earlyStoppingPatience, earlyStoppingMinDelta)
                    .earlyStoppingMonitorMetric(earlyStoppingMonitorMetric, earlyStoppingMonitorMode)
                    .resumeFromCheckpoint(resumeFromCheckpoint)
                    .failOnCheckpointLoadError(failOnCheckpointLoadError)
                    .saveBestModelCheckpoint(saveBestModelCheckpoint)
                    .restoreBestModelAtEnd(restoreBestModelAtEnd)
                    .schedulerMonitorMetric(schedulerMonitorMetric)
                    .device(deviceId);
            if (bestModelMonitorMetric == null) {
                trainerBuilder.bestModelMonitorValidationLoss(bestModelMonitorMode);
            } else {
                trainerBuilder.bestModelMonitorMetric(bestModelMonitorMetric, bestModelMonitorMode);
            }
            trainerBuilder.metrics(metricFactories);
            if (scheduler != null) {
                trainerBuilder.scheduler(
                        scheduler,
                        schedulerStepUnit == null ? CanonicalTrainer.SchedulerStepUnit.BATCH : schedulerStepUnit);
            }
            if (checkpointDir != null) {
                trainerBuilder.checkpointDir(checkpointDir);
            }

            try (CanonicalTrainer trainer = trainerBuilder.build()) {
                trainer.fit(trainLoader, validationLoader);
                return trainer.summary();
            }
        }

        public record TrainingOptions(
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                String earlyStoppingMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                boolean saveBestModelCheckpoint,
                boolean restoreBestModelAtEnd,
                String bestModelMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode,
                int gradientAccumulationSteps,
                String device,
                List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories,
                SchedulerFactory schedulerFactory,
                CanonicalTrainer.SchedulerStepUnit schedulerStepUnit,
                String schedulerMonitorMetric,
                float[] crossEntropyClassWeights,
                Float focalGamma,
                Float focalAlpha,
                float[] focalClassWeights,
                float[] bcePositiveWeights) {

            public TrainingOptions {
                device = device == null || device.isBlank() ? "auto" : device.trim();
                metricFactories = List.copyOf(metricFactories == null ? List.of() : metricFactories);
                schedulerStepUnit = schedulerStepUnit == null
                        ? CanonicalTrainer.SchedulerStepUnit.BATCH
                        : schedulerStepUnit;
                schedulerMonitorMetric = normalizeBestModelMonitorMetric(schedulerMonitorMetric);
                earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(earlyStoppingMonitorMetric);
                earlyStoppingMonitorMode = earlyStoppingMonitorMode == null
                        ? (earlyStoppingMonitorMetric == null
                                ? CanonicalTrainer.BestModelMonitorMode.MIN
                                : CanonicalTrainer.BestModelMonitorMode.MAX)
                        : earlyStoppingMonitorMode;
                bestModelMonitorMetric = normalizeBestModelMonitorMetric(bestModelMonitorMetric);
                bestModelMonitorMode = bestModelMonitorMode == null
                        ? (bestModelMonitorMetric == null
                                ? CanonicalTrainer.BestModelMonitorMode.MIN
                                : CanonicalTrainer.BestModelMonitorMode.MAX)
                        : bestModelMonitorMode;
                crossEntropyClassWeights = normalizeCrossEntropyClassWeights(crossEntropyClassWeights);
                focalGamma = normalizeFocalGamma(focalGamma);
                focalAlpha = normalizeFocalAlpha(focalAlpha);
                focalClassWeights = normalizeFocalClassWeights(focalClassWeights);
                bcePositiveWeights = normalizeBcePositiveWeights(bcePositiveWeights);
            }

            public TrainingOptions(
                    double gradientClip,
                    int earlyStoppingPatience,
                    double earlyStoppingMinDelta,
                    Path checkpointDir,
                    boolean resumeFromCheckpoint,
                    boolean failOnCheckpointLoadError,
                    int gradientAccumulationSteps,
                    String device) {
                this(
                        gradientClip,
                        earlyStoppingPatience,
                        earlyStoppingMinDelta,
                        null,
                        CanonicalTrainer.BestModelMonitorMode.MIN,
                        checkpointDir,
                        resumeFromCheckpoint,
                        failOnCheckpointLoadError,
                        true,
                        false,
                        null,
                        CanonicalTrainer.BestModelMonitorMode.MIN,
                        gradientAccumulationSteps,
                        device,
                        List.of(),
                        null,
                        CanonicalTrainer.SchedulerStepUnit.BATCH,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null);
            }

            public static Builder builder() {
                return new Builder();
            }

            public static final class Builder {
                private double gradientClip = 1.0;
                private int earlyStoppingPatience = 0;
                private double earlyStoppingMinDelta = 0.0;
                private String earlyStoppingMonitorMetric;
                private CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode =
                        CanonicalTrainer.BestModelMonitorMode.MIN;
                private Path checkpointDir;
                private boolean resumeFromCheckpoint = false;
                private boolean failOnCheckpointLoadError = true;
                private boolean saveBestModelCheckpoint = true;
                private boolean restoreBestModelAtEnd = false;
                private String bestModelMonitorMetric;
                private CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode =
                        CanonicalTrainer.BestModelMonitorMode.MIN;
                private int gradientAccumulationSteps = 1;
                private String device = "auto";
                private final List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories =
                        new ArrayList<>();
                private SchedulerFactory schedulerFactory;
                private CanonicalTrainer.SchedulerStepUnit schedulerStepUnit =
                        CanonicalTrainer.SchedulerStepUnit.BATCH;
                private String schedulerMonitorMetric;
                private float[] crossEntropyClassWeights;
                private Float focalGamma;
                private Float focalAlpha;
                private float[] focalClassWeights;
                private float[] bcePositiveWeights;

                private Builder() {
                }

                public Builder gradientClip(double gradientClip) {
                    this.gradientClip = Math.max(0.0, gradientClip);
                    return this;
                }

                public Builder earlyStopping(int patience) {
                    return earlyStopping(patience, earlyStoppingMinDelta);
                }

                public Builder earlyStopping(int patience, double minDelta) {
                    this.earlyStoppingPatience = Math.max(0, patience);
                    this.earlyStoppingMinDelta = Math.max(0.0, minDelta);
                    return this;
                }

                public Builder earlyStoppingMonitorMetric(String metricName) {
                    return earlyStoppingMonitorMetric(metricName, CanonicalTrainer.BestModelMonitorMode.MAX);
                }

                public Builder earlyStoppingMonitorMetric(
                        String metricName,
                        CanonicalTrainer.BestModelMonitorMode mode) {
                    this.earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(metricName);
                    this.earlyStoppingMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MAX : mode;
                    return this;
                }

                public Builder earlyStoppingMonitorValidationLoss() {
                    this.earlyStoppingMonitorMetric = null;
                    this.earlyStoppingMonitorMode = CanonicalTrainer.BestModelMonitorMode.MIN;
                    return this;
                }

                public Builder checkpointDir(Path checkpointDir) {
                    this.checkpointDir = checkpointDir;
                    return this;
                }

                public Builder resumeFromCheckpoint() {
                    return resumeFromCheckpoint(true);
                }

                public Builder resumeFromCheckpoint(boolean resumeFromCheckpoint) {
                    this.resumeFromCheckpoint = resumeFromCheckpoint;
                    return this;
                }

                public Builder failOnCheckpointLoadError(boolean failOnCheckpointLoadError) {
                    this.failOnCheckpointLoadError = failOnCheckpointLoadError;
                    return this;
                }

                public Builder saveBestModelCheckpoint(boolean saveBestModelCheckpoint) {
                    this.saveBestModelCheckpoint = saveBestModelCheckpoint;
                    return this;
                }

                public Builder bestModelCheckpoint(boolean saveBestModelCheckpoint) {
                    return saveBestModelCheckpoint(saveBestModelCheckpoint);
                }

                public Builder restoreBestModelAtEnd() {
                    return restoreBestModelAtEnd(true);
                }

                public Builder restoreBestModelAtEnd(boolean restoreBestModelAtEnd) {
                    this.restoreBestModelAtEnd = restoreBestModelAtEnd;
                    if (restoreBestModelAtEnd) {
                        this.saveBestModelCheckpoint = true;
                    }
                    return this;
                }

                public Builder bestModelMonitorMetric(String metricName) {
                    return bestModelMonitorMetric(metricName, CanonicalTrainer.BestModelMonitorMode.MAX);
                }

                public Builder bestModelMonitorMetric(
                        String metricName,
                        CanonicalTrainer.BestModelMonitorMode mode) {
                    this.bestModelMonitorMetric = normalizeBestModelMonitorMetric(metricName);
                    this.bestModelMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MAX : mode;
                    return this;
                }

                public Builder bestModelMonitorValidationLoss() {
                    return bestModelMonitorValidationLoss(CanonicalTrainer.BestModelMonitorMode.MIN);
                }

                public Builder bestModelMonitorValidationLoss(CanonicalTrainer.BestModelMonitorMode mode) {
                    this.bestModelMonitorMetric = null;
                    this.bestModelMonitorMode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MIN : mode;
                    return this;
                }

                public Builder gradientAccumulationSteps(int gradientAccumulationSteps) {
                    this.gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
                    return this;
                }

                public Builder device(String device) {
                    this.device = device == null || device.isBlank() ? "auto" : device.trim();
                    return this;
                }

                public Builder accelerator(String device) {
                    return device(device);
                }

                public Builder metric(java.util.function.Supplier<? extends CanonicalTrainer.Metric> metricFactory) {
                    this.metricFactories.add(Objects.requireNonNull(metricFactory, "metric factory must not be null"));
                    return this;
                }

                public Builder metrics(
                        List<? extends java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metrics) {
                    if (metrics == null) {
                        return this;
                    }
                    for (java.util.function.Supplier<? extends CanonicalTrainer.Metric> metric : metrics) {
                        metric(metric);
                    }
                    return this;
                }

                public Builder accuracyMetric() {
                    return metric(Gollek.DL.accuracyMetric());
                }

                public Builder confusionMatrixMetric() {
                    return metric(Gollek.DL.confusionMatrixMetric());
                }

                public Builder classificationConfusionMatrixMetric() {
                    return metric(Gollek.DL.classificationConfusionMatrixMetric());
                }

                public Builder topKAccuracyMetric(int k) {
                    return metric(Gollek.DL.topKAccuracyMetric(k));
                }

                public Builder binaryAccuracyMetric() {
                    return metric(Gollek.DL.binaryAccuracyMetric());
                }

                public Builder binaryAccuracyMetric(float logitThreshold) {
                    return metric(Gollek.DL.binaryAccuracyMetric(logitThreshold));
                }

                public Builder binaryConfusionMatrixMetric() {
                    return metric(Gollek.DL.binaryConfusionMatrixMetric());
                }

                public Builder binaryConfusionMatrixMetric(float logitThreshold) {
                    return metric(Gollek.DL.binaryConfusionMatrixMetric(logitThreshold));
                }

                public Builder binaryPrecisionMetric() {
                    return metric(Gollek.DL.binaryPrecisionMetric());
                }

                public Builder binaryPrecisionMetric(float logitThreshold) {
                    return metric(Gollek.DL.binaryPrecisionMetric(logitThreshold));
                }

                public Builder binaryRecallMetric() {
                    return metric(Gollek.DL.binaryRecallMetric());
                }

                public Builder binaryRecallMetric(float logitThreshold) {
                    return metric(Gollek.DL.binaryRecallMetric(logitThreshold));
                }

                public Builder binaryF1Metric() {
                    return metric(Gollek.DL.binaryF1Metric());
                }

                public Builder binaryF1Metric(float logitThreshold) {
                    return metric(Gollek.DL.binaryF1Metric(logitThreshold));
                }

                public Builder binaryRocAucMetric() {
                    return metric(Gollek.DL.binaryRocAucMetric());
                }

                public Builder binaryAurocMetric() {
                    return metric(Gollek.DL.binaryAurocMetric());
                }

                public Builder binaryAveragePrecisionMetric() {
                    return metric(Gollek.DL.binaryAveragePrecisionMetric());
                }

                public Builder binaryRankingMetrics() {
                    return binaryRocAucMetric()
                            .binaryAveragePrecisionMetric();
                }

                public Builder multiLabelExactMatchMetric() {
                    return metric(Gollek.DL.multiLabelExactMatchMetric());
                }

                public Builder multiLabelExactMatchMetric(float logitThreshold) {
                    return metric(Gollek.DL.multiLabelExactMatchMetric(logitThreshold));
                }

                public Builder multiLabelHammingLossMetric() {
                    return metric(Gollek.DL.multiLabelHammingLossMetric());
                }

                public Builder multiLabelHammingLossMetric(float logitThreshold) {
                    return metric(Gollek.DL.multiLabelHammingLossMetric(logitThreshold));
                }

                public Builder multiLabelMacroPrecisionMetric() {
                    return metric(Gollek.DL.multiLabelMacroPrecisionMetric());
                }

                public Builder multiLabelMacroPrecisionMetric(float logitThreshold) {
                    return metric(Gollek.DL.multiLabelMacroPrecisionMetric(logitThreshold));
                }

                public Builder multiLabelMacroRecallMetric() {
                    return metric(Gollek.DL.multiLabelMacroRecallMetric());
                }

                public Builder multiLabelMacroRecallMetric(float logitThreshold) {
                    return metric(Gollek.DL.multiLabelMacroRecallMetric(logitThreshold));
                }

                public Builder multiLabelMacroF1Metric() {
                    return metric(Gollek.DL.multiLabelMacroF1Metric());
                }

                public Builder multiLabelMacroF1Metric(float logitThreshold) {
                    return metric(Gollek.DL.multiLabelMacroF1Metric(logitThreshold));
                }

                public Builder multiLabelMacroRocAucMetric() {
                    return metric(Gollek.DL.multiLabelMacroRocAucMetric());
                }

                public Builder multiLabelMacroAurocMetric() {
                    return metric(Gollek.DL.multiLabelMacroAurocMetric());
                }

                public Builder multiLabelMacroAveragePrecisionMetric() {
                    return metric(Gollek.DL.multiLabelMacroAveragePrecisionMetric());
                }

                public Builder multiLabelRankingMetrics() {
                    return multiLabelMacroRocAucMetric()
                            .multiLabelMacroAveragePrecisionMetric();
                }

                public Builder multiLabelBinaryMetrics() {
                    return multiLabelExactMatchMetric()
                            .multiLabelHammingLossMetric()
                            .multiLabelMacroPrecisionMetric()
                            .multiLabelMacroRecallMetric()
                            .multiLabelMacroF1Metric();
                }

                public Builder multiLabelBinaryMetrics(float logitThreshold) {
                    return multiLabelExactMatchMetric(logitThreshold)
                            .multiLabelHammingLossMetric(logitThreshold)
                            .multiLabelMacroPrecisionMetric(logitThreshold)
                            .multiLabelMacroRecallMetric(logitThreshold)
                            .multiLabelMacroF1Metric(logitThreshold);
                }

                public Builder multiLabelMetrics() {
                    return multiLabelBinaryMetrics();
                }

                public Builder multiLabelMetrics(float logitThreshold) {
                    return multiLabelBinaryMetrics(logitThreshold);
                }

                public Builder binaryClassificationMetrics() {
                    return binaryAccuracyMetric()
                            .binaryPrecisionMetric()
                            .binaryRecallMetric()
                            .binaryF1Metric();
                }

                public Builder binaryClassificationMetrics(float logitThreshold) {
                    return binaryAccuracyMetric(logitThreshold)
                            .binaryPrecisionMetric(logitThreshold)
                            .binaryRecallMetric(logitThreshold)
                            .binaryF1Metric(logitThreshold);
                }

                public Builder binaryMetrics() {
                    return binaryClassificationMetrics();
                }

                public Builder binaryMetrics(float logitThreshold) {
                    return binaryClassificationMetrics(logitThreshold);
                }

                public Builder crossEntropyClassWeights(float... classWeights) {
                    this.crossEntropyClassWeights = normalizeCrossEntropyClassWeights(classWeights);
                    return this;
                }

                public Builder classWeights(int... labels) {
                    this.crossEntropyClassWeights = DataLoader.classWeights(labels);
                    return this;
                }

                public Builder classWeightsFor(int numClasses, int... labels) {
                    this.crossEntropyClassWeights = DataLoader.classWeightsFor(numClasses, labels);
                    return this;
                }

                public Builder focalGamma(float gamma) {
                    this.focalGamma = normalizeFocalGamma(gamma);
                    return this;
                }

                public Builder focalAlpha(float alpha) {
                    this.focalAlpha = normalizeFocalAlpha(alpha);
                    return this;
                }

                public Builder focal(float gamma, float alpha) {
                    return focalGamma(gamma).focalAlpha(alpha);
                }

                public Builder focalClassWeights(float... classWeights) {
                    this.focalClassWeights = normalizeFocalClassWeights(classWeights);
                    return this;
                }

                public Builder focalClassWeights(int... labels) {
                    this.focalClassWeights = DataLoader.classWeights(labels);
                    return this;
                }

                public Builder focalClassWeightsFor(int numClasses, int... labels) {
                    this.focalClassWeights = DataLoader.classWeightsFor(numClasses, labels);
                    return this;
                }

                public Builder bcePositiveWeight(float positiveWeight) {
                    this.bcePositiveWeights = normalizeBcePositiveWeights(new float[] { positiveWeight });
                    return this;
                }

                public Builder bcePositiveWeights(float... positiveWeights) {
                    this.bcePositiveWeights = normalizeBcePositiveWeights(positiveWeights);
                    return this;
                }

                public Builder binaryPositiveWeight(int... labels) {
                    return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
                }

                public Builder binaryPositiveWeight(boolean... labels) {
                    return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
                }

                public Builder binaryPositiveWeight(float... labels) {
                    return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
                }

                public Builder multiLabelPositiveWeights(int[][] labels) {
                    return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
                }

                public Builder multiLabelPositiveWeights(boolean[][] labels) {
                    return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
                }

                public Builder multiLabelPositiveWeights(float[][] labels) {
                    return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
                }

                public Builder precisionMetric() {
                    return metric(Gollek.DL.precisionMetric());
                }

                public Builder recallMetric() {
                    return metric(Gollek.DL.recallMetric());
                }

                public Builder f1Metric() {
                    return metric(Gollek.DL.f1Metric());
                }

                public Builder macroF1Metric() {
                    return metric(Gollek.DL.macroF1Metric());
                }

                public Builder classificationMetrics() {
                    return accuracyMetric()
                            .precisionMetric()
                            .recallMetric()
                            .f1Metric();
                }

                public Builder classificationMacroRocAucMetric() {
                    return metric(Gollek.DL.classificationMacroRocAucMetric());
                }

                public Builder classificationMacroAurocMetric() {
                    return metric(Gollek.DL.classificationMacroAurocMetric());
                }

                public Builder classificationMacroAveragePrecisionMetric() {
                    return metric(Gollek.DL.classificationMacroAveragePrecisionMetric());
                }

                public Builder classificationRankingMetrics() {
                    return classificationMacroRocAucMetric()
                            .classificationMacroAveragePrecisionMetric();
                }

                public Builder meanAbsoluteErrorMetric() {
                    return metric(Gollek.DL.meanAbsoluteErrorMetric());
                }

                public Builder maeMetric() {
                    return meanAbsoluteErrorMetric();
                }

                public Builder meanSquaredErrorMetric() {
                    return metric(Gollek.DL.meanSquaredErrorMetric());
                }

                public Builder mseMetric() {
                    return meanSquaredErrorMetric();
                }

                public Builder rootMeanSquaredErrorMetric() {
                    return metric(Gollek.DL.rootMeanSquaredErrorMetric());
                }

                public Builder rmseMetric() {
                    return rootMeanSquaredErrorMetric();
                }

                public Builder r2ScoreMetric() {
                    return metric(Gollek.DL.r2ScoreMetric());
                }

                public Builder r2Metric() {
                    return r2ScoreMetric();
                }

                public Builder regressionMetrics() {
                    return meanAbsoluteErrorMetric()
                            .meanSquaredErrorMetric()
                            .rootMeanSquaredErrorMetric()
                            .r2ScoreMetric();
                }

                public Builder scheduler(SchedulerFactory schedulerFactory) {
                    return scheduler(schedulerFactory, CanonicalTrainer.SchedulerStepUnit.BATCH);
                }

                public Builder scheduler(
                        SchedulerFactory schedulerFactory,
                        CanonicalTrainer.SchedulerStepUnit stepUnit) {
                    this.schedulerFactory = Objects.requireNonNull(
                            schedulerFactory, "schedulerFactory must not be null");
                    this.schedulerStepUnit = stepUnit == null
                            ? CanonicalTrainer.SchedulerStepUnit.BATCH
                            : stepUnit;
                    return this;
                }

                public Builder schedulerMonitorMetric(String metricName) {
                    this.schedulerMonitorMetric = normalizeBestModelMonitorMetric(metricName);
                    return this;
                }

                public Builder schedulerMonitorValidationLoss() {
                    this.schedulerMonitorMetric = null;
                    return this;
                }

                public Builder stepLr(int stepSize, float gamma) {
                    return stepLrBatches(stepSize, gamma);
                }

                public Builder stepLrBatches(int stepSize, float gamma) {
                    return scheduler(
                            optimizer -> new StepLR(optimizer, Math.max(1, stepSize), gamma),
                            CanonicalTrainer.SchedulerStepUnit.BATCH);
                }

                public Builder stepLrEpochs(int stepSize, float gamma) {
                    return scheduler(
                            optimizer -> new StepLR(optimizer, Math.max(1, stepSize), gamma),
                            CanonicalTrainer.SchedulerStepUnit.EPOCH);
                }

                public Builder cosineAnnealingLr(int tMax, float minLr) {
                    return cosineAnnealingLrBatches(tMax, minLr);
                }

                public Builder cosineAnnealingLrBatches(int tMax, float minLr) {
                    return scheduler(
                            optimizer -> new CosineAnnealingLR(
                                    optimizer,
                                    Math.max(1, tMax),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.BATCH);
                }

                public Builder cosineAnnealingLrEpochs(int tMax, float minLr) {
                    return scheduler(
                            optimizer -> new CosineAnnealingLR(
                                    optimizer,
                                    Math.max(1, tMax),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.EPOCH);
                }

                public Builder warmupCosineLr(int warmupSteps, int totalSteps, float minLr) {
                    return warmupCosineLrBatches(warmupSteps, totalSteps, minLr);
                }

                public Builder warmupCosineLr(int warmupSteps, int totalSteps, float maxLr, float minLr) {
                    return warmupCosineLrBatches(warmupSteps, totalSteps, maxLr, minLr);
                }

                public Builder warmupCosineLrBatches(int warmupSteps, int totalSteps, float minLr) {
                    return scheduler(
                            optimizer -> new WarmupCosineScheduler(
                                    optimizer,
                                    Math.max(0, warmupSteps),
                                    Math.max(1, totalSteps),
                                    Math.max(1.0e-12f, optimizer.learningRate()),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.BATCH);
                }

                public Builder warmupCosineLrBatches(int warmupSteps, int totalSteps, float maxLr, float minLr) {
                    return scheduler(
                            optimizer -> new WarmupCosineScheduler(
                                    optimizer,
                                    Math.max(0, warmupSteps),
                                    Math.max(1, totalSteps),
                                    Math.max(1.0e-12f, maxLr),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.BATCH);
                }

                public Builder warmupCosineLrEpochs(int warmupSteps, int totalSteps, float minLr) {
                    return scheduler(
                            optimizer -> new WarmupCosineScheduler(
                                    optimizer,
                                    Math.max(0, warmupSteps),
                                    Math.max(1, totalSteps),
                                    Math.max(1.0e-12f, optimizer.learningRate()),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.EPOCH);
                }

                public Builder warmupCosineLrEpochs(int warmupSteps, int totalSteps, float maxLr, float minLr) {
                    return scheduler(
                            optimizer -> new WarmupCosineScheduler(
                                    optimizer,
                                    Math.max(0, warmupSteps),
                                    Math.max(1, totalSteps),
                                    Math.max(1.0e-12f, maxLr),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.EPOCH);
                }

                public Builder reduceLrOnPlateauValidationLoss(float factor, int patience, float minLr) {
                    return reduceLrOnPlateauValidationLoss(factor, patience, 0.0, 0, minLr);
                }

                public Builder reduceLrOnPlateauValidationLoss(
                        float factor,
                        int patience,
                        double threshold,
                        int cooldown,
                        float minLr) {
                    schedulerMonitorValidationLoss();
                    return scheduler(
                            optimizer -> new ReduceLROnPlateau(
                                    optimizer,
                                    ReduceLROnPlateau.Mode.MIN,
                                    factor,
                                    Math.max(0, patience),
                                    Math.max(0.0, threshold),
                                    Math.max(0, cooldown),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.VALIDATION);
                }

                public Builder reduceLrOnPlateauMetric(
                        String metricName,
                        ReduceLROnPlateau.Mode mode,
                        float factor,
                        int patience,
                        double threshold,
                        int cooldown,
                        float minLr) {
                    schedulerMonitorMetric(metricName);
                    return scheduler(
                            optimizer -> new ReduceLROnPlateau(
                                    optimizer,
                                    mode,
                                    factor,
                                    Math.max(0, patience),
                                    Math.max(0.0, threshold),
                                    Math.max(0, cooldown),
                                    Math.max(0.0f, minLr)),
                            CanonicalTrainer.SchedulerStepUnit.VALIDATION);
                }

                public TrainingOptions build() {
                    return new TrainingOptions(
                            gradientClip,
                            earlyStoppingPatience,
                            earlyStoppingMinDelta,
                            earlyStoppingMonitorMetric,
                            earlyStoppingMonitorMode,
                            checkpointDir,
                            resumeFromCheckpoint,
                            failOnCheckpointLoadError,
                            saveBestModelCheckpoint,
                            restoreBestModelAtEnd,
                            bestModelMonitorMetric,
                            bestModelMonitorMode,
                            gradientAccumulationSteps,
                            device,
                            metricFactories,
                            schedulerFactory,
                            schedulerStepUnit,
                            schedulerMonitorMetric,
                            crossEntropyClassWeights,
                            focalGamma,
                            focalAlpha,
                            focalClassWeights,
                            bcePositiveWeights);
                }
            }
        }

        @FunctionalInterface
        public interface SchedulerFactory {
            LRScheduler create(Optimizer optimizer);
        }

        public record EvaluationSummary(
                double loss,
                int batchCount,
                long sampleCount,
                Map<String, Double> metrics,
                Map<String, Object> metricDetails,
                Map<String, Object> metadata) {

            public EvaluationSummary(
                    double loss,
                    int batchCount,
                    long sampleCount,
                    Map<String, Double> metrics,
                    Map<String, Object> metadata) {
                this(loss, batchCount, sampleCount, metrics, Map.of(), metadata);
            }

            public EvaluationSummary {
                metrics = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metrics == null ? Map.of() : metrics));
                metricDetails = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metricDetails == null ? Map.of() : metricDetails));
                metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metadata == null ? Map.of() : metadata));
            }

            public double metric(String name) {
                return metrics.getOrDefault(name, Double.NaN);
            }
        }

        private static Optimizer presetOptimizer(NNModule model, float learningRate, TrainingPreset preset) {
            return switch (preset) {
                case REGRESSION_MSE_ADAMW,
                        REGRESSION_HUBER_ADAMW,
                        CLASSIFICATION_CROSS_ENTROPY_ADAMW,
                        CLASSIFICATION_FOCAL_ADAMW,
                        BINARY_FOCAL_WITH_LOGITS_ADAMW,
                        BINARY_BCE_WITH_LOGITS_ADAMW ->
                    AdamW.builder(model.parameters(), learningRate).build();
                case REGRESSION_MSE_SGD,
                        REGRESSION_HUBER_SGD,
                        CLASSIFICATION_CROSS_ENTROPY_SGD,
                        CLASSIFICATION_FOCAL_SGD,
                        BINARY_FOCAL_WITH_LOGITS_SGD,
                        BINARY_BCE_WITH_LOGITS_SGD ->
                    SGD.builder(model.parameters(), learningRate).momentum(0.9f).build();
            };
        }

        private static CanonicalTrainer.LossFunction presetLoss(TrainingPreset preset) {
            return presetLoss(preset, null, null, null, null, null);
        }

        private static CanonicalTrainer.LossFunction presetLoss(
                TrainingPreset preset,
                float[] crossEntropyClassWeights,
                Float focalGamma,
                Float focalAlpha,
                float[] focalClassWeights,
                float[] bcePositiveWeights) {
            return switch (preset) {
                case REGRESSION_MSE_ADAMW, REGRESSION_MSE_SGD -> {
                    MSELoss mse = new MSELoss();
                    yield mse::compute;
                }
                case REGRESSION_HUBER_ADAMW, REGRESSION_HUBER_SGD -> {
                    HuberLoss huber = new HuberLoss();
                    yield huber::compute;
                }
                case CLASSIFICATION_CROSS_ENTROPY_ADAMW, CLASSIFICATION_CROSS_ENTROPY_SGD -> {
                    CrossEntropyLoss crossEntropy = crossEntropyClassWeights == null
                            ? new CrossEntropyLoss()
                            : new CrossEntropyLoss(crossEntropyClassWeights);
                    yield crossEntropy::compute;
                }
                case CLASSIFICATION_FOCAL_ADAMW, CLASSIFICATION_FOCAL_SGD -> {
                    float gamma = focalGamma == null ? 2.0f : focalGamma;
                    FocalLoss focal = focalClassWeights != null
                            ? new FocalLoss(gamma, focalClassWeights)
                            : new FocalLoss(gamma, focalAlpha == null ? 0.25f : focalAlpha);
                    yield focal::compute;
                }
                case BINARY_FOCAL_WITH_LOGITS_ADAMW, BINARY_FOCAL_WITH_LOGITS_SGD -> {
                    float gamma = focalGamma == null ? 2.0f : focalGamma;
                    float alpha = focalAlpha == null ? 0.25f : focalAlpha;
                    BinaryFocalWithLogitsLoss focal = bcePositiveWeights == null
                            ? new BinaryFocalWithLogitsLoss(gamma, alpha)
                            : new BinaryFocalWithLogitsLoss(gamma, alpha, bcePositiveWeights);
                    yield focal::compute;
                }
                case BINARY_BCE_WITH_LOGITS_ADAMW, BINARY_BCE_WITH_LOGITS_SGD -> {
                    BCEWithLogitsLoss bce = bcePositiveWeights == null
                            ? new BCEWithLogitsLoss()
                            : new BCEWithLogitsLoss(bcePositiveWeights);
                    yield bce::compute;
                }
            };
        }

        private static Float normalizeFocalGamma(Float gamma) {
            if (gamma == null) {
                return null;
            }
            if (!Float.isFinite(gamma) || gamma < 0.0f) {
                throw new IllegalArgumentException("focalGamma must be finite and non-negative, got: " + gamma);
            }
            return gamma;
        }

        private static String normalizeBestModelMonitorMetric(String metricName) {
            if (metricName == null || metricName.isBlank()) {
                return null;
            }
            String normalized = metricName.trim();
            if (normalized.startsWith("validationMetric.")) {
                normalized = normalized.substring("validationMetric.".length());
            }
            if (normalized.isBlank()) {
                throw new IllegalArgumentException("bestModelMonitorMetric must not be blank");
            }
            return normalized;
        }

        private static Float normalizeFocalAlpha(Float alpha) {
            if (alpha == null) {
                return null;
            }
            if (!Float.isFinite(alpha) || alpha <= 0.0f) {
                throw new IllegalArgumentException("focalAlpha must be finite and positive, got: " + alpha);
            }
            return alpha;
        }

        private static float[] normalizeFocalClassWeights(float[] weights) {
            if (weights == null) {
                return null;
            }
            if (weights.length == 0) {
                throw new IllegalArgumentException("focalClassWeights must contain at least one value");
            }
            float[] copy = weights.clone();
            for (float weight : copy) {
                if (!Float.isFinite(weight) || weight <= 0.0f) {
                    throw new IllegalArgumentException(
                            "focalClassWeights must be finite and positive, got: " + weight);
                }
            }
            return copy;
        }

        private static float[] normalizeCrossEntropyClassWeights(float[] weights) {
            if (weights == null) {
                return null;
            }
            if (weights.length == 0) {
                throw new IllegalArgumentException("crossEntropyClassWeights must contain at least one value");
            }
            float[] copy = weights.clone();
            for (float weight : copy) {
                if (!Float.isFinite(weight) || weight <= 0.0f) {
                    throw new IllegalArgumentException(
                            "crossEntropyClassWeights must be finite and positive, got: " + weight);
                }
            }
            return copy;
        }

        private static float[] normalizeBcePositiveWeights(float[] weights) {
            if (weights == null) {
                return null;
            }
            if (weights.length == 0) {
                throw new IllegalArgumentException("bcePositiveWeights must contain at least one value");
            }
            float[] copy = weights.clone();
            for (float weight : copy) {
                if (!Float.isFinite(weight) || weight <= 0.0f) {
                    throw new IllegalArgumentException(
                            "bcePositiveWeights must be finite and positive, got: " + weight);
                }
            }
            return copy;
        }

        private static List<CanonicalTrainer.Metric> instantiateEvaluationMetrics(
                List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
            if (metricFactories == null || metricFactories.isEmpty()) {
                return List.of();
            }
            List<CanonicalTrainer.Metric> metrics = new ArrayList<>(metricFactories.size());
            for (java.util.function.Supplier<? extends CanonicalTrainer.Metric> factory : metricFactories) {
                CanonicalTrainer.Metric metric = Objects.requireNonNull(factory, "metric factory must not be null")
                        .get();
                Objects.requireNonNull(metric, "metric factory returned null");
                requireEvaluationMetricName(metric.name());
                metric.reset();
                metrics.add(metric);
            }
            return metrics;
        }

        private static Map<String, Object> evaluationMetricDetails(List<CanonicalTrainer.Metric> metrics) {
            if (metrics == null || metrics.isEmpty()) {
                return Map.of();
            }
            Map<String, Object> details = new LinkedHashMap<>();
            for (CanonicalTrainer.Metric metric : metrics) {
                if (metric instanceof CanonicalTrainer.DetailedMetric detailedMetric) {
                    Map<String, Object> metricDetails = detailedMetric.details();
                    if (metricDetails != null && !metricDetails.isEmpty()) {
                        details.put(requireEvaluationMetricName(metric.name()), metricDetails);
                    }
                }
            }
            return details.isEmpty() ? Map.of() : Collections.unmodifiableMap(details);
        }

        private static String requireEvaluationMetricName(String name) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("metric name must not be blank");
            }
            return name.trim();
        }

        private static double requireFiniteEvaluationLoss(double loss) {
            if (!Double.isFinite(loss)) {
                throw new IllegalArgumentException("evaluation loss must be finite, got " + loss);
            }
            return loss;
        }

        private static long batchSampleCount(Batch batch) {
            long[] shape = batch.labels().shape();
            return shape.length == 0 ? 1L : Math.max(1L, shape[0]);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Traditional ML (Scikit-Learn Style)
    // ══════════════════════════════════════════════════════════════════════

    public static class ML {
        // Classification
        public static RandomForestClassifier randomForest() {
            return new RandomForestClassifier();
        }

        public static GradientBoostingClassifier gradientBoosting() {
            return new GradientBoostingClassifier();
        }

        public static SVC svm() {
            return new SVC();
        }

        public static GaussianNB naiveBayes() {
            return new GaussianNB();
        }

        public static VotingClassifier voting(List<BaseEstimator> estimators, String voting) {
            return new VotingClassifier(estimators, voting, null);
        }

        // Regression
        public static LinearModel linearRegression() {
            return new LinearModel("none", 0, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel ridge(double alpha) {
            return new LinearModel("l2", alpha, 0, 1e-4, 1000, 0.01);
        }

        public static LinearModel lasso(double alpha) {
            return new LinearModel("l1", alpha, 1, 1e-4, 1000, 0.01);
        }

        // Clustering
        public static KMeans kMeans(int nClusters) {
            return new KMeans(nClusters);
        }

        public static DBSCAN dbscan(double eps, int minSamples) {
            return new DBSCAN(eps, minSamples, "euclidean", -1);
        }

        // Preprocessing
        public static StandardScaler standardScaler() {
            return new StandardScaler();
        }

        public static PCA pca(int nComponents) {
            return new PCA(nComponents);
        }

        public static PolynomialFeatures polynomialFeatures(int degree) {
            return new PolynomialFeatures(degree);
        }

        public static Pipeline pipeline(BaseEstimator... steps) {
            if (steps.length == 0)
                throw new IllegalArgumentException("Pipeline must have at least one step");
            List<BaseTransformer> transformers = new ArrayList<>();
            for (int i = 0; i < steps.length - 1; i++) {
                if (!(steps[i] instanceof BaseTransformer)) {
                    throw new IllegalArgumentException("All steps except the last must be transformers");
                }
                transformers.add((BaseTransformer) steps[i]);
            }
            return new Pipeline(transformers, steps[steps.length - 1]);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Selection & Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static class Selection {
        public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y, int nFolds) {
            return CrossValidation.crossValScore(estimator, X, y, nFolds, "accuracy");
        }

        public static CrossValidation.GridSearchResult gridSearch(BaseEstimator estimator,
                Map<String, Object[]> paramGrid,
                float[][] X, int[] y) {
            return CrossValidation.gridSearch(estimator, paramGrid, X, y, 5);
        }

        public static ModelSelection.TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize,
                int randomState) {
            return ModelSelection.trainTestSplit(X, y, testSize, randomState);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Hub
    // ══════════════════════════════════════════════════════════════════════

    public static class Hub {
        public static HubConfig.Builder config() {
            return HubConfig.builder();
        }

        public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
            return ModelHub.loadWeights(modelId);
        }

        public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) throws IOException {
            return ModelHub.loadWeights(modelId, config);
        }

        public static void loadInto(NNModule model, String modelId) throws IOException {
            ModelHub.loadInto(model, modelId);
        }

        public static void loadInto(NNModule model, String modelId, HubConfig config) throws IOException {
            ModelHub.loadInto(model, modelId, config);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Export
    // ══════════════════════════════════════════════════════════════════════

    public static class Export {
        public static ModelExporter.Builder model(NNModule model) {
            return ModelExporter.builder().model(model);
        }

        public static Benchmark benchmark(Object model) {
            return new Benchmark(model);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation — delegates to GradTensor
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    public static GradTensor uniform(double lo, double hi, long... shape) {
        return GradTensor.uniform(lo, hi, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Operations (static)
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor cat(GradTensor... tensors) {
        return GradTensor.cat(tensors);
    }

    public static GradTensor cat(int dim, GradTensor... tensors) {
        return GradTensor.cat(dim, tensors);
    }

    public static GradTensor stack(GradTensor... tensors) {
        return GradTensor.stack(tensors);
    }

    public static GradTensor stack(int dim, GradTensor... tensors) {
        return GradTensor.stack(dim, tensors);
    }

    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        return GradTensor.where(condition, x, y);
    }

    public static GradTensor einsum(String equation, GradTensor a, GradTensor b) {
        return a.einsum(equation, b);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    private static GollekSdk _sdk;

    private static synchronized GollekSdk sdk() {
        if (_sdk == null) {
            _sdk = GollekSdk.builder().build();
        }
        return _sdk;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Facade
    // ══════════════════════════════════════════════════════════════════════

    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, sdk());
    }

    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, sdk());
    }

    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, sdk());
    }

    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, sdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    public static DeviceType defaultDevice() {
        if (isCudaAvailable())
            return DeviceType.CUDA;
        if (isMetalAvailable())
            return DeviceType.METAL;
        return DeviceType.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    public static void printInfo() {
        var device = defaultDevice();
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework                         ║");
        System.out.println("║          Version " + String.format("%-32s", VERSION) + "║");
        System.out.println("╠══════════════════════════════════════════════════════╣");
        System.out.println("║  Device:  " + String.format("%-32s", device) + "║");
        System.out.println("║  CUDA:    " + String.format("%-32s", isCudaAvailable()) + "║");
        System.out.println("║  Metal:   " + String.format("%-32s", isMetalAvailable()) + "║");
        System.out.println("║  Java:    " + String.format("%-32s", System.getProperty("java.version")) + "║");
        System.out.println("║  Vector:  "
                + String.format("%-32s", jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() + "-bit")
                + "║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
