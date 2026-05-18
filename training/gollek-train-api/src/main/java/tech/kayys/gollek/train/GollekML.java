package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.svm.*;
import tech.kayys.gollek.ml.naive_bayes.*;
import tech.kayys.gollek.ml.linear_model.*;
import tech.kayys.gollek.ml.clustering.*;
import tech.kayys.gollek.ml.pipeline.*;
import tech.kayys.gollek.ml.feature.PolynomialFeatures;
import tech.kayys.gollek.ml.model_selection.*;
import tech.kayys.gollek.ml.util.*;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * @deprecated Use {@link Gollek} as the unified ML entry point.
 *             {@code Gollek.ML} mirrors the scikit-learn factory API and
 *             {@code Gollek.DL} mirrors the PyTorch/autograd API.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class GollekML {

    // ==================== Deep Learning ====================
    public static class DL {
        public static GradTensor tensor(float[] data, long... shape) {
            return Gollek.tensor(data, shape);
        }

        public static NNModule sequential(NNModule... layers) {
            return Gollek.DL.sequential(layers);
        }

        public static Optimizer adamW(List<Parameter> params, float lr) {
            return Gollek.DL.adamW(params, lr);
        }

        public static tech.kayys.gollek.ml.optim.WarmupCosineScheduler warmupCosineScheduler(
                Optimizer optimizer,
                int warmupSteps,
                int totalSteps,
                float maxLr,
                float minLr) {
            return Gollek.DL.warmupCosineScheduler(optimizer, warmupSteps, totalSteps, maxLr, minLr);
        }

        public static tech.kayys.gollek.ml.optim.ReduceLROnPlateau reduceLrOnPlateauScheduler(
                Optimizer optimizer,
                tech.kayys.gollek.ml.optim.ReduceLROnPlateau.Mode mode,
                float factor,
                int patience,
                double threshold,
                int cooldown,
                float minLr) {
            return Gollek.DL.reduceLrOnPlateauScheduler(
                    optimizer, mode, factor, patience, threshold, cooldown, minLr);
        }

        public static CrossEntropyLoss crossEntropy() {
            return Gollek.DL.crossEntropy();
        }

        public static CrossEntropyLoss crossEntropy(float[] classWeights) {
            return Gollek.DL.crossEntropy(classWeights);
        }

        public static FocalLoss focalLoss() {
            return Gollek.DL.focalLoss();
        }

        public static FocalLoss focalLoss(float gamma) {
            return Gollek.DL.focalLoss(gamma);
        }

        public static FocalLoss focalLoss(float gamma, float alpha) {
            return Gollek.DL.focalLoss(gamma, alpha);
        }

        public static FocalLoss focalLoss(float gamma, float[] classWeights) {
            return Gollek.DL.focalLoss(gamma, classWeights);
        }

        public static MSELoss mseLoss() {
            return Gollek.DL.mseLoss();
        }

        public static L1Loss l1Loss() {
            return Gollek.DL.l1Loss();
        }

        public static HuberLoss huberLoss() {
            return Gollek.DL.huberLoss();
        }

        public static HuberLoss huberLoss(float delta) {
            return Gollek.DL.huberLoss(delta);
        }

        public static SmoothL1Loss smoothL1Loss() {
            return Gollek.DL.smoothL1Loss();
        }

        public static SmoothL1Loss smoothL1Loss(float beta) {
            return Gollek.DL.smoothL1Loss(beta);
        }

        public static BCEWithLogitsLoss bceWithLogitsLoss() {
            return Gollek.DL.bceWithLogitsLoss();
        }

        public static BCEWithLogitsLoss bceWithLogitsLoss(float positiveWeight) {
            return Gollek.DL.bceWithLogitsLoss(positiveWeight);
        }

        public static BCEWithLogitsLoss bceWithLogitsLoss(float[] positiveWeights) {
            return Gollek.DL.bceWithLogitsLoss(positiveWeights);
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits() {
            return Gollek.DL.binaryCrossEntropyWithLogits();
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float positiveWeight) {
            return Gollek.DL.binaryCrossEntropyWithLogits(positiveWeight);
        }

        public static BCEWithLogitsLoss binaryCrossEntropyWithLogits(float[] positiveWeights) {
            return Gollek.DL.binaryCrossEntropyWithLogits(positiveWeights);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss() {
            return Gollek.DL.binaryFocalWithLogitsLoss();
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(float gamma, float alpha) {
            return Gollek.DL.binaryFocalWithLogitsLoss(gamma, alpha);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
                float gamma,
                float alpha,
                float positiveWeight) {
            return Gollek.DL.binaryFocalWithLogitsLoss(gamma, alpha, positiveWeight);
        }

        public static BinaryFocalWithLogitsLoss binaryFocalWithLogitsLoss(
                float gamma,
                float alpha,
                float[] positiveWeights) {
            return Gollek.DL.binaryFocalWithLogitsLoss(gamma, alpha, positiveWeights);
        }

        public static CanonicalTrainer.Builder trainer() {
            return Gollek.DL.trainer();
        }

        public static tech.kayys.gollek.train.diffusion.opd.DefaultDiffusionOpdTrainer.Builder diffusionOpdTrainer() {
            return Gollek.DL.diffusionOpdTrainer();
        }

        public static Gollek.DL.TrainingOptions.Builder trainingOptions() {
            return Gollek.DL.trainingOptions();
        }

        public static Acceleration.BackendStatus accelerationStatus() {
            return Gollek.DL.accelerationStatus();
        }

        public static Acceleration.BackendStatus accelerationStatus(String deviceId) {
            return Gollek.DL.accelerationStatus(deviceId);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> accuracyMetric() {
            return Gollek.DL.accuracyMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> confusionMatrixMetric() {
            return Gollek.DL.confusionMatrixMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationConfusionMatrixMetric() {
            return Gollek.DL.classificationConfusionMatrixMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> topKAccuracyMetric(int k) {
            return Gollek.DL.topKAccuracyMetric(k);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric() {
            return Gollek.DL.binaryAccuracyMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric(float logitThreshold) {
            return Gollek.DL.binaryAccuracyMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric() {
            return Gollek.DL.binaryConfusionMatrixMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric(
                float logitThreshold) {
            return Gollek.DL.binaryConfusionMatrixMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric() {
            return Gollek.DL.binaryPrecisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric(float logitThreshold) {
            return Gollek.DL.binaryPrecisionMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRecallMetric() {
            return Gollek.DL.binaryRecallMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRecallMetric(float logitThreshold) {
            return Gollek.DL.binaryRecallMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryF1Metric() {
            return Gollek.DL.binaryF1Metric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryF1Metric(float logitThreshold) {
            return Gollek.DL.binaryF1Metric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryRocAucMetric() {
            return Gollek.DL.binaryRocAucMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAurocMetric() {
            return Gollek.DL.binaryAurocMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> binaryAveragePrecisionMetric() {
            return Gollek.DL.binaryAveragePrecisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric() {
            return Gollek.DL.multiLabelExactMatchMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric(
                float logitThreshold) {
            return Gollek.DL.multiLabelExactMatchMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric() {
            return Gollek.DL.multiLabelHammingLossMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric(
                float logitThreshold) {
            return Gollek.DL.multiLabelHammingLossMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric() {
            return Gollek.DL.multiLabelMacroPrecisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric(
                float logitThreshold) {
            return Gollek.DL.multiLabelMacroPrecisionMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric() {
            return Gollek.DL.multiLabelMacroRecallMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric(
                float logitThreshold) {
            return Gollek.DL.multiLabelMacroRecallMetric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric() {
            return Gollek.DL.multiLabelMacroF1Metric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric(
                float logitThreshold) {
            return Gollek.DL.multiLabelMacroF1Metric(logitThreshold);
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroRocAucMetric() {
            return Gollek.DL.multiLabelMacroRocAucMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroAurocMetric() {
            return Gollek.DL.multiLabelMacroAurocMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> multiLabelMacroAveragePrecisionMetric() {
            return Gollek.DL.multiLabelMacroAveragePrecisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> precisionMetric() {
            return Gollek.DL.precisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> recallMetric() {
            return Gollek.DL.recallMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> f1Metric() {
            return Gollek.DL.f1Metric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> macroF1Metric() {
            return Gollek.DL.macroF1Metric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroRocAucMetric() {
            return Gollek.DL.classificationMacroRocAucMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroAurocMetric() {
            return Gollek.DL.classificationMacroAurocMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> classificationMacroAveragePrecisionMetric() {
            return Gollek.DL.classificationMacroAveragePrecisionMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> meanAbsoluteErrorMetric() {
            return Gollek.DL.meanAbsoluteErrorMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> maeMetric() {
            return Gollek.DL.maeMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> meanSquaredErrorMetric() {
            return Gollek.DL.meanSquaredErrorMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> mseMetric() {
            return Gollek.DL.mseMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> rootMeanSquaredErrorMetric() {
            return Gollek.DL.rootMeanSquaredErrorMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> rmseMetric() {
            return Gollek.DL.rmseMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> r2ScoreMetric() {
            return Gollek.DL.r2ScoreMetric();
        }

        public static java.util.function.Supplier<CanonicalTrainer.Metric> r2Metric() {
            return Gollek.DL.r2Metric();
        }

        public static DataLoader.TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
            return Gollek.DL.tensorDataset(inputs, labels);
        }

        public static GradTensor classLabels(int... labels) {
            return Gollek.DL.classLabels(labels);
        }

        public static float[] classWeights(int... labels) {
            return Gollek.DL.classWeights(labels);
        }

        public static float[] classWeightsFor(int numClasses, int... labels) {
            return Gollek.DL.classWeightsFor(numClasses, labels);
        }

        public static GradTensor binaryLabels(int... labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(boolean... labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(float... labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(int[][] labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(boolean[][] labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor binaryLabels(float[][] labels) {
            return Gollek.DL.binaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(int[][] labels) {
            return Gollek.DL.multiLabelBinaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(boolean[][] labels) {
            return Gollek.DL.multiLabelBinaryLabels(labels);
        }

        public static GradTensor multiLabelBinaryLabels(float[][] labels) {
            return Gollek.DL.multiLabelBinaryLabels(labels);
        }

        public static float binaryPositiveWeight(int... labels) {
            return Gollek.DL.binaryPositiveWeight(labels);
        }

        public static float binaryPositiveWeight(boolean... labels) {
            return Gollek.DL.binaryPositiveWeight(labels);
        }

        public static float binaryPositiveWeight(float... labels) {
            return Gollek.DL.binaryPositiveWeight(labels);
        }

        public static float[] multiLabelPositiveWeights(int[][] labels) {
            return Gollek.DL.multiLabelPositiveWeights(labels);
        }

        public static float[] multiLabelPositiveWeights(boolean[][] labels) {
            return Gollek.DL.multiLabelPositiveWeights(labels);
        }

        public static float[] multiLabelPositiveWeights(float[][] labels) {
            return Gollek.DL.multiLabelPositiveWeights(labels);
        }

        public static DataLoader.TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
            return Gollek.DL.classificationDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
            return Gollek.DL.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
            return Gollek.DL.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
            return Gollek.DL.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
            return Gollek.DL.binaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, int[][] labels) {
            return Gollek.DL.multiLabelBinaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, boolean[][] labels) {
            return Gollek.DL.multiLabelBinaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, float[][] labels) {
            return Gollek.DL.multiLabelBinaryDataset(inputs, labels);
        }

        public static DataLoader.TensorDatasetSplit split(
                GradTensor inputs,
                GradTensor labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.split(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.classificationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationStratifiedSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.classificationStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryStratifiedSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binarySplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinarySplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit split(
                DataLoader.TensorDataset dataset,
                double trainFraction,
                long seed) {
            return Gollek.DL.split(dataset, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit trainValidationSplit(
                GradTensor inputs,
                GradTensor labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.trainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.classificationTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit classificationStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.classificationStratifiedTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryStratifiedTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                int[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                boolean[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor inputs,
                float[][] labels,
                double trainFraction,
                long seed) {
            return Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(inputs, labels, trainFraction, seed);
        }

        public static DataLoader.TensorDatasetSplit trainValidationSplit(
                DataLoader.TensorDataset dataset,
                double trainFraction,
                long seed) {
            return Gollek.DL.trainValidationSplit(dataset, trainFraction, seed);
        }

        public static DataLoader.TensorBuilder dataLoader(DataLoader.TensorDatasetAdapter dataset) {
            return Gollek.DL.dataLoader(dataset);
        }

        public static DataLoader.TensorDataLoader dataLoader(GradTensor inputs, GradTensor labels, int batchSize) {
            return Gollek.DL.dataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader classificationDataLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return Gollek.DL.classificationDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return Gollek.DL.binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return Gollek.DL.binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return Gollek.DL.binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryDataLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return Gollek.DL.binaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryDataLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader classificationLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return Gollek.DL.classificationLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                int[] labels,
                int batchSize) {
            return Gollek.DL.binaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return Gollek.DL.binaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return Gollek.DL.binaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader binaryLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return Gollek.DL.binaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                int[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                boolean[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryLoader(inputs, labels, batchSize);
        }

        public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
                GradTensor inputs,
                float[][] labels,
                int batchSize) {
            return Gollek.DL.multiLabelBinaryLoader(inputs, labels, batchSize);
        }

        @SafeVarargs
        public static final Gollek.DL.EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                Gollek.DL.TrainingPreset preset,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return Gollek.DL.evaluate(model, loader, preset, metrics);
        }

        @SafeVarargs
        public static final Gollek.DL.EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                Gollek.DL.TrainingPreset preset,
                String deviceId,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return Gollek.DL.evaluate(model, loader, preset, deviceId, metrics);
        }

        @SafeVarargs
        public static final Gollek.DL.EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                TrainingLossFunction loss,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return Gollek.DL.evaluate(model, loader, loss, metrics);
        }

        @SafeVarargs
        public static final Gollek.DL.EvaluationSummary evaluate(
                NNModule model,
                Iterable<Batch> loader,
                TrainingLossFunction loss,
                String deviceId,
                java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
            return Gollek.DL.evaluate(model, loader, loss, deviceId, metrics);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset) {
            return Gollek.DL.fit(model, trainLoader, epochs, learningRate, preset);
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset) {
            return Gollek.DL.fit(model, split, batchSize, epochs, learningRate, preset);
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                Gollek.DL.TrainingOptions options) {
            return Gollek.DL.fit(model, split, batchSize, epochs, learningRate, preset, options);
        }

        public static TrainingSummary fit(
                NNModule model,
                DataLoader.TensorDatasetSplit split,
                int batchSize,
                boolean shuffleTraining,
                long shuffleSeed,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                Gollek.DL.TrainingOptions options) {
            return Gollek.DL.fit(
                    model,
                    split,
                    batchSize,
                    shuffleTraining,
                    shuffleSeed,
                    epochs,
                    learningRate,
                    preset,
                    options);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                String deviceId) {
            return Gollek.DL.fit(model, trainLoader, epochs, learningRate, preset, deviceId);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                double gradientClip) {
            return Gollek.DL.fit(model, trainLoader, validationLoader, epochs, learningRate, preset, gradientClip);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                Gollek.DL.TrainingOptions options) {
            return Gollek.DL.fit(model, trainLoader, validationLoader, epochs, learningRate, preset, options);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta) {
            return Gollek.DL.fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                int gradientAccumulationSteps) {
            return Gollek.DL.fit(
                    model,
                    trainLoader,
                    validationLoader,
                    epochs,
                    learningRate,
                    preset,
                    gradientClip,
                    earlyStoppingPatience,
                    earlyStoppingMinDelta,
                    gradientAccumulationSteps);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                java.nio.file.Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError) {
            return Gollek.DL.fit(
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
                    failOnCheckpointLoadError);
        }

        public static TrainingSummary fit(
                NNModule model,
                Iterable<Batch> trainLoader,
                Iterable<Batch> validationLoader,
                int epochs,
                float learningRate,
                Gollek.DL.TrainingPreset preset,
                double gradientClip,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                java.nio.file.Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                int gradientAccumulationSteps) {
            return Gollek.DL.fit(
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
                    gradientAccumulationSteps);
        }
    }

    // ==================== Traditional ML ====================
    public static class ML {
        // Classification
        public static RandomForestClassifier randomForest() {
            return Gollek.ML.randomForest();
        }

        public static GradientBoostingClassifier gradientBoosting() {
            return Gollek.ML.gradientBoosting();
        }

        public static SVC svm() {
            return Gollek.ML.svm();
        }

        public static GaussianNB naiveBayes() {
            return Gollek.ML.naiveBayes();
        }

        // Regression
        public static LinearModel linearRegression() {
            return Gollek.ML.linearRegression();
        }

        public static LinearModel ridge(double alpha) {
            return Gollek.ML.ridge(alpha);
        }

        public static LinearModel lasso(double alpha) {
            return Gollek.ML.lasso(alpha);
        }

        // Clustering
        public static KMeans kMeans(int nClusters) {
            return Gollek.ML.kMeans(nClusters);
        }

        public static DBSCAN dbscan(double eps, int minSamples) {
            return Gollek.ML.dbscan(eps, minSamples);
        }

        // Preprocessing
        public static StandardScaler standardScaler() {
            return Gollek.ML.standardScaler();
        }

        public static PCA pca(int nComponents) {
            return Gollek.ML.pca(nComponents);
        }

        public static PolynomialFeatures polynomialFeatures(int degree) {
            return Gollek.ML.polynomialFeatures(degree);
        }
    }

    // ==================== Model Selection ====================
    public static class Selection {
        public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y, int nFolds) {
            return Gollek.Selection.crossValScore(estimator, X, y, nFolds);
        }

        public static CrossValidation.GridSearchResult gridSearch(BaseEstimator estimator,
                Map<String, Object[]> paramGrid,
                float[][] X, int[] y) {
            return Gollek.Selection.gridSearch(estimator, paramGrid, X, y);
        }

        public static ModelSelection.TrainTestSplit trainTestSplit(float[][] X, int[] y, double testSize, int randomState) {
            return Gollek.Selection.trainTestSplit(X, y, testSize, randomState);
        }
    }

    public static class Hub {
        public static tech.kayys.gollek.ml.hub.HubConfig.Builder config() {
            return Gollek.Hub.config();
        }

        public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
            return Gollek.Hub.loadWeights(modelId);
        }

        public static Map<String, GradTensor> loadWeights(String modelId, tech.kayys.gollek.ml.hub.HubConfig config)
                throws IOException {
            return Gollek.Hub.loadWeights(modelId, config);
        }

        public static void loadInto(NNModule model, String modelId) throws IOException {
            Gollek.Hub.loadInto(model, modelId);
        }

        public static void loadInto(NNModule model, String modelId, tech.kayys.gollek.ml.hub.HubConfig config)
                throws IOException {
            Gollek.Hub.loadInto(model, modelId, config);
        }
    }

    public static class Export {
        public static tech.kayys.gollek.ml.export.ModelExporter.Builder model(NNModule model) {
            return Gollek.Export.model(model);
        }

        public static tech.kayys.gollek.ml.export.Benchmark benchmark(Object model) {
            return Gollek.Export.benchmark(model);
        }
    }
}
