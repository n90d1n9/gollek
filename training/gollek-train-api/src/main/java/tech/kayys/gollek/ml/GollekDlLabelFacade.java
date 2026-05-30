package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader;

/**
 * Label tensor and imbalance-weight helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlLabelFacade extends GollekDlMetricsFacade {
    protected GollekDlLabelFacade() {
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

    public static float[] classBalancedSampleWeights(int... labels) {
        return DataLoader.classBalancedSampleWeights(labels);
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

    public static float[] binaryBalancedSampleWeights(int... labels) {
        return DataLoader.binaryBalancedSampleWeights(labels);
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
}
