package tech.kayys.gollek.ml.metrics;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Core evaluation metrics used by the Java ML framework.
 */
public final class Metrics {

    private Metrics() {
    }

    public static double accuracy(int[] yTrue, int[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }

    public static double precision(int[] yTrue, int[] yPred, int label) {
        int tp = 0;
        int fp = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == label) {
                if (yTrue[i] == label) {
                    tp++;
                } else {
                    fp++;
                }
            }
        }
        return tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
    }

    public static double recall(int[] yTrue, int[] yPred, int label) {
        int tp = 0;
        int fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == label) {
                if (yPred[i] == label) {
                    tp++;
                } else {
                    fn++;
                }
            }
        }
        return tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;
    }

    public static double f1Score(int[] yTrue, int[] yPred, int label) {
        double precision = precision(yTrue, yPred, label);
        double recall = recall(yTrue, yPred, label);
        return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
    }

    public static double f1Score(int[] yTrue, int[] yPred) {
        return f1Score(yTrue, yPred, 1);
    }

    public static double[] confusionMatrix(int[] yTrue, int[] yPred, int numClasses) {
        double[] cm = new double[numClasses * numClasses];
        for (int i = 0; i < yTrue.length; i++) {
            cm[yTrue[i] * numClasses + yPred[i]]++;
        }
        return cm;
    }

    public static double rocAuc(float[] scores, int[] labels) {
        List<Pair> pairs = new ArrayList<>();
        int posCount = 0;
        for (int i = 0; i < scores.length; i++) {
            pairs.add(new Pair(scores[i], labels[i]));
            if (labels[i] == 1) {
                posCount++;
            }
        }
        pairs.sort((a, b) -> Float.compare(b.score, a.score));

        double auc = 0;
        int tp = 0;
        int fp = 0;
        int negCount = labels.length - posCount;

        if (posCount == 0 || negCount == 0) {
            return 0.5;
        }

        for (Pair pair : pairs) {
            if (pair.label == 1) {
                tp++;
                auc += (double) fp / negCount;
            } else {
                fp++;
            }
        }

        return auc / posCount;
    }

    public static double mse(float[] yTrue, float[] yPred) {
        double sum = 0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            sum += diff * diff;
        }
        return sum / yTrue.length;
    }

    public static double rmse(float[] yTrue, float[] yPred) {
        return Math.sqrt(mse(yTrue, yPred));
    }

    public static double mae(float[] yTrue, float[] yPred) {
        double sum = 0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        return sum / yTrue.length;
    }

    public static double r2Score(float[] yTrue, float[] yPred) {
        double sum = 0;
        for (float value : yTrue) {
            sum += value;
        }
        double mean = sum / yTrue.length;

        double ssRes = 0;
        double ssTot = 0;
        for (int i = 0; i < yTrue.length; i++) {
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
            ssTot += Math.pow(yTrue[i] - mean, 2);
        }
        return ssTot > 0 ? 1 - ssRes / ssTot : 0.0;
    }

    public static double silhouetteScore(float[][] x, int[] labels) {
        int n = x.length;
        double total = 0;

        for (int i = 0; i < n; i++) {
            double a = meanDistanceToCluster(x[i], x, labels, labels[i], true);

            double b = Double.MAX_VALUE;
            Set<Integer> otherClusters = new HashSet<>();
            for (int label : labels) {
                otherClusters.add(label);
            }
            otherClusters.remove(labels[i]);

            if (otherClusters.isEmpty()) {
                continue;
            }

            for (int cluster : otherClusters) {
                double dist = meanDistanceToCluster(x[i], x, labels, cluster, false);
                b = Math.min(b, dist);
            }

            total += (b - a) / Math.max(a, b);
        }

        return total / n;
    }

    private static double meanDistanceToCluster(float[] point, float[][] x, int[] labels,
            int cluster, boolean sameCluster) {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < x.length; i++) {
            if (labels[i] == cluster) {
                if (sameCluster && x[i] == point) {
                    continue;
                }
                sum += euclideanDistance(point, x[i]);
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    }

    private static double euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private static final class Pair {
        final float score;
        final int label;

        Pair(float score, int label) {
            this.score = score;
            this.label = label;
        }
    }
}
