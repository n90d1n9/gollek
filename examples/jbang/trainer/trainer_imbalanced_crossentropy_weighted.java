///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_imbalanced_crossentropy_weighted {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Imbalanced CrossEntropy Trainer (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        int[] labels = new int[] {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1,
                2, 2, 2
        };
        float[] classWeights = Gollek.DL.classWeightsFor(3, labels);

        Linear model = new Linear(2, 3);
        var split = Gollek.DL.classificationStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        -2.0f, -1.5f,
                        -1.7f, -1.3f,
                        -2.2f, -0.8f,
                        -1.4f, -1.8f,
                        -1.1f, -1.2f,
                        -2.3f, -1.0f,
                        -1.5f, -0.7f,
                        -0.9f, -1.6f,
                        -1.8f, -0.6f,
                        -1.2f, -1.9f,
                         1.2f,  1.0f,
                         1.6f,  0.7f,
                         0.9f,  1.4f,
                         1.0f, -1.2f,
                         1.6f, -0.8f,
                         0.8f, -1.6f
                }, 16, 2),
                labels,
                0.75,
                2026L);

        TensorDataLoader train = split.trainLoader(4, true, 2026L);
        TensorDataLoader validation = split.validationLoader(4);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                4,
                true,
                2026L,
                80,
                0.06f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_ADAMW,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .crossEntropyClassWeights(classWeights)
                        .classificationMetrics()
                        .topKAccuracyMetric(2)
                        .cosineAnnealingLrEpochs(80, 0.005f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.crossEntropy(classWeights)::compute,
                requestedDevice,
                Gollek.DL.accuracyMetric(),
                Gollek.DL.topKAccuracyMetric(2),
                Gollek.DL.precisionMetric(),
                Gollek.DL.recallMetric(),
                Gollek.DL.f1Metric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.println("classWeights=" + Arrays.toString(classWeights));
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalAccuracy=%.6f evalTop2=%.6f evalF1=%.6f%n",
                evaluation.loss(),
                evaluation.metric("accuracy"),
                evaluation.metric("top2_accuracy"),
                evaluation.metric("f1"));
        System.out.println("trainAccuracy=" + metadata.get("trainMetric.accuracy"));
        System.out.println("trainF1=" + metadata.get("trainMetric.f1"));
        System.out.println("validationAccuracy=" + metadata.get("validationMetric.accuracy"));
        System.out.println("validationF1=" + metadata.get("validationMetric.f1"));
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
        printProbeScores(model);
    }

    private static void printProbeScores(Linear model) {
        GradTensor logits = model.forward(GradTensor.of(new float[] {
                -1.8f, -1.1f,
                 1.4f,  1.0f,
                 1.2f, -1.3f
        }, 3, 2));
        float[] values = logits.data();
        System.out.println("----------------------------------------------------");
        for (int row = 0; row < 3; row++) {
            int offset = row * 3;
            int predicted = argmax(values, offset, 3);
            System.out.printf("probe[%d] logits=[%.6f, %.6f, %.6f] predictedClass=%d%n",
                    row,
                    values[offset],
                    values[offset + 1],
                    values[offset + 2],
                    predicted);
        }
    }

    private static int argmax(float[] values, int offset, int width) {
        int best = 0;
        for (int i = 1; i < width; i++) {
            if (values[offset + i] > values[offset + best]) {
                best = i;
            }
        }
        return best;
    }

    private static String requestedDevice(String[] args) {
        if (args.length == 0) {
            return "auto";
        }
        for (int i = 0; i < args.length; i++) {
            if ("--device".equals(args[i]) && i + 1 < args.length) {
                return args[i + 1];
            }
        }
        return args[0];
    }
}
