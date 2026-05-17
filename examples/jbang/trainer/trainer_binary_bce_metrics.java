///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_binary_bce_metrics {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Binary BCE Trainer Metrics (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        float decisionLogitThreshold = 0.5f;
        Linear model = new Linear(2, 1);
        var split = Gollek.DL.binaryStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        -2.0f, -2.0f,
                        -1.5f, -1.0f,
                        -2.0f, -0.8f,
                        -0.8f, -1.7f,
                        -1.0f, -1.0f,
                        -1.8f, -0.3f,
                        -0.5f, -1.5f,
                        -2.2f, -1.1f,
                         1.0f,  1.0f,
                         1.4f,  0.8f,
                         2.0f,  1.5f,
                         0.6f,  1.3f,
                         1.8f,  0.3f,
                         0.2f,  1.7f,
                         1.2f,  1.5f,
                         2.2f,  0.9f
                }, 16, 2),
                new int[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
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
                40,
                0.05f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .binaryClassificationMetrics(decisionLogitThreshold)
                        .binaryConfusionMatrixMetric(decisionLogitThreshold)
                        .cosineAnnealingLrEpochs(40, 0.005f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW,
                requestedDevice,
                Gollek.DL.binaryAccuracyMetric(decisionLogitThreshold),
                Gollek.DL.binaryConfusionMatrixMetric(decisionLogitThreshold),
                Gollek.DL.binaryPrecisionMetric(decisionLogitThreshold),
                Gollek.DL.binaryRecallMetric(decisionLogitThreshold),
                Gollek.DL.binaryF1Metric(decisionLogitThreshold));

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalAccuracy=%.6f evalConfusionAccuracy=%.6f evalPrecision=%.6f evalRecall=%.6f evalF1=%.6f%n",
                evaluation.loss(),
                evaluation.metric("binary_accuracy"),
                evaluation.metric("binary_confusion_matrix_accuracy"),
                evaluation.metric("binary_precision"),
                evaluation.metric("binary_recall"),
                evaluation.metric("binary_f1"));
        System.out.println("decisionLogitThreshold=" + decisionLogitThreshold);
        System.out.println("trainBinaryAccuracy=" + metadata.get("trainMetric.binary_accuracy"));
        System.out.println("trainBinaryConfusionAccuracy="
                + metadata.get("trainMetric.binary_confusion_matrix_accuracy"));
        System.out.println("trainBinaryPrecision=" + metadata.get("trainMetric.binary_precision"));
        System.out.println("trainBinaryRecall=" + metadata.get("trainMetric.binary_recall"));
        System.out.println("trainBinaryF1=" + metadata.get("trainMetric.binary_f1"));
        System.out.println("validationBinaryAccuracy=" + metadata.get("validationMetric.binary_accuracy"));
        System.out.println("validationBinaryConfusionAccuracy="
                + metadata.get("validationMetric.binary_confusion_matrix_accuracy"));
        System.out.println("validationBinaryConfusionMatrix="
                + metadata.get("validationMetricDetails.binary_confusion_matrix_accuracy"));
        System.out.println("validationBinaryF1=" + metadata.get("validationMetric.binary_f1"));
        System.out.println("schedulerSteps=" + metadata.get("learningRateSchedulerStepCount"));
        System.out.println("finalLearningRate=" + metadata.get("learningRate"));
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("requestedDeviceAvailable=" + metadata.get("requestedDeviceAvailable"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
        printProbeScores(model, decisionLogitThreshold);
    }

    private static void printProbeScores(Linear model, float decisionLogitThreshold) {
        GradTensor logits = model.forward(GradTensor.of(new float[] {
                -1.8f, -1.4f,
                -0.7f, -1.2f,
                 0.8f,  1.1f,
                 1.9f,  0.7f
        }, 4, 2));
        float[] values = logits.data();
        System.out.println("----------------------------------------------------");
        for (int i = 0; i < values.length; i++) {
            double probability = sigmoid(values[i]);
            int predicted = values[i] >= decisionLogitThreshold ? 1 : 0;
            System.out.printf("probe[%d] logit=%.6f probability=%.6f predicted=%d%n",
                    i,
                    values[i],
                    probability,
                    predicted);
        }
    }

    private static double sigmoid(float logit) {
        return 1.0 / (1.0 + Math.exp(-logit));
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
