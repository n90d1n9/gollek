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

public class trainer_binary_focal_weighted {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);
        float gamma = 2.0f;
        float alpha = 0.75f;

        System.out.println("====================================================");
        System.out.println(" Gollek Binary Focal Weighted Trainer (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        int[] labels = new int[] {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1
        };
        float positiveWeight = Gollek.DL.binaryPositiveWeight(labels);

        Linear model = new Linear(2, 1);
        var split = Gollek.DL.binaryStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        -2.2f, -1.4f,
                        -1.8f, -1.7f,
                        -1.5f, -0.9f,
                        -1.2f, -2.1f,
                        -2.4f, -0.7f,
                        -0.9f, -1.8f,
                        -1.6f, -1.1f,
                        -1.1f, -1.5f,
                        -2.0f, -0.8f,
                        -0.8f, -2.0f,
                        -1.7f, -0.6f,
                        -0.6f, -1.4f,
                         1.0f,  1.4f,
                         1.5f,  0.9f,
                         0.9f,  1.8f,
                         2.1f,  1.1f
                }, 16, 2),
                labels,
                0.75,
                20260516L);

        TensorDataLoader train = split.trainLoader(4, true, 20260516L);
        TensorDataLoader validation = split.validationLoader(4);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                4,
                true,
                20260516L,
                80,
                0.04f,
                Gollek.DL.TrainingPreset.BINARY_FOCAL_WITH_LOGITS_ADAMW,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .focal(gamma, alpha)
                        .bcePositiveWeight(positiveWeight)
                        .binaryClassificationMetrics()
                        .binaryRankingMetrics()
                        .cosineAnnealingLrEpochs(80, 0.004f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.binaryFocalWithLogitsLoss(gamma, alpha, positiveWeight)::compute,
                requestedDevice,
                Gollek.DL.binaryAccuracyMetric(),
                Gollek.DL.binaryPrecisionMetric(),
                Gollek.DL.binaryRecallMetric(),
                Gollek.DL.binaryF1Metric(),
                Gollek.DL.binaryRocAucMetric(),
                Gollek.DL.binaryAveragePrecisionMetric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("gamma=%.3f alpha=%.3f positiveWeight=%.6f%n", gamma, alpha, positiveWeight);
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalAccuracy=%.6f evalPrecision=%.6f evalRecall=%.6f evalF1=%.6f%n",
                evaluation.loss(),
                evaluation.metric("binary_accuracy"),
                evaluation.metric("binary_precision"),
                evaluation.metric("binary_recall"),
                evaluation.metric("binary_f1"));
        System.out.printf("evalRocAuc=%.6f evalAveragePrecision=%.6f%n",
                evaluation.metric("binary_roc_auc"),
                evaluation.metric("binary_average_precision"));
        System.out.println("trainBinaryF1=" + metadata.get("trainMetric.binary_f1"));
        System.out.println("validationBinaryF1=" + metadata.get("validationMetric.binary_f1"));
        System.out.println("trainRocAuc=" + metadata.get("trainMetric.binary_roc_auc"));
        System.out.println("validationRocAuc=" + metadata.get("validationMetric.binary_roc_auc"));
        System.out.println("trainAveragePrecision=" + metadata.get("trainMetric.binary_average_precision"));
        System.out.println("validationAveragePrecision=" + metadata.get("validationMetric.binary_average_precision"));
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
                -1.8f, -1.5f,
                -0.7f, -1.1f,
                 0.9f,  1.4f,
                 1.9f,  1.0f
        }, 4, 2));
        float[] values = logits.data();
        System.out.println("----------------------------------------------------");
        for (int i = 0; i < values.length; i++) {
            double probability = sigmoid(values[i]);
            int predicted = values[i] >= 0.0f ? 1 : 0;
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
