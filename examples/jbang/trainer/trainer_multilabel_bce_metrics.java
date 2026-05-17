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

public class trainer_multilabel_bce_metrics {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Multi-Label BCE Trainer Metrics (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        Linear model = new Linear(3, 3);
        var split = Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        -2.0f, -2.0f, -2.0f,
                         2.0f, -2.0f, -2.0f,
                        -2.0f,  2.0f, -2.0f,
                        -2.0f, -2.0f,  2.0f,
                         2.0f,  2.0f, -2.0f,
                         2.0f, -2.0f,  2.0f,
                        -2.0f,  2.0f,  2.0f,
                         2.0f,  2.0f,  2.0f,
                        -1.4f, -1.6f, -1.2f,
                         1.5f, -1.7f, -1.1f,
                        -1.8f,  1.3f, -1.5f,
                        -1.2f, -1.4f,  1.7f,
                         1.8f,  1.1f, -1.6f,
                         1.2f, -1.3f,  1.4f,
                        -1.5f,  1.6f,  1.2f,
                         1.3f,  1.5f,  1.7f
                }, 16, 3),
                new int[][] {
                        {0, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 1, 1},
                        {0, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 1, 1}
                },
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
                60,
                0.05f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .binaryClassificationMetrics()
                        .multiLabelBinaryMetrics()
                        .multiLabelRankingMetrics()
                        .cosineAnnealingLrEpochs(60, 0.005f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW,
                requestedDevice,
                Gollek.DL.binaryAccuracyMetric(),
                Gollek.DL.binaryPrecisionMetric(),
                Gollek.DL.binaryRecallMetric(),
                Gollek.DL.binaryF1Metric(),
                Gollek.DL.multiLabelExactMatchMetric(),
                Gollek.DL.multiLabelHammingLossMetric(),
                Gollek.DL.multiLabelMacroPrecisionMetric(),
                Gollek.DL.multiLabelMacroRecallMetric(),
                Gollek.DL.multiLabelMacroF1Metric(),
                Gollek.DL.multiLabelMacroRocAucMetric(),
                Gollek.DL.multiLabelMacroAveragePrecisionMetric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalMicroAccuracy=%.6f evalMicroPrecision=%.6f evalMicroRecall=%.6f evalMicroF1=%.6f%n",
                evaluation.loss(),
                evaluation.metric("binary_accuracy"),
                evaluation.metric("binary_precision"),
                evaluation.metric("binary_recall"),
                evaluation.metric("binary_f1"));
        System.out.println("trainMicroF1=" + metadata.get("trainMetric.binary_f1"));
        System.out.println("validationMicroF1=" + metadata.get("validationMetric.binary_f1"));
        System.out.println("validationExactMatch=" + metadata.get("validationMetric.multilabel_exact_match"));
        System.out.println("validationHammingLoss=" + metadata.get("validationMetric.multilabel_hamming_loss"));
        System.out.println("validationMacroF1=" + metadata.get("validationMetric.multilabel_macro_f1"));
        System.out.println("validationMacroRocAuc=" + metadata.get("validationMetric.multilabel_macro_roc_auc"));
        System.out.println("validationMacroAveragePrecision="
                + metadata.get("validationMetric.multilabel_macro_average_precision"));
        System.out.println("evalExactMatch=" + evaluation.metric("multilabel_exact_match"));
        System.out.println("evalHammingLoss=" + evaluation.metric("multilabel_hamming_loss"));
        System.out.println("evalMacroPrecision=" + evaluation.metric("multilabel_macro_precision"));
        System.out.println("evalMacroRecall=" + evaluation.metric("multilabel_macro_recall"));
        System.out.println("evalMacroF1=" + evaluation.metric("multilabel_macro_f1"));
        System.out.println("evalMacroRocAuc=" + evaluation.metric("multilabel_macro_roc_auc"));
        System.out.println("evalMacroAveragePrecision="
                + evaluation.metric("multilabel_macro_average_precision"));
        System.out.println("schedulerSteps=" + metadata.get("learningRateSchedulerStepCount"));
        System.out.println("finalLearningRate=" + metadata.get("learningRate"));
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("requestedDeviceAvailable=" + metadata.get("requestedDeviceAvailable"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
        printProbeTags(model);
    }

    private static void printProbeTags(Linear model) {
        GradTensor logits = model.forward(GradTensor.of(new float[] {
                -1.8f, -1.6f,  1.5f,
                 1.4f,  1.6f, -1.7f,
                 1.2f,  1.3f,  1.4f
        }, 3, 3));
        float[] values = logits.data();
        System.out.println("----------------------------------------------------");
        for (int row = 0; row < 3; row++) {
            System.out.printf("probe[%d] predictedTags=[%d,%d,%d] probabilities=[%.4f,%.4f,%.4f]%n",
                    row,
                    values[row * 3] >= 0.0f ? 1 : 0,
                    values[row * 3 + 1] >= 0.0f ? 1 : 0,
                    values[row * 3 + 2] >= 0.0f ? 1 : 0,
                    sigmoid(values[row * 3]),
                    sigmoid(values[row * 3 + 1]),
                    sigmoid(values[row * 3 + 2]));
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
