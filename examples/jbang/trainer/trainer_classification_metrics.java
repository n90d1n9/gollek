///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.util.Locale;
import java.util.List;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_classification_metrics {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Classification Trainer Metrics (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        Linear model = new Linear(3, 3);
        var split = Gollek.DL.classificationStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        1f, 0f, 0f,
                        0.9f, 0.1f, 0f,
                        0f, 1f, 0f,
                        0.1f, 0.9f, 0f,
                        0f, 0f, 1f,
                        0f, 0.2f, 0.8f,
                        0.8f, 0f, 0.2f,
                        0.2f, 0.7f, 0.1f,
                        0.1f, 0.1f, 0.8f
                }, 9, 3),
                new int[] {0, 0, 1, 1, 2, 2, 0, 1, 2},
                0.67,
                2026L);

        TensorDataLoader train = split.trainLoader(3, true, 2026L);
        TensorDataLoader validation = split.validationLoader(3);
        Path checkpointDir = Path.of("trainer_checkpoints", "classification_metrics");
        Path reportFile = checkpointDir.resolve("canonical-report.json");

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                3,
                true,
                2026L,
                20,
                0.08f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .classificationMetrics()
                        .confusionMatrixMetric()
                        .classificationRankingMetrics()
                        .topKAccuracyMetric(2)
                        .checkpointDir(checkpointDir)
                        .bestModelMonitorMetric("f1", CanonicalTrainer.BestModelMonitorMode.MAX)
                        .earlyStopping(4, 0.0)
                        .earlyStoppingMonitorMetric("f1", CanonicalTrainer.BestModelMonitorMode.MAX)
                        .restoreBestModelAtEnd()
                        .stepLrEpochs(10, 0.5f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                requestedDevice,
                Gollek.DL.accuracyMetric(),
                Gollek.DL.topKAccuracyMetric(2),
                Gollek.DL.precisionMetric(),
                Gollek.DL.recallMetric(),
                Gollek.DL.f1Metric(),
                Gollek.DL.confusionMatrixMetric(),
                Gollek.DL.classificationMacroRocAucMetric(),
                Gollek.DL.classificationMacroAveragePrecisionMetric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf(
                "evalLoss=%.6f evalAccuracy=%.6f evalTop2=%.6f evalF1=%.6f evalConfusionAccuracy=%.6f evalMacroRocAuc=%.6f evalMacroAveragePrecision=%.6f%n",
                evaluation.loss(),
                evaluation.metric("accuracy"),
                evaluation.metric("top2_accuracy"),
                evaluation.metric("f1"),
                evaluation.metric("confusion_matrix_accuracy"),
                evaluation.metric("classification_macro_roc_auc"),
                evaluation.metric("classification_macro_average_precision"));
        System.out.println("trainAccuracy=" + metadata.get("trainMetric.accuracy"));
        System.out.println("trainPrecision=" + metadata.get("trainMetric.precision"));
        System.out.println("trainRecall=" + metadata.get("trainMetric.recall"));
        System.out.println("trainF1=" + metadata.get("trainMetric.f1"));
        System.out.println("trainConfusionAccuracy=" + metadata.get("trainMetric.confusion_matrix_accuracy"));
        System.out.println("trainTop2=" + metadata.get("trainMetric.top2_accuracy"));
        System.out.println("trainMacroRocAuc=" + metadata.get("trainMetric.classification_macro_roc_auc"));
        System.out.println("trainMacroAveragePrecision="
                + metadata.get("trainMetric.classification_macro_average_precision"));
        System.out.println("validationAccuracy=" + metadata.get("validationMetric.accuracy"));
        System.out.println("validationF1=" + metadata.get("validationMetric.f1"));
        System.out.println("validationConfusionAccuracy="
                + metadata.get("validationMetric.confusion_matrix_accuracy"));
        if (metadata.get("validationMetricDetails.confusion_matrix_accuracy") instanceof Map<?, ?> details) {
            System.out.println("validationConfusionMatrix=" + details.get("matrix"));
        }
        System.out.println("validationMacroRocAuc=" + metadata.get("validationMetric.classification_macro_roc_auc"));
        System.out.println("validationMacroAveragePrecision="
                + metadata.get("validationMetric.classification_macro_average_precision"));
        System.out.println("checkpointDir=" + checkpointDir);
        System.out.println("bestModelCheckpointMonitor=" + metadata.get("bestModelCheckpointMonitor"));
        System.out.println("bestModelCheckpointMonitorMode=" + metadata.get("bestModelCheckpointMonitorMode"));
        System.out.println("bestModelCheckpointMonitorValue=" + metadata.get("bestModelCheckpointMonitorValue"));
        System.out.println("bestModelCheckpointEpoch=" + metadata.get("bestModelCheckpointEpoch"));
        System.out.println("bestModelCheckpointRestored=" + metadata.get("bestModelCheckpointRestored"));
        System.out.println("earlyStoppingMonitor=" + metadata.get("earlyStoppingMonitor"));
        System.out.println("earlyStoppingMonitorMode=" + metadata.get("earlyStoppingMonitorMode"));
        System.out.println("earlyStoppingMonitorBestValue=" + metadata.get("earlyStoppingMonitorBestValue"));
        System.out.println("earlyStoppingMonitorBestEpoch=" + metadata.get("earlyStoppingMonitorBestEpoch"));
        System.out.println("earlyStoppingTriggered=" + metadata.get("earlyStoppingTriggered"));
        System.out.println("stopReason=" + metadata.get("stopReason"));
        System.out.println("epochHistorySize=" + metadata.get("epochHistorySize"));
        System.out.println("trainingHistoryFile=" + metadata.get("trainingHistoryFile"));
        System.out.println("trainingReportSaved=" + metadata.get("trainingReportSaved"));
        System.out.println("trainingReportFile=" + metadata.get("trainingReportFile"));
        System.out.println("trainingReportBytes=" + Files.size(reportFile));
        if (metadata.get("epochHistory") instanceof List<?> history
                && !history.isEmpty()
                && history.get(history.size() - 1) instanceof Map<?, ?> lastEpoch) {
            System.out.println("lastEpochHistory=" + lastEpoch);
        }
        System.out.println("schedulerSteps=" + metadata.get("learningRateSchedulerStepCount"));
        System.out.println("finalLearningRate=" + metadata.get("learningRate"));
        System.out.println("gradientL2BeforeClip=" + metadata.get("latestGradientL2NormBeforeClip"));
        System.out.println("gradientL2AfterClip=" + metadata.get("latestGradientL2Norm"));
        System.out.println("gradientClipped=" + metadata.get("latestGradientClipped"));
        System.out.println("parameterL2=" + metadata.get("latestParameterL2Norm"));
        System.out.println("trainBatchCount=" + metadata.get("trainBatchCount"));
        System.out.println("validationBatchCount=" + metadata.get("validationBatchCount"));
        System.out.println("trainSampleCount=" + metadata.get("trainSampleCount"));
        System.out.println("validationSampleCount=" + metadata.get("validationSampleCount"));
        System.out.println("trainSamplesPerSecond=" + metadata.get("trainSamplesPerSecond"));
        System.out.println("validationSamplesPerSecond=" + metadata.get("validationSamplesPerSecond"));
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
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
