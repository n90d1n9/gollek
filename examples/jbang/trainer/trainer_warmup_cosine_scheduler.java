///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_warmup_cosine_scheduler {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Warmup Cosine Scheduler (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        Linear model = new Linear(1, 1);
        var split = Gollek.DL.trainValidationSplit(
                GradTensor.of(new float[] {1f, 1.5f, 2f, 2.5f, 3f, 3.5f, 4f, 4.5f}, 8, 1),
                GradTensor.of(new float[] {2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 8, 1),
                0.75,
                2026L);

        int epochs = 3;
        int batchSize = 2;
        TensorDataLoader train = split.trainLoader(batchSize, true, 2026L);
        TensorDataLoader validation = split.validationLoader(batchSize);
        int totalSchedulerSteps = epochs * train.numBatches();
        int warmupSteps = Math.max(1, totalSchedulerSteps / 3);
        Path checkpointDir = Path.of("trainer_checkpoints", "warmup_cosine");

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                batchSize,
                true,
                2026L,
                epochs,
                0.08f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .checkpointDir(checkpointDir)
                        .restoreBestModelAtEnd()
                        .regressionMetrics()
                        .warmupCosineLrBatches(warmupSteps, totalSchedulerSteps, 0.08f, 0.008f)
                        .build());

        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                requestedDevice,
                Gollek.DL.maeMetric(),
                Gollek.DL.rmseMetric(),
                Gollek.DL.r2Metric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalMAE=%.6f evalRMSE=%.6f evalR2=%.6f%n",
                evaluation.loss(),
                evaluation.metric("mae"),
                evaluation.metric("rmse"),
                evaluation.metric("r2"));
        System.out.println("checkpointDir=" + checkpointDir);
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("warmupSteps=" + warmupSteps);
        System.out.println("totalSchedulerSteps=" + totalSchedulerSteps);
        System.out.println("schedulerType=" + metadata.get("learningRateSchedulerType"));
        System.out.println("schedulerStepUnit=" + metadata.get("learningRateSchedulerStepUnit"));
        System.out.println("schedulerSteps=" + metadata.get("learningRateSchedulerStepCount"));
        System.out.println("schedulerState=" + metadata.get("learningRateSchedulerState"));
        System.out.println("finalLearningRate=" + metadata.get("learningRate"));
        System.out.println("trainingReportSaved=" + metadata.get("trainingReportSaved"));
        System.out.println("trainingReportFile=" + metadata.get("trainingReportFile"));
        if (metadata.get("trainingReportFile") instanceof String reportFile) {
            System.out.println("trainingReportBytes=" + Files.size(Path.of(reportFile)));
        }
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("requestedDeviceAvailable=" + metadata.get("requestedDeviceAvailable"));
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
