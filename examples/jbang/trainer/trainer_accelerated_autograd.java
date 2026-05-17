///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT
// For CUDA hosts, publish and add the CUDA binding module to the JBang classpath.

import java.util.Locale;
import java.util.Map;
import java.nio.file.Path;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_accelerated_autograd {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Accelerated Autograd Trainer (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        Linear model = new Linear(1, 1);
        var split = Gollek.DL.trainValidationSplit(
                GradTensor.of(new float[] {1f, 1.5f, 2f, 2.5f, 3f, 4f}, 6, 1),
                GradTensor.of(new float[] {2f, 3f, 4f, 5f, 6f, 8f}, 6, 1),
                0.67,
                2026L);
        TensorDataLoader train = split.trainLoader(2, true, 2026L);
        TensorDataLoader validation = split.validationLoader(2);
        Path checkpointDir = Path.of("trainer_checkpoints", "accelerated_autograd");

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                2,
                true,
                2026L,
                3,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .gradientAccumulationSteps(1)
                        .checkpointDir(checkpointDir)
                        .restoreBestModelAtEnd()
                        .regressionMetrics()
                        .stepLrBatches(2, 0.8f)
                        .build());
        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                requestedDevice,
                Gollek.DL.maeMetric(),
                Gollek.DL.mseMetric(),
                Gollek.DL.rmseMetric(),
                Gollek.DL.r2Metric());

        Map<String, Object> metadata = summary.metadata();
        System.out.println("----------------------------------------------------");
        System.out.printf("epochs=%d trainLoss=%.6f validationLoss=%.6f%n",
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss());
        System.out.printf("evalLoss=%.6f evalMAE=%.6f evalMSE=%.6f evalRMSE=%.6f evalR2=%.6f%n",
                evaluation.loss(),
                evaluation.metric("mae"),
                evaluation.metric("mse"),
                evaluation.metric("rmse"),
                evaluation.metric("r2"));
        System.out.println("trainMAE=" + metadata.get("trainMetric.mae"));
        System.out.println("validationMAE=" + metadata.get("validationMetric.mae"));
        System.out.println("validationRMSE=" + metadata.get("validationMetric.rmse"));
        System.out.println("validationR2=" + metadata.get("validationMetric.r2"));
        System.out.println("checkpointDir=" + checkpointDir);
        System.out.println("bestModelCheckpointFile=" + metadata.get("bestModelCheckpointFile"));
        System.out.println("bestModelCheckpointEpoch=" + metadata.get("bestModelCheckpointEpoch"));
        System.out.println("bestModelCheckpointRestored=" + metadata.get("bestModelCheckpointRestored"));
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
        System.out.println("requestedDeviceAvailable=" + metadata.get("requestedDeviceAvailable"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
        System.out.println("acceleratedMatmulCallsAtStart=" + metadata.get("acceleratedMatmulCallsAtStart"));
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
