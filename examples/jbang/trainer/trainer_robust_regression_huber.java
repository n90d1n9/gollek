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

public class trainer_robust_regression_huber {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);

        System.out.println("====================================================");
        System.out.println(" Gollek Robust Huber Regression Trainer (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        Linear model = new Linear(1, 1);
        TensorDataLoader train = Gollek.DL.dataLoader(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 8f}, 6, 1),
                GradTensor.of(new float[] {0f, 2f, 4f, 6f, 8f, 30f}, 6, 1),
                3);
        TensorDataLoader validation = Gollek.DL.dataLoader(
                GradTensor.of(new float[] {1.5f, 2.5f, 5f}, 3, 1),
                GradTensor.of(new float[] {3f, 5f, 10f}, 3, 1),
                3);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                validation,
                80,
                0.02f,
                Gollek.DL.TrainingPreset.REGRESSION_HUBER_SGD,
                Gollek.DL.trainingOptions()
                        .device(requestedDevice)
                        .gradientClip(1.0)
                        .regressionMetrics()
                        .stepLrEpochs(40, 0.5f)
                        .build());
        var evaluation = Gollek.DL.evaluate(
                model,
                validation,
                Gollek.DL.TrainingPreset.REGRESSION_HUBER_SGD,
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
        System.out.println("trainRMSE=" + metadata.get("trainMetric.rmse"));
        System.out.println("validationRMSE=" + metadata.get("validationMetric.rmse"));
        System.out.println("validationR2=" + metadata.get("validationMetric.r2"));
        System.out.println("schedulerSteps=" + metadata.get("learningRateSchedulerStepCount"));
        System.out.println("finalLearningRate=" + metadata.get("learningRate"));
        System.out.println("trainBatches=" + train.numBatches());
        System.out.println("validationBatches=" + validation.numBatches());
        System.out.println("executionBackend=" + metadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + metadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + metadata.get("executionAccelerated"));
        System.out.println("requestedDeviceAvailable=" + metadata.get("requestedDeviceAvailable"));
        System.out.println("acceleratedMatmulCalls=" + metadata.get("acceleratedMatmulCalls"));
        System.out.println("----------------------------------------------------");
        for (float x : new float[] {1f, 3f, 6f}) {
            float prediction = model.forward(GradTensor.of(new float[] {x}, 1, 1)).data()[0];
            System.out.printf("probe x=%.1f predicted=%.4f expectedClean=%.1f%n", x, prediction, 2f * x);
        }
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
