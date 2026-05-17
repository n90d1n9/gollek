///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.ml.optim.StepLR;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.train.data.DataLoader.TensorDataLoader;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_resume_history {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String requestedDevice = requestedDevice(args);
        Path checkpointDir = Path.of("trainer_checkpoints", "resume_history");
        Path historyFile = checkpointDir.resolve("canonical-history.csv");
        Path reportFile = checkpointDir.resolve("canonical-report.json");
        deleteDirectory(checkpointDir);

        System.out.println("====================================================");
        System.out.println(" Gollek Resume + Durable History (JBang)");
        System.out.println("====================================================");
        System.out.println("requestedDevice=" + requestedDevice);
        System.out.println("initialBackend=" + Gollek.DL.accelerationStatus(requestedDevice));

        var split = Gollek.DL.trainValidationSplit(
                GradTensor.of(new float[] {1f, 1.5f, 2f, 2.5f, 3f, 4f}, 6, 1),
                GradTensor.of(new float[] {2f, 3f, 4f, 5f, 6f, 8f}, 6, 1),
                0.67,
                2026L);
        TensorDataLoader train = split.trainLoader(2, true, 2026L);
        TensorDataLoader validation = split.validationLoader(2);
        MSELoss mseLoss = new MSELoss();

        TrainingListener interruptAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        TrainingSummary interrupted = runTrainer(
                new Linear(1, 1),
                mseLoss,
                train,
                validation,
                checkpointDir,
                requestedDevice,
                false,
                interruptAfterFirstEpoch);
        long interruptedHistoryLines = csvLineCount(historyFile);

        TrainingSummary resumed = runTrainer(
                new Linear(1, 1),
                mseLoss,
                train,
                validation,
                checkpointDir,
                requestedDevice,
                true,
                null);

        Map<String, Object> resumedMetadata = resumed.metadata();
        System.out.println("----------------------------------------------------");
        System.out.println("checkpointDir=" + checkpointDir);
        System.out.println("historyFile=" + historyFile);
        System.out.println("reportFile=" + reportFile);
        System.out.println("interruptedEpochs=" + interrupted.epochCount());
        System.out.println("interruptedHistoryLines=" + interruptedHistoryLines);
        System.out.println("resumedEpochs=" + resumed.epochCount());
        System.out.println("trainingHistoryLoaded=" + resumedMetadata.get("trainingHistoryLoaded"));
        System.out.println("trainingHistoryLoadFailed=" + resumedMetadata.get("trainingHistoryLoadFailed"));
        System.out.println("trainingHistorySaved=" + resumedMetadata.get("trainingHistorySaved"));
        System.out.println("trainingReportSaved=" + resumedMetadata.get("trainingReportSaved"));
        System.out.println("trainingReportFile=" + resumedMetadata.get("trainingReportFile"));
        System.out.println("epochHistorySize=" + resumedMetadata.get("epochHistorySize"));
        System.out.println("validationMAE=" + resumedMetadata.get("validationMetric.mae"));
        System.out.println("finalLearningRate=" + resumedMetadata.get("learningRate"));
        System.out.println("gradientL2BeforeClip=" + resumedMetadata.get("latestGradientL2NormBeforeClip"));
        System.out.println("gradientL2AfterClip=" + resumedMetadata.get("latestGradientL2Norm"));
        System.out.println("gradientClipped=" + resumedMetadata.get("latestGradientClipped"));
        System.out.println("parameterL2=" + resumedMetadata.get("latestParameterL2Norm"));
        System.out.println("trainBatchCount=" + resumedMetadata.get("trainBatchCount"));
        System.out.println("validationBatchCount=" + resumedMetadata.get("validationBatchCount"));
        System.out.println("trainSampleCount=" + resumedMetadata.get("trainSampleCount"));
        System.out.println("validationSampleCount=" + resumedMetadata.get("validationSampleCount"));
        System.out.println("trainSamplesPerSecond=" + resumedMetadata.get("trainSamplesPerSecond"));
        System.out.println("validationSamplesPerSecond=" + resumedMetadata.get("validationSamplesPerSecond"));
        System.out.println("csvLineCount=" + csvLineCount(historyFile));
        System.out.println("reportBytes=" + Files.size(reportFile));
        if (resumedMetadata.get("epochHistory") instanceof List<?> history
                && !history.isEmpty()
                && history.get(0) instanceof Map<?, ?> firstEpoch
                && history.get(history.size() - 1) instanceof Map<?, ?> lastEpoch) {
            System.out.println("firstEpochHistory=" + firstEpoch);
            System.out.println("lastEpochHistory=" + lastEpoch);
        }
        System.out.println("executionBackend=" + resumedMetadata.get("executionBackend"));
        System.out.println("executionDeviceName=" + resumedMetadata.get("executionDeviceName"));
        System.out.println("executionAccelerated=" + resumedMetadata.get("executionAccelerated"));
        System.out.println("acceleratedMatmulCalls=" + resumedMetadata.get("acceleratedMatmulCalls"));
    }

    private static TrainingSummary runTrainer(
            Linear model,
            MSELoss mseLoss,
            TensorDataLoader train,
            TensorDataLoader validation,
            Path checkpointDir,
            String requestedDevice,
            boolean resume,
            TrainingListener listener) {
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        CanonicalTrainer.Builder builder = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .device(requestedDevice)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .scheduler(new StepLR(optimizer, 2, 0.8f));
        if (resume) {
            builder.resumeFromCheckpoint();
        }
        if (listener != null) {
            builder.listener(listener);
        }
        try (CanonicalTrainer trainer = builder.build()) {
            trainer.fit(train, validation);
            return trainer.summary();
        }
    }

    private static long csvLineCount(Path path) throws IOException {
        return Files.readString(path).lines().count();
    }

    private static void deleteDirectory(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (var stream = Files.walk(path)) {
            for (Path entry : stream.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(entry);
            }
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
