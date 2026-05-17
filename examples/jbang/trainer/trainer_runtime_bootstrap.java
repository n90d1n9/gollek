///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-trainer-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-trainer:0.1.0-SNAPSHOT

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import tech.kayys.gollek.trainer.CanonicalTrainerRuntime;
import tech.kayys.gollek.trainer.Trainers;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_runtime_bootstrap {

    private static final class ConsoleListener implements TrainingListener {
        @Override
        public void onTrainingStart(TrainerSession session) {
            System.out.println("[listener] training started");
        }

        @Override
        public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
            System.out.printf("[listener] epoch=%d trainLoss=%.5f%n", epoch, trainLoss);
        }

        @Override
        public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
            System.out.printf("[listener] epoch=%d valLoss=%.5f%n", epoch, valLoss);
        }

        @Override
        public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
            System.out.printf(
                    "[listener] done epochs=%d bestValLoss=%.5f bestEpoch=%d%n",
                    summary.epochCount(),
                    summary.bestValidationLoss(),
                    summary.bestValidationEpoch());
        }
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        int epochs = args.length > 0 ? parseEpochs(args[0]) : 3;

        System.out.println("====================================================");
        System.out.println(" Gollek Trainer Runtime Bootstrap (JBang)");
        System.out.println("====================================================");
        System.out.printf("runtimeAvailable=%s%n", Trainers.runtimeAvailable());
        System.out.printf("runtimeMode=%s%n", Trainers.runtimeMode());
        System.out.printf("requestedEpochs=%d%n", epochs);

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(epochs)
                .gradientClip(1.0)
                .mixedPrecision(true)
                .checkpointDir(Path.of("trainer_checkpoints"))
                .listener(new ConsoleListener())
                .build();

        List<Integer> trainBatches = List.of(1, 2, 3, 4);
        List<Integer> validationBatches = List.of(1, 2);

        runtime.fit(trainBatches, validationBatches);

        TrainingSummary summary = runtime.summary();
        System.out.println("----------------------------------------------------");
        System.out.printf("finalEpochs=%d%n", summary.epochCount());
        System.out.printf("bestValidationLoss=%.5f%n", summary.bestValidationLoss());
        System.out.printf("bestValidationEpoch=%d%n", summary.bestValidationEpoch());
        System.out.printf("latestTrainLoss=%s%n", summary.latestTrainLoss());
        System.out.printf("latestValidationLoss=%s%n", summary.latestValidationLoss());
        System.out.printf("durationMs=%d%n", summary.durationMs());
        System.out.printf("metadata=%s%n", summary.metadata());

        runtime.close();
        System.out.println("trainer session closed");
    }

    private static int parseEpochs(String arg) {
        try {
            return Math.max(1, Integer.parseInt(arg.trim()));
        } catch (NumberFormatException ex) {
            System.out.printf("invalid epoch value '%s', fallback to 3%n", arg);
            return 3;
        }
    }
}
