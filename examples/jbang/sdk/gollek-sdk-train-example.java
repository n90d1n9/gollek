///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
// Canonical trainer modules are used under the hood.
//
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

class GollekSdkTrainExample {

    private static final class ConsoleListener implements TrainingListener {
        @Override
        public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
            System.out.printf("epoch=%d trainLoss=%.5f%n", epoch, trainLoss);
        }

        @Override
        public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
            System.out.printf(
                    "done epochs=%d bestVal=%.5f mode=%s%n",
                    summary.epochCount(),
                    summary.bestValidationLoss(),
                    summary.metadata().getOrDefault("runtime", "unknown"));
        }
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        int epochs = args.length > 0 ? parseEpochs(args[0]) : 3;

        System.out.println("==============================================");
        System.out.println(" Gollek Train Example (Compatibility Entry)");
        System.out.println("==============================================");
        System.out.println("runtimeAvailable=" + Trainers.runtimeAvailable());
        System.out.println("runtimeMode=" + Trainers.runtimeMode());

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(epochs)
                .gradientClip(1.0)
                .mixedPrecision(true)
                .checkpointDir(Path.of("trainer_checkpoints"))
                .listener(new ConsoleListener())
                .build();

        runtime.fit(List.of(1, 2, 3, 4), List.of(1, 2));

        TrainingSummary summary = runtime.summary();
        System.out.println("----------------------------------------------");
        System.out.println("finalEpochs=" + summary.epochCount());
        System.out.println("bestValidationEpoch=" + summary.bestValidationEpoch());
        System.out.println("durationMs=" + summary.durationMs());
        runtime.close();
    }

    private static int parseEpochs(String raw) {
        try {
            return Math.max(1, Integer.parseInt(raw));
        } catch (NumberFormatException ex) {
            return 3;
        }
    }
}
