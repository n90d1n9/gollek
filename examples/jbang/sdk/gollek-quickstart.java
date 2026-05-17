///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
// This quickstart is runnable with current local artifacts.
//
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;
import tech.kayys.gollek.ml.nn.Trainer;
import tech.kayys.gollek.ml.nn.optim.Adam;
import java.util.Locale;

class GollekQuickstart {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        System.out.println("==============================================");
        System.out.println(" Gollek Quickstart (Compatibility Entry)");
        System.out.println("==============================================");

        Sequential model = new Sequential(
                new Linear(8, 16),
                new ReLU(),
                new Linear(16, 1));

        GradTensor inputs = GradTensor.randn(32, 8);
        GradTensor targets = GradTensor.randn(32, 1);

        Trainer trainer = Trainer.builder()
                .model(model)
                .optimizer(new Adam(model.parameters(), 0.01f))
                .lossFunction((prediction, target) -> prediction.sub(target).pow(2f).mean())
                .epochs(5)
                .callback(Trainer.printCallback())
                .build();

        Trainer.TrainingResult result = trainer.fit(inputs, targets, 8);

        System.out.println("----------------------------------------------");
        System.out.printf("Final loss: %.6f%n", result.finalLoss());
        System.out.printf("Converged(<1.0): %s%n", result.converged(1.0f));
        System.out.println("Model parameters: " + model.parameterCountFormatted());
        System.out.println();
        System.out.println("Canonical trainer example:");
        System.out.println("  jbang trainer/trainer_runtime_bootstrap.java");
    }
}
