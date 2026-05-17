///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
// This demo uses a lightweight vision-like classifier path that runs today.
//
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import java.util.Arrays;
import java.util.Locale;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;

class GollekSdkVisionExample {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        System.out.println("==============================================");
        System.out.println(" Gollek Vision Example (Compatibility Entry)");
        System.out.println("==============================================");

        // Simulated CHW image batch: 1 x (3*32*32)
        GradTensor input = GradTensor.randn(1, 3 * 32 * 32);

        Sequential classifier = new Sequential(
                new Linear(3 * 32 * 32, 128),
                new ReLU(),
                new Linear(128, 10));

        GradTensor logits = classifier.forward(input);
        float[] values = logits.data();

        int topClass = 0;
        float topScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > topScore) {
                topScore = values[i];
                topClass = i;
            }
        }

        System.out.println("Input shape: " + Arrays.toString(input.shape()));
        System.out.println("Output shape: " + Arrays.toString(logits.shape()));
        System.out.println("Top class index: " + topClass);
        System.out.printf("Top score: %.6f%n", topScore);
        System.out.println("Note: this is a synthetic compatibility demo (random input).");
    }
}
