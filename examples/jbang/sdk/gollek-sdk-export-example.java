///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
// This export demo uses currently available save/load capabilities.
//
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;

class GollekSdkExportExample {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);

        System.out.println("==============================================");
        System.out.println(" Gollek Export Example (Compatibility Entry)");
        System.out.println("==============================================");

        Sequential model = new Sequential(
                new Linear(16, 32),
                new ReLU(),
                new Linear(32, 4));

        Path outDir = Path.of("compat_exports");
        Files.createDirectories(outDir);
        Path artifact = outDir.resolve("model.bin");

        model.save(artifact);
        System.out.println("Saved model: " + artifact.toAbsolutePath());

        Sequential restored = new Sequential(
                new Linear(16, 32),
                new ReLU(),
                new Linear(32, 4));
        restored.load(artifact);

        GradTensor sample = GradTensor.randn(1, 16);
        GradTensor originalOut = model.forward(sample);
        GradTensor restoredOut = restored.forward(sample);

        System.out.println("Output shape: " + Arrays.toString(originalOut.shape()));
        System.out.printf("Original first value: %.6f%n", originalOut.item(0));
        System.out.printf("Restored first value: %.6f%n", restoredOut.item(0));

        int warmup = 10;
        int iterations = 100;
        for (int i = 0; i < warmup; i++) {
            model.forward(sample);
        }
        long started = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            model.forward(sample);
        }
        long elapsedNs = System.nanoTime() - started;
        double avgMs = (elapsedNs / 1_000_000.0d) / iterations;
        System.out.printf("Avg forward latency (ms): %.6f%n", avgMs);
    }
}
