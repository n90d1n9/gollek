///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-java-local:0.1.0-SNAPSHOT
//JAVA 25

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModalityType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Stable Diffusion Batch Generation Example
 * 
 * Generates multiple image variations using different seeds.
 * 
 * Usage:
 *   jbang stable_diffusion_batch.java
 *   jbang stable_diffusion_batch.java --prompt "a cat" --count 5 --steps 20
 */
public class stable_diffusion_batch {

    public static void main(String[] args) throws Exception {
        String prompt = getArg(args, "--prompt", "a cat playing ball, digital art");
        int count = Integer.parseInt(getArg(args, "--count", "5"));
        int steps = Integer.parseInt(getArg(args, "--steps", "20"));
        float guidanceScale = Float.parseFloat(getArg(args, "--guidance-scale", "7.5"));

        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SD Batch Generation - Multiple Variations    ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Prompt: " + prompt);
        System.out.println("Count: " + count);
        System.out.println("Steps: " + steps);
        System.out.println("Guidance: " + guidanceScale);
        System.out.println();

        GollekSdk sdk = GollekSdk.builder().local().build();

        // Generate variations with different seeds
        long[] seeds = new long[count];
        for (int i = 0; i < count; i++) {
            seeds[i] = 100L + (i * 100); // 100, 200, 300, ...
        }

        int successCount = 0;
        long totalStart = System.currentTimeMillis();

        for (int i = 0; i < seeds.length; i++) {
            final long seed = seeds[i];
            String outputFile = String.format("batch-variation-%d-seed-%d.png", i + 1, seed);
            
            System.out.printf("[%d/%d] Generating with seed %d...%n", i + 1, count, seed);

            InferenceRequest request = InferenceRequest.builder()
                    .model("CompVis/stable-diffusion-v1-4")
                    .prompt(prompt)
                    .parameter("seed", seed)
                    .parameter("steps", steps)
                    .parameter("guidance_scale", guidanceScale)
                    .streaming(true)
                    .build();

            CountDownLatch latch = new CountDownLatch(1);
            boolean[] success = {false};

            sdk.streamCompletion(request)
                    .subscribe().with(
                            chunk -> {
                                if (chunk.modality() == ModalityType.IMAGE && chunk.imageDeltaBase64() != null) {
                                    try {
                                        byte[] png = Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                        Files.write(Path.of(outputFile), png);
                                        System.out.printf("  ✅ Saved: %s (%,d bytes)%n", outputFile, png.length);
                                        success[0] = true;
                                    } catch (Exception e) {
                                        System.err.println("  ❌ Failed: " + e.getMessage());
                                    }
                                }
                            },
                            error -> System.err.println("  ❌ Error: " + error.getMessage()),
                            latch::countDown
                    );

            latch.await();
            if (success[0]) successCount++;
            System.out.println();
        }

        long totalDuration = System.currentTimeMillis() - totalStart;
        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║                   Batch Summary                        ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.printf("Total: %d/%d successful%n", successCount, count);
        System.out.printf("Time: %.1f seconds (~%.1f sec/image)%n", 
                totalDuration / 1000.0, totalDuration / 1000.0 / count);
    }

    private static String getArg(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) return args[i + 1];
        }
        return defaultValue;
    }
}
