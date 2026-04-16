///usr/bin/env jbang "$0" "$@" ; exit $?
// Update dependencies with //DEPS
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
import java.util.concurrent.CountDownLatch;

/**
 * Stable Diffusion Image Generation Example
 * 
 * Generates an image from a text prompt using ONNX-based Stable Diffusion.
 * 
 * Usage:
 *   jbang stable_diffusion_generation.java
 *   jbang stable_diffusion_generation.java --prompt "a cyberpunk cat" --output cyberpunk-cat.png
 * 
 * Parameters:
 *   --prompt         Text prompt (default: "a cat playing ball")
 *   --output         Output file path (default: output.png)
 *   --seed           Random seed (default: 42)
 *   --steps          Denoising steps (default: 20)
 *   --guidance-scale CFG scale (default: 7.5)
 */
public class stable_diffusion_generation {

    public static void main(String[] args) throws Exception {
        // Parse command-line arguments
        String prompt = getArg(args, "--prompt", "a cat playing ball");
        String outputFile = getArg(args, "--output", "output.png");
        long seed = Long.parseLong(getArg(args, "--seed", "42"));
        int steps = Integer.parseInt(getArg(args, "--steps", "20"));
        float guidanceScale = Float.parseFloat(getArg(args, "--guidance-scale", "7.5"));

        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek Stable Diffusion - Image Generation Example  ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Prompt: " + prompt);
        System.out.println("Seed: " + seed);
        System.out.println("Steps: " + steps);
        System.out.println("Guidance Scale: " + guidanceScale);
        System.out.println("Output: " + outputFile);
        System.out.println();

        // Initialize SDK
        GollekSdk sdk = GollekSdk.builder().local().build();

        // Build inference request
        InferenceRequest request = InferenceRequest.builder()
                .model("CompVis/stable-diffusion-v1-4")
                .prompt(prompt)
                .parameter("seed", seed)
                .parameter("steps", steps)
                .parameter("guidance_scale", guidanceScale)
                .streaming(true)
                .build();

        System.out.println("⏳ Generating image...");
        System.out.println();

        CountDownLatch latch = new CountDownLatch(1);
        long startTime = System.currentTimeMillis();

        // Stream completion with image chunk detection
        sdk.streamCompletion(request)
                .subscribe().with(
                        chunk -> {
                            // Progress updates
                            if (chunk.delta() != null && chunk.delta().startsWith("[")) {
                                System.out.print("\r" + chunk.delta() + "  ");
                                System.out.flush();
                            }

                            // Image chunk
                            if (chunk.modality() == ModalityType.IMAGE) {
                                if (chunk.imageDeltaBase64() != null) {
                                    try {
                                        byte[] pngData = Base64.getDecoder()
                                                .decode(chunk.imageDeltaBase64());
                                        Path outputPath = Path.of(outputFile);
                                        Files.write(outputPath, pngData);
                                        
                                        long duration = System.currentTimeMillis() - startTime;
                                        System.out.println();
                                        System.out.println();
                                        System.out.println("✅ Image saved to: " + outputPath.toAbsolutePath());
                                        System.out.printf("⏱️  Generation time: %.1f seconds%n", duration / 1000.0);
                                        System.out.printf("📊 File size: %,d bytes%n", pngData.length);
                                        
                                        // Auto-open on macOS
                                        if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                                            Runtime.getRuntime().exec(new String[]{"open", outputPath.toAbsolutePath().toString()});
                                        }
                                    } catch (Exception e) {
                                        System.err.println("\n❌ Failed to save image: " + e.getMessage());
                                    }
                                }
                            }
                        },
                        error -> {
                            System.err.println("\n❌ Generation failed: " + error.getMessage());
                            latch.countDown();
                        },
                        () -> {
                            latch.countDown();
                        });

        latch.await();
    }

    private static String getArg(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
