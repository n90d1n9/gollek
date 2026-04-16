///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-java-local:0.1.0-SNAPSHOT
//JAVA 25

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModalityType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * 🎨 Stable Diffusion Image Generation - Jupyter/Interactive Version
 * 
 * Run with:
 *   jbang sd_generation.java
 *   jbang sd_generation.java --prompt "a cat" --steps 10
 *   jbang sd_generation.java --prompt "cyberpunk city" --seed 123 --steps 30
 */
public class sd_generation {

    public static void main(String[] args) throws Exception {
        String prompt = getArg(args, "--prompt", "a cat playing ball");
        long seed = Long.parseLong(getArg(args, "--seed", "42"));
        int steps = Integer.parseInt(getArg(args, "--steps", "20"));
        float guidance = Float.parseFloat(getArg(args, "--guidance", "7.5"));
        String output = getArg(args, "--output", "output.png");

        System.out.println("🎨 Gollek Stable Diffusion");
        System.out.println("Prompt: " + prompt);
        System.out.println("Seed: " + seed + " | Steps: " + steps + " | CFG: " + guidance);
        System.out.println("Output: " + output);
        System.out.println();

        // Initialize SDK
        System.out.println("⏳ Initializing SDK...");
        GollekSdk sdk = GollekSdk.builder().local().build();
        System.out.println("✅ SDK ready\n");

        // Build request
        InferenceRequest request = InferenceRequest.builder()
            .model("CompVis/stable-diffusion-v1-4")
            .prompt(prompt)
            .parameter("seed", seed)
            .parameter("steps", steps)
            .parameter("guidance_scale", guidance)
            .streaming(true)
            .build();

        System.out.println("⏳ Generating...");
        CountDownLatch latch = new CountDownLatch(1);
        long startTime = System.currentTimeMillis();
        final boolean[] success = {false};

        sdk.streamCompletion(request)
            .subscribe().with(
                chunk -> {
                    // Progress updates
                    if (chunk.delta() != null && chunk.delta().startsWith("[")) {
                        System.out.print("\r" + chunk.delta() + "  ");
                        System.out.flush();
                    }

                    // Image chunk
                    if (chunk.modality() == ModalityType.IMAGE && chunk.imageDeltaBase64() != null) {
                        try {
                            byte[] png = Base64.getDecoder().decode(chunk.imageDeltaBase64());
                            Files.write(Path.of(output), png);
                            success[0] = true;
                            
                            long duration = System.currentTimeMillis() - startTime;
                            System.out.println();
                            System.out.println("\n✅ Image saved: " + output);
                            System.out.printf("⏱️  Time: %.1fs%n", duration / 1000.0);
                            System.out.printf("📊 Size: %,d bytes%n", png.length);
                            
                            // Verify and display image info
                            BufferedImage img = ImageIO.read(new ByteArrayInputStream(png));
                            if (img != null) {
                                System.out.printf("🖼️  Dimensions: %dx%d pixels%n", 
                                    img.getWidth(), img.getHeight());
                            }
                            
                            // Auto-open on macOS
                            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                                Runtime.getRuntime().exec(
                                    new String[]{"open", Path.of(output).toAbsolutePath().toString()});
                            }
                        } catch (Exception e) {
                            System.err.println("\n❌ Save failed: " + e.getMessage());
                        }
                    }
                },
                error -> {
                    System.err.println("\n❌ Error: " + error.getMessage());
                    latch.countDown();
                },
                latch::countDown
            );

        latch.await(600, TimeUnit.SECONDS);
        
        if (!success[0]) {
            System.err.println("\n❌ Generation failed or timed out");
            System.exit(1);
        }
    }

    private static String getArg(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) return args[i + 1];
        }
        return defaultValue;
    }
}
