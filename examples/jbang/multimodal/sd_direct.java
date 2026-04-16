///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-model-repo-local:0.1.0-SNAPSHOT
//JAVA 25

import tech.kayys.gollek.onnx.runner.StableDiffusionOnnxRunner;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.spi.model.ModalityType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * 🎨 Stable Diffusion - Direct Runner Example (No CDI)
 * 
 * This example uses the SD runner directly without the SDK/CDI layer.
 * 
 * Run with:
 *   jbang sd_direct.java
 *   jbang sd_direct.java --prompt "a cat" --steps 5
 */
public class sd_direct {

    public static void main(String[] args) throws Exception {
        String prompt = getArg(args, "--prompt", "a cat playing ball");
        long seed = Long.parseLong(getArg(args, "--seed", "42"));
        int steps = Integer.parseInt(getArg(args, "--steps", "10"));
        float guidance = Float.parseFloat(getArg(args, "--guidance", "7.5"));
        String output = getArg(args, "--output", "output.png");

        System.out.println("🎨 Stable Diffusion (Direct Runner)");
        System.out.println("Prompt: " + prompt);
        System.out.println("Seed: " + seed + " | Steps: " + steps);
        System.out.println();

        // Find model path
        Path modelPath = findModelPath();
        if (modelPath == null) {
            System.err.println("❌ Model not found. Run this first:");
            System.err.println("   java -jar gollek-runner.jar pull CompVis/stable-diffusion-v1-4 --branch onnx");
            System.exit(1);
        }

        System.out.println("📂 Model: " + modelPath);

        // Initialize runner
        StableDiffusionOnnxRunner runner = new StableDiffusionOnnxRunner();
        ModelManifest manifest = ModelManifest.builder()
            .modelId("CompVis/stable-diffusion-v1-4")
            .name("stable-diffusion-v1-4")
            .version("1.4")
            .path(modelPath.toString())
            .artifacts(Map.of(ModelFormat.ONNX, 
                new ArtifactLocation(modelPath.toUri().toString(), null, null, null)))
            .build();

        RunnerConfiguration config = RunnerConfiguration.builder()
            .build();

        System.out.println("⏳ Loading model...");
        runner.initialize(manifest, config);
        System.out.println("✅ Model loaded\n");

        // Build request
        InferenceRequest request = InferenceRequest.builder()
            .requestId("sd-001")
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

        runner.stream(request)
            .subscribe().with(
                chunk -> {
                    if (chunk.delta() != null && chunk.delta().startsWith("[")) {
                        System.out.print("\r" + chunk.delta() + "  ");
                        System.out.flush();
                    }

                    if (chunk.modality() == ModalityType.IMAGE && chunk.imageDeltaBase64() != null) {
                        try {
                            byte[] png = Base64.getDecoder().decode(chunk.imageDeltaBase64());
                            Files.write(Path.of(output), png);
                            
                            long duration = System.currentTimeMillis() - startTime;
                            System.out.println();
                            System.out.println("\n✅ Image saved: " + output);
                            System.out.printf("⏱️  Time: %.1fs%n", duration / 1000.0);
                            System.out.printf("📊 Size: %,d bytes%n", png.length);

                            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                                Runtime.getRuntime().exec(
                                    new String[]{"open", Path.of(output).toAbsolutePath().toString()});
                            }
                        } catch (Exception e) {
                            System.err.println("\n❌ Save failed: " + e.getMessage());
                            e.printStackTrace();
                        }
                    }
                },
                error -> {
                    System.err.println("\n❌ Error: " + error.getMessage());
                    error.printStackTrace();
                    latch.countDown();
                },
                () -> {
                    runner.close();
                    latch.countDown();
                }
            );

        latch.await(600, TimeUnit.SECONDS);
    }

    private static Path findModelPath() {
        // Check common locations
        Path[] paths = {
            Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs"),
            Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", 
                    "CompVis", "stable-diffusion-v1-4"),
        };

        for (Path p : paths) {
            if (Files.isDirectory(p)) {
                // Look for unet/model.onnx
                try (var walk = Files.walk(p, 3)) {
                    Path found = walk.filter(f -> f.toString().endsWith("unet/model.onnx"))
                        .findFirst()
                        .map(Path::getParent)
                        .orElse(null);
                    if (found != null) return found.getParent();
                } catch (Exception ignored) {}
            }
        }
        return null;
    }

    private static String getArg(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) return args[i + 1];
        }
        return defaultValue;
    }
}
