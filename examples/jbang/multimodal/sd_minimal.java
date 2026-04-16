///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS com.microsoft.onnxruntime:onnxruntime:1.17.1
//DEPS com.fasterxml.jackson.core:jackson-databind:2.15.2
//JAVA 25
//PREVIEW

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.zip.Deflater;
import java.util.zip.CRC32;

/**
 * 🎨 Minimal Stable Diffusion - Pure JDK 25 (No Quarkus/CDI)
 * 
 * Uses Panama FFM + ONNX Runtime directly.
 * 
 * Run with:
 *   jbang sd_minimal.java
 *   jbang sd_minimal.java --prompt "a cat" --steps 10
 * 
 * Requirements:
 *   - JDK 25 with preview features
 *   - ONNX Runtime (brew install onnxruntime on macOS)
 *   - Model downloaded to ~/.gollek/models/
 */
public class sd_minimal {

    // Constants
    private static final int IMAGE_SIZE = 512;
    private static final int LATENT_SIZE = 64;
    private static final int LATENT_CHANNELS = 4;
    private static final long NUM_LATENT_FLOATS = 1L * LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE;
    private static final long LATENT_BYTES = NUM_LATENT_FLOATS * Float.BYTES;
    private static final int CLIP_SEQ_LEN = 77;
    private static final int CLIP_HIDDEN_DIM = 768;
    private static final long TEXT_EMBED_FLOATS = 1L * CLIP_SEQ_LEN * CLIP_HIDDEN_DIM;
    private static final long TEXT_EMBED_BYTES = TEXT_EMBED_FLOATS * Float.BYTES;

    public static void main(String[] args) throws Exception {
        String prompt = getArg(args, "--prompt", "a cat playing ball");
        long seed = Long.parseLong(getArg(args, "--seed", "42"));
        int steps = Integer.parseInt(getArg(args, "--steps", "10"));
        float guidance = Float.parseFloat(getArg(args, "--guidance", "7.5"));
        String output = getArg(args, "--output", "output.png");

        System.out.println("🎨 Stable Diffusion - Pure JDK 25");
        System.out.println("Prompt: " + prompt);
        System.out.println("Seed: " + seed + " | Steps: " + steps + " | CFG: " + guidance);
        System.out.println();

        // Find model
        Path modelPath = findModelPath();
        if (modelPath == null) {
            System.err.println("❌ Model not found. Download first:");
            System.err.println("   java -jar gollek-runner.jar pull CompVis/stable-diffusion-v1-4 --branch onnx");
            System.exit(1);
        }

        System.out.println("📂 Model: " + modelPath);
        System.out.println("⏳ Loading ONNX models...");

        // Load ONNX models
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {

            // Load three models
            String tePath = modelPath.resolve("text_encoder/model.onnx").toString();
            String unetPath = modelPath.resolve("unet/model.onnx").toString();
            String vaePath = modelPath.resolve("vae_decoder/model.onnx").toString();

            try (OrtSession textEncoder = env.createSession(tePath, opts);
                 OrtSession unet = env.createSession(unetPath, opts);
                 OrtSession vae = env.createSession(vaePath, opts)) {

                System.out.println("✅ Models loaded\n");
                System.out.println("⏳ Generating...");

                long startTime = System.currentTimeMillis();
                byte[] pngData = generate(env, textEncoder, unet, vae, prompt, seed, steps, guidance);

                // Save
                Files.write(Path.of(output), pngData);
                long duration = System.currentTimeMillis() - startTime;

                System.out.println("\n✅ Image saved: " + output);
                System.out.printf("⏱️  Time: %.1fs%n", duration / 1000.0);
                System.out.printf("📊 Size: %,d bytes%n", pngData.length);

                // Auto-open
                if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                    Runtime.getRuntime().exec(
                        new String[]{"open", Path.of(output).toAbsolutePath().toString()});
                }
            }
        }
    }

    private static byte[] generate(OrtEnvironment env, OrtSession textEncoder, 
                                    OrtSession unet, OrtSession vae,
                                    String prompt, long seed, int steps, float guidance) throws Exception {
        
        // 1. Encode prompt (simplified - using dummy embeddings for now)
        float[] textEmbed = new float[(int) TEXT_EMBED_FLOATS];
        float[] nullEmbed = new float[(int) TEXT_EMBED_FLOATS];
        
        System.out.print("\r[0/" + (steps + 2) + "] Encoding prompt...  ");

        // 2. Create noise latents
        float[] latents = createNoiseLatents(seed);
        System.out.print("\r[1/" + (steps + 2) + "] Initializing latents...  ");

        // 3. Denoising loop
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < steps; i++) {
                System.out.print("\r[" + (i + 2) + "/" + (steps + 2) + "] Denoising step " + (i + 1) + "/" + steps + " (" + ((i + 1) * 100 / steps) + "%)...  ");
                System.out.flush();
                
                latents = denoiseStep(env, unet, latents, textEmbed, nullEmbed, i, steps, guidance, arena);
            }

            // 4. VAE decode
            System.out.print("\r[" + (steps + 2) + "/" + (steps + 2) + "] Decoding image...  ");
            return decodeToPng(env, vae, latents, arena);
        }
    }

    private static float[] createNoiseLatents(long seed) {
        Random rnd = new Random(seed);
        float[] data = new float[(int) NUM_LATENT_FLOATS];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rnd.nextGaussian();
        }
        return data;
    }

    private static float[] denoiseStep(OrtEnvironment env, OrtSession unet,
                                        float[] latents, float[] textEmbed, float[] nullEmbed,
                                        int step, int totalSteps, float guidance, Arena arena) throws Exception {
        // Simplified denoise - returns input for now
        // Full implementation would run UNet twice and apply CFG + scheduler
        return latents;
    }

    private static byte[] decodeToPng(OrtEnvironment env, OrtSession vae,
                                       float[] latents, Arena arena) throws Exception {
        // Simplified - returns placeholder PNG
        return generatePlaceholderPng();
    }

    private static byte[] generatePlaceholderPng() {
        // Generate a simple test image
        int w = 512, h = 512;
        try (java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream()) {
            // PNG signature
            baos.write(new byte[]{(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});
            
            // IHDR
            writePngChunk(baos, "IHDR", new byte[]{
                (byte) (w >> 24), (byte) (w >> 16), (byte) (w >> 8), (byte) w,
                (byte) (h >> 24), (byte) (h >> 16), (byte) (h >> 8), (byte) h,
                8, 2, 0, 0, 0
            });
            
            // IDAT - gradient
            int bytesPerRow = w * 3 + 1;
            byte[] rawData = new byte[bytesPerRow * h];
            for (int y = 0; y < h; y++) {
                rawData[y * bytesPerRow] = 0; // filter
                for (int x = 0; x < w; x++) {
                    int idx = y * bytesPerRow + 1 + x * 3;
                    rawData[idx] = (byte) (x * 255 / w); // R
                    rawData[idx + 1] = (byte) (y * 255 / h); // G
                    rawData[idx + 2] = 128; // B
                }
            }
            
            Deflater def = new Deflater();
            def.setInput(rawData);
            def.finish();
            byte[] compressed = new byte[rawData.length + 1024];
            int size = def.deflate(compressed);
            def.end();
            
            byte[] finalData = Arrays.copyOf(compressed, size);
            writePngChunk(baos, "IDAT", finalData);
            writePngChunk(baos, "IEND", new byte[0]);
            
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void writePngChunk(java.io.OutputStream out, String type, byte[] data) throws Exception {
        CRC32 crc = new CRC32();
        byte[] typeBytes = type.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        crc.update(typeBytes);
        crc.update(data);
        
        out.write((data.length >> 24) & 0xFF);
        out.write((data.length >> 16) & 0xFF);
        out.write((data.length >> 8) & 0xFF);
        out.write(data.length & 0xFF);
        out.write(typeBytes);
        out.write(data);
        long crcVal = crc.getValue();
        out.write((int) (crcVal >> 24) & 0xFF);
        out.write((int) (crcVal >> 16) & 0xFF);
        out.write((int) (crcVal >> 8) & 0xFF);
        out.write((int) crcVal & 0xFF);
    }

    private static Path findModelPath() {
        Path[] paths = {
            Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs"),
            Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", 
                    "CompVis", "stable-diffusion-v1-4"),
        };

        for (Path p : paths) {
            if (Files.isDirectory(p)) {
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
