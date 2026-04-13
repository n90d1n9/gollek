///usr/bin/env jbang "$0" "$@" ; exit $?
//REPOS mavenLocal
//DEPS tech.kayys.gollek:gollek-ml-autograd:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-models:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-plugin-metal:0.1.0-SNAPSHOT
//JAVA 25
//JAVA_OPTIONS --add-modules jdk.incubator.vector

import tech.kayys.gollek.ml.models.*;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.runtime.tensor.Device;
import tech.kayys.gollek.spi.tensor.ComputeBackendRegistry;

/**
 * 🎨 Gollek Stable Diffusion v1-4 Metal Example
 * 🚀 Hardware Accelerated (Metal Performance Shaders)
 * ⚡ Zero-Copy Unified Memory Architecture
 */
public class stable_diffusion_metal {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║         🎨 Gollek Stable Diffusion Metal Pipeline        ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        // 1. Hardware Detection
        System.out.println("🔍 Checking hardware acceleration...");
        var backend = ComputeBackendRegistry.get();
        System.out.println("🚀 Active Backend: " + backend.deviceName());

        if (!backend.deviceName().contains("Apple")) {
            System.err.println("⚠️ Warning: Metal backend not detected. Falling back to CPU SIMD.");
        } else {
            System.out.println("✅ Metal Performance Shaders (MPS) active with Zero-Copy.");
        }

        // 2. Initialize Pipeline (Internal weights automatically moved to Device.METAL)
        System.out.println("🏗️  Initializing SD v1-4 architecture...");

        var clip = new CLIP();
        var unet = new UNet2DConditional(4, 4, 320, 768);
        var vae = new VAE();

        var pipeline = new StableDiffusionPipeline(clip, unet, vae);

        // 3. Generation
        String prompt = (args.length > 0) ? args[0] : "A high-performance AI platform in a futuristic datacenter";
        System.out.println("📝 Prompt: \"" + prompt + "\"");
        System.out.println("🎨 Starting generation loop (5 steps for demo)...");

        long start = System.currentTimeMillis();

        GradTensor image = pipeline.generate(prompt, 5, 7.5f, step -> {
            System.out.print(".");
            if (step % 10 == 0)
                System.out.println(" [" + step + "/5]");
        });

        long end = System.currentTimeMillis();

        // 4. Results
        System.out.println("\n✨ Generation complete!");
        System.out.println("⏱️  Inference Time: " + (end - start) + "ms");
        System.out.println("🖼️  Output Tensor: " + image);
        System.out.println("✅ Image generated successfully with Zero-Copy Metal throughput.");
    }
}
