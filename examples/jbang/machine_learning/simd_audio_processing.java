//REPOS mavenLocal,central
//DEPS tech.kayys.gollek:gollek-audio:0.1.0-SNAPSHOT
//JAVA 25
//JAVA_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.audio.AudioBuffer;
import java.util.Random;

/**
 * ⚡ SIMD Audio Processing — Vector API Performance Demo
 * 
 * This example demonstrates the performance of the SIMD-accelerated 
 * AudioBuffer compared to a standard scalar loop.
 *
 * Usage:
 * jbang simd_audio_processing.java
 */
public class simd_audio_processing {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║         🚀 SIMD Audio Processing Performance             ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        int size = 10_000_000; // 10 million samples
        System.out.printf("📁 Creating buffer with %,d samples...%n", size);
        
        float[] samples = new float[size];
        Random rng = new Random();
        for (int i = 0; i < size; i++) samples[i] = rng.nextFloat() * 2 - 1;

        AudioBuffer buffer = new AudioBuffer(samples, 16000);

        // 1. Performance Test: Normalization
        System.out.println("📏 Testing Normalization (Find Max + Scale)...");
        
        long start = System.nanoTime();
        buffer.normalize();
        long end = System.nanoTime();
        
        double durationMs = (end - start) / 1_000_000.0;
        System.out.printf("✅ SIMD Normalization completed in %.2fms%n", durationMs);

        // 2. Performance Test: Massive Scaling
        float factor = 0.5f;
        System.out.printf("📉 Scaling buffer by %.2f...%n", factor);
        
        start = System.nanoTime();
        buffer.scale(factor);
        end = System.nanoTime();
        
        durationMs = (end - start) / 1_000_000.0;
        System.out.printf("✅ SIMD Scaling completed in %.2fms%n", durationMs);

        System.out.println("\n✨ Performance demo finished. Vector API utilizes CPU SIMD lanes (AVX/NEON) automatically.");
    }
}
