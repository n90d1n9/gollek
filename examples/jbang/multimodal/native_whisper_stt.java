//REPOS mavenLocal,central
//DEPS tech.kayys.gollek:gollek-audio:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-cnn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-transformer:0.1.0-SNAPSHOT

// Include model sources directly to avoid transitive framework dependencies (Quarkus)
//SOURCES ../../../framework/lib/gollek-ml-models/src/main/java/tech/kayys/gollek/ml/models/Whisper.java
//SOURCES ../../../framework/lib/gollek-ml-models/src/main/java/tech/kayys/gollek/ml/models/WhisperPipeline.java

//JAVA 25
//JAVA_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.audio.AudioBuffer;
import tech.kayys.gollek.ml.audio.MelSpectrogram;
import tech.kayys.gollek.ml.models.Whisper;
import tech.kayys.gollek.ml.models.WhisperPipeline;

import java.util.List;
import java.util.Random;

/**
 * 🎙️ Native Whisper STT — SIMD Accelerated Speech-to-Text
 * 
 * This example demonstrates the pure Java implementation of the Whisper
 * pipeline using Vector API and FFM API for high-performance audio processing.
 *
 * Usage:
 * jbang native_whisper_stt.java
 */
public class native_whisper_stt {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║         🗣️ Gollek Native Whisper STT Pipeline            ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        // 1. Initialize Whisper Model (Tiny configuration)
        System.out.println("🏗️ Initializing native Whisper architecture...");
        Whisper.WhisperConfig config = Whisper.WhisperConfig.tiny();
        Whisper model = new Whisper(config);

        // 2. Initialize Audio Processor (Mel Spectrogram)
        // Standard Whisper params: 16kHz, nFft=512 (padded), winLen=400, hop=160,
        // mels=80
        System.out.println("🎹 Configuring Mel-Spectrogram processor (SIMD accelerated)...");
        MelSpectrogram processor = new MelSpectrogram(16000, 512, 400, 160, 80);

        // 3. Create Pipeline
        WhisperPipeline pipeline = new WhisperPipeline(model, processor, config);
        config.maxTargetPos = 20; // Cap for demo performance

        // 4. Create Mock Audio (1 second of white noise)
        System.out.println("🔊 Generating 1 second of sample audio...");
        float[] samples = new float[16000];
        Random rng = new Random();
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (rng.nextFloat() * 2 - 1) * 0.1f; // low volume noise
        }

        // Wrap in SIMD-optimized AudioBuffer
        AudioBuffer audio = new AudioBuffer(samples, 16000);

        System.out.println("⚡ Normalizing audio using Vector API...");
        audio.normalize();

        // 5. Transcribe
        System.out.println("🚀 Transcribing (Greedy Decoding)...");
        long start = System.currentTimeMillis();
        List<Integer> tokenIds = pipeline.transcribe(audio);
        long end = System.currentTimeMillis();

        System.out.println("\n📃 Transcription Result (Token IDs):");
        System.out.println("----------------------------------------------------------");
        System.out.println(tokenIds);
        System.out.println("----------------------------------------------------------");

        System.out.printf("%n📊 Processing time: %dms%n", (end - start));
        System.out.println("✨ Success! Native STT pipeline executed successfully.");
    }
}
