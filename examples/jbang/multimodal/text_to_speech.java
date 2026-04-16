///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-safetensor-engine:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-audio:0.1.0-SNAPSHOT
//JAVA 25

import tech.kayys.gollek.ml.audio.speech.SpeechSynthesizer;
import tech.kayys.gollek.ml.audio.AudioFormat;

import java.nio.file.Path;

/**
 * Text-to-Speech (TTS) Example using SpeechT5
 * 
 * Converts text to natural-sounding speech audio.
 * 
 * Usage:
 *   jbang text_to_speech.java
 *   jbang text_to_speech.java --text "Hello World" --output hello.wav
 */
public class text_to_speech {

    public static void main(String[] args) throws Exception {
        String text = getArg(args, "--text", "Hello! Welcome to Gollek, the Java-native AI platform.");
        String outputFile = getArg(args, "--output", "output.wav");

        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek Text-to-Speech - SpeechT5 Example            ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Text: " + text);
        System.out.println("Output: " + outputFile);
        System.out.println();

        // Initialize speech synthesizer
        System.out.println("⏳ Loading SpeechT5 model...");
        SpeechSynthesizer synthesizer = SpeechSynthesizer.builder()
                .model("microsoft/speecht5_tts")
                .format(AudioFormat.WAV)
                .build();

        System.out.println("⏳ Synthesizing speech...");
        long startTime = System.currentTimeMillis();

        // Synthesize and save to file
        Path outputPath = Path.of(outputFile);
        synthesizer.synthesizeToFile(text, outputPath);

        long duration = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.println("✅ Audio saved to: " + outputPath.toAbsolutePath());
        System.out.printf("⏱️  Synthesis time: %.1f seconds%n", duration / 1000.0);

        // Auto-play on macOS
        if (System.getProperty("os.name").toLowerCase().contains("mac")) {
            System.out.println("🔊 Playing audio...");
            Runtime.getRuntime().exec(new String[]{"afplay", outputPath.toAbsolutePath().toString()});
        }
    }

    private static String getArg(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) return args[i + 1];
        }
        return defaultValue;
    }
}
