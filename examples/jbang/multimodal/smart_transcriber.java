//DEPS tech.kayys.gollek:gollek-sdk-ml:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-multimodal:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runtime-tensor:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-model:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-tensor:0.1.0-SNAPSHOT
//DEPS com.fasterxml.jackson.core:jackson-databind:2.15.2
//DEPS io.smallrye.reactive:mutiny:2.5.1
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.sdk.multimodal.MultimodalResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * 🎙️ Smart Transcriber — Professional Audio Transcription
 * 
 * This example uses Whisper or Gemini audio models via the Gollek SDK
 * to transcribe audio files and provide a clean, punctuated transcript.
 *
 * Use Cases:
 * - Meeting minutes generation
 * - Podcast transcription
 * - Voice memo archiving
 *
 * Usage:
 * jbang smart_transcriber.java [audio_path]
 */
public class smart_transcriber {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║            🔊 Gollek Smart Transcriber                   ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        String audioPathStr;
        if (args.length > 0) {
            audioPathStr = args[0];
        } else {
            System.out.print("🎵 Enter audio file path (e.g., meeting.wav or speech.mp3): ");
            try (Scanner scanner = new Scanner(System.in)) {
                audioPathStr = scanner.nextLine();
            }
        }

        Path audioPath = Path.of(audioPathStr);
        if (!Files.exists(audioPath)) {
            System.err.println("❌ Error: File not found at " + audioPath.toAbsolutePath());
            System.out.println("💡 Usage: jbang smart_transcriber.java <path_to_audio>");
            System.exit(1);
        }

        try {
            System.out.printf("%n🎧 Transcribing: %s...%n", audioPath.getFileName());
            
            // Step 1: Use Audio Builder for high-fidelity transcription
            MultimodalResult transcriptResult = Gollek.audio("whisper-large-v3")
                    .audioFile(audioPath)
                    .task("transcribe")
                    .language("en")  // Optional: override language
                    .generate();

            System.out.println("\n📃 Full Transcript:");
            System.out.println("----------------------------------------------------------");
            System.out.println(transcriptResult.text());
            System.out.println("----------------------------------------------------------");

            System.out.println("\n✨ Extracting key highlights...");

            // Step 2: Use Multimodal/Text capability to summarize the transcript
            // This demonstrates pipelining different model tasks
            MultimodalResult summaryResult = Gollek.multimodal("gpt-4o-mini")
                    .text("Summarize the following transcript into 5 bullet points focused on key actions and decisions:\n\n" + transcriptResult.text())
                    .maxTokens(200)
                    .generate();

            System.out.println("\n📍 Key Points:");
            System.out.println("==========================================================");
            System.out.println(summaryResult.text());
            System.out.println("==========================================================");

            System.out.printf("%n📊 Processing time: %dms | Audio segments: %d%n", 
                    transcriptResult.durationMs(), transcriptResult.metadata().getOrDefault("segments", 0));

        } catch (Exception e) {
            System.err.println("❌ Error during audio transcription: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
