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
 * 🎥 Video Analyst — Temporal Scene Understanding
 * 
 * This example demonstrates the Video-to-Text capabilities of the Gollek SDK.
 * It processes a video clip to identify key events, objects, and transitions 
 * across the timeline.
 *
 * Use Cases:
 * - Security footage summarization
 * - Sports highlight extraction
 * - Video content indexing for search
 *
 * Usage:
 * jbang video_analyst.java [video_path]
 */
public class video_analyst {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║            🎬 Gollek Video Analyst                       ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        String videoPathStr;
        if (args.length > 0) {
            videoPathStr = args[0];
        } else {
            System.out.print("🎞️ Enter video file path (e.g., sample.mp4): ");
            try (Scanner scanner = new Scanner(System.in)) {
                videoPathStr = scanner.nextLine();
            }
        }

        Path videoPath = Path.of(videoPathStr);
        if (!Files.exists(videoPath)) {
            System.err.println("❌ Error: File not found at " + videoPath.toAbsolutePath());
            System.out.println("💡 Usage: jbang video_analyst.java <path_to_video>");
            System.exit(1);
        }

        try {
            System.out.printf("%n📽️ Analyzing video: %s...%n", videoPath.getFileName());
            
            // Step 1: Use Video Builder for temporal understanding
            // The SDK handles frame extraction and sequence batching automatically.
            MultimodalResult vgaResult = Gollek.video("gemini-2.0-flash")
                    .videoFile(videoPath)
                    .prompt("Give me a second-by-second summary of what is happening in this video. Identify characters and objects.")
                    .generate();

            System.out.println("\n🎞️ Video Timeline Analysis:");
            System.out.println("----------------------------------------------------------");
            System.out.println(vgaResult.text());
            System.out.println("----------------------------------------------------------");

            System.out.println("\n🏷️ Extracting main sequence labels...");

            // Use another model to format the timeline into clean tags
            MultimodalResult tagsResult = Gollek.multimodal("gpt-4o")
                    .text("Extract 5 comma-separated action keywords from this description:\n\n" + vgaResult.text())
                    .generate();

            System.out.println("\n📌 Video Action Tags:");
            System.out.println("----------------------------------------------------------");
            System.out.println(tagsResult.text());
            System.out.println("----------------------------------------------------------");

            System.out.printf("%n📊 Analysis Complete: %dms | Model: %s%n", 
                    vgaResult.durationMs(), vgaResult.model());

        } catch (Exception e) {
            System.err.println("❌ Error during video analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
