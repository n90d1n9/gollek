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
 * 🎨 Vision Storyteller — Turn Images into Narratives
 * 
 * This example demonstrates the Vision-to-Text capabilities of the Gollek SDK.
 * It uses the 'llava-v1.6' model (or equivalent) to describe an image and 
 * then expands that description into a creative short story.
 *
 * Use Cases:
 * - Automated content generation for social media
 * - Accessibility (describing scenes for the visually impaired)
 * - Creative writing assistance
 *
 * Usage:
 * jbang vision_storyteller.java [image_path]
 */
public class vision_storyteller {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║            🌌 Gollek Vision Storyteller                  ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        String imagePathStr;
        if (args.length > 0) {
            imagePathStr = args[0];
        } else {
            System.out.print("📂 Enter image path (e.g., sample.jpg): ");
            try (Scanner scanner = new Scanner(System.in)) {
                imagePathStr = scanner.nextLine();
            }
        }

        Path imagePath = Path.of(imagePathStr);
        if (!Files.exists(imagePath)) {
            System.err.println("❌ Error: File not found at " + imagePath.toAbsolutePath());
            System.out.println("💡 Usage: jbang vision_storyteller.java <path_to_image>");
            System.exit(1);
        }

        try {
            System.out.println("\n👁️ Analyzing image...");
            
            // Step 1: Use Vision Builder to get image description
            MultimodalResult descriptionResult = Gollek.vision("llava-v1.6")
                    .image(imagePath)
                    .prompt("Provide a detailed description of this image, focusing on the mood, subjects, and setting.")
                    .maxTokens(256)
                    .temperature(0.2f) // Lower temperature for accurate description
                    .generate();

            String description = descriptionResult.text();
            System.out.println("\n📝 Image Description:");
            System.out.println("----------------------------------------------------------");
            System.out.println(description);
            System.out.println("----------------------------------------------------------");

            System.out.println("\n📖 Generating story based on the scene...");

            // Step 2: Use Vision Builder again (or Multimodal) to generate a story
            // We can pass the description back or just ask for a story directly with the image
            MultimodalResult storyResult = Gollek.vision("llava-v1.6")
                    .image(imagePath)
                    .prompt("Based on this image, write a captivating 3-paragraph story. Make it atmospheric and focused on the hidden details.")
                    .maxTokens(512)
                    .temperature(0.8f) // Higher temperature for more creativity
                    .generate();

            System.out.println("\n✨ The Story:");
            System.out.println("==========================================================");
            System.out.println(storyResult.text());
            System.out.println("==========================================================");

            System.out.printf("%n📊 Stats: %d tokens | %dms latency%n", 
                    storyResult.totalTokens(), storyResult.durationMs());

        } catch (Exception e) {
            System.err.println("❌ Error during vision processing: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
