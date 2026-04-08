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
 * 🤖 Omni Assistant — Advanced Mixed-Modality Reasoning
 * 
 * This example showcases the full power of the Gollek Multimodal SDK 
 * by combining multiple inputs (Image + Audio + Text) into a single 
 * reasoning context.
 *
 * Use Cases:
 * - Complex evidence analysis (Comparing a photo with a voice note)
 * - Interactive multi-sensory tutoring
 * - High-level decision support with sensory data
 *
 * Usage:
 * jbang omni_assistant.java [image_path] [audio_path] [prompt]
 */
public class omni_assistant {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║            🤖 Gollek Omni Assistant                      ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        String imagePathStr, audioPathStr, prompt;

        if (args.length >= 3) {
            imagePathStr = args[0];
            audioPathStr = args[1];
            prompt = args[2];
        } else {
            try (Scanner scanner = new Scanner(System.in)) {
                System.out.print("🖼️ Enter image path: ");
                imagePathStr = scanner.nextLine();
                System.out.print("🔉 Enter audio path: ");
                audioPathStr = scanner.nextLine();
                System.out.print("💬 Enter your question: ");
                prompt = scanner.nextLine();
            }
        }

        Path imagePath = Path.of(imagePathStr);
        Path audioPath = Path.of(audioPathStr);

        if (!Files.exists(imagePath) || !Files.exists(audioPath)) {
            System.err.println("❌ Error: Files not found!");
            System.out.printf("   Image: %s%n   Audio: %s%n", 
                    imagePath.toAbsolutePath(), audioPath.toAbsolutePath());
            System.exit(1);
        }

        try {
            System.out.println("\n🧠 Reasoning across modalities...");
            
            // The magic happens here: combining visual and audio context with text
            MultimodalResult omniResult = Gollek.multimodal("gpt-4o")
                    .text(prompt)
                    .image(imagePath)
                    .audioFile(audioPath)
                    .maxTokens(400)
                    .generate();

            System.out.println("\n💡 Assistant Response:");
            System.out.println("==========================================================");
            System.out.println(omniResult.text());
            System.out.println("==========================================================");

            System.out.printf("%n📊 Tokens Used: %d (Input: %d, Output: %d)%n", 
                    omniResult.totalTokens(), omniResult.inputTokens(), omniResult.outputTokens());
            System.out.println("✅ Processed on " + omniResult.model());

        } catch (Exception e) {
            System.err.println("❌ Error during omni-reasoning: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
