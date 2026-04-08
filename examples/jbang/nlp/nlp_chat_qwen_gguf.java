//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-api-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-inference-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-ext-runner-gguf-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-engine-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-provider-core-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-provider-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-runtime-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-error-code-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/mutiny-3.1.1.jar
//DEPS ${user.home}/.gollek/jbang/libs/reactive-streams-1.0.4.jar
//DEPS ${user.home}/.gollek/jbang/libs/jboss-logging-3.6.2.Final.jar
//DEPS ${user.home}/.gollek/jbang/libs/jackson-databind-2.15.2.jar
//DEPS ${user.home}/.gollek/jbang/libs/jackson-core-2.15.2.jar
//DEPS ${user.home}/.gollek/jbang/libs/jinjava-2.7.1.jar
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.inference.gguf.*;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.*;
import tech.kayys.gollek.spi.model.ModelManifest;
import io.smallrye.mutiny.Multi;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Chat with LLM Example (Qwen 0.5B GGUF).
 * Native Version: Powered by LibTorch + llama.cpp via Gollek SDK.
 */
public class nlp_chat_qwen_gguf {

    private static final String MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║       Gollek LLM Chat — Qwen 2.5 0.5B (GGUF)       ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        Path modelPath = Path.of("models/qwen2.5-0.5b-instruct-q4_0.gguf");
        ensureModel(modelPath);

        // ── Initialize Native Runner ──
        System.out.println("\n🚀 Initializing llama.cpp backend...");
        
        // Setup minimal config for JBang
        var config = new GGUFProviderConfig(); 
        var binding = new LlamaCppBinding();
        binding.backendInit();
        
        var runner = new LlamaCppRunner(binding, config, new GGUFChatTemplateService());
        
        var manifest = ModelManifest.builder()
                .modelId("qwen2.5-0.5b")
                .requestId(UUID.randomUUID().toString())
                .build();
                
        runner.initialize(manifest, Map.of("model_path", modelPath.toString()));
        System.out.println("✅ Model loaded successfully.");

        // ── Chat State ──
        List<Message> history = new ArrayList<>();
        history.add(Message.system("You are a helpful and concise AI assistant."));
        
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nAssistant > My context window is ready. Ask me anything! (type 'exit' to quit)");

        while (true) {
            System.out.print("\n\u001B[34mUser > \u001B[0m");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();
            if (input == null || input.equalsIgnoreCase("exit")) break;
            if (input.isBlank()) continue;

            history.add(Message.user(input));

            // ── Construct Request ──
            var request = InferenceRequest.builder()
                    .model("qwen2.5-0.5b")
                    .messages(history)
                    .parameter("max_tokens", 512)
                    .parameter("temperature", 0.7f)
                    .parameter("stop", List.of("<|im_end|>", "<|endoftext|>", "</s>"))
                    .build();

            System.out.print("\u001B[32mAssistant > \u001B[0m");
            
            Instant t0 = Instant.now();
            StringBuilder fullResponse = new StringBuilder();
            AtomicBoolean firstToken = new AtomicBoolean(true);
            long[] ttft = {0};

            // Stream and print
            runner.inferStream(request).subscribe().with(
                chunk -> {
                    if (firstToken.compareAndSet(true, false)) {
                        ttft[0] = Duration.between(t0, Instant.now()).toMillis();
                    }
                    String delta = chunk.delta();
                    System.out.print(delta);
                    System.out.flush();
                    fullResponse.append(delta);
                },
                failure -> System.err.println("\n❌ Error: " + failure.getMessage()),
                () -> {
                    long totalMs = Duration.between(t0, Instant.now()).toMillis();
                    int tokens = fullResponse.toString().split("\\s+").length; // Rough word count
                    float tps = tokens / (totalMs / 1000.0f);
                    
                    System.out.printf("\n\n\u001B[90m[Stats: %dms TTFT | %.1f tokens/sec | %dms total]\u001B[0m%n", 
                        ttft[0], tps, totalMs);
                    
                    history.add(Message.assistant(fullResponse.toString().trim()));
                }
            );
        }
        
        runner.close();
        System.out.println("\n👋 Chat session ended.");
    }

    private static void ensureModel(Path path) throws Exception {
        if (Files.exists(path)) return;
        System.out.println("Model not found at: " + path);
        System.out.println("Would you like to download Qwen 0.5B (350MB)? [y/N]");
        try (Scanner s = new Scanner(System.in)) {
            if (s.hasNextLine() && s.nextLine().equalsIgnoreCase("y")) {
                Files.createDirectories(path.getParent());
                System.out.println("Downloading model... this may take a few minutes...");
                new ProcessBuilder("curl", "-L", MODEL_URL, "-o", path.toString()).inheritIO().start().waitFor();
            } else {
                System.out.println("Please place model manually.");
                System.exit(1);
            }
        }
    }
}
