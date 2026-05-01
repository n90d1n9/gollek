


import io.llamajava.gguf.GGUFFile;
import io.llamajava.gguf.GGUFLoader;
import io.llamajava.inference.KVCache;
import io.llamajava.model.ModelConfig;
import io.llamajava.model.LlamaWeights;
import io.llamajava.model.LlamaForward;
import io.llamajava.tokenizer.BPETokenizer;
import io.llamajava.sampler.Sampler;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java --enable-preview --add-modules=jdk.incubator.vector -jar llamajava.jar <model.gguf>");
            System.exit(1);
        }
        
        try {
            // Load model
            System.out.println("Loading " + args[0] + "...");
            GGUFFile gguf = GGUFLoader.load(Path.of(args[0]));
            ModelConfig config = ModelConfig.fromGGUF(gguf);
            BPETokenizer tokenizer = BPETokenizer.load(gguf);
            LlamaWeights weights = LlamaWeights.load(gguf, config);
            
            // Create inference components
            LlamaForward forward = new LlamaForward(config, weights);
            KVCache cache = new KVCache(config.nLayers(), config.contextLength(), 
                                        config.nKVHeads(), config.headDim());
            Sampler sampler = new Sampler(0.8f, 40, 0.95f, 1.1f);
            
            // Interactive loop
            System.out.println("\nModel ready. Type 'quit' to exit.\n");
            java.util.Scanner scanner = new java.util.Scanner(System.in);
            float[] hiddenState = new float[config.embeddingDim()];
            
            while (true) {
                System.out.print("> ");
                String prompt = scanner.nextLine();
                if (prompt.equals("quit")) break;
                
                // Encode prompt
                int[] tokens = tokenizer.encode(prompt, true, false);
                
                // Process prompt
                for (int i = 0; i < tokens.length; i++) {
                    forward.forward(tokens[i], i, cache, hiddenState);
                }
                cache.advance();
                
                // Generate response
                List<Integer> generated = new ArrayList<>();
                int pos = tokens.length;
                for (int step = 0; step < 200; step++) {
                    int lastToken = step == 0 ? tokens[tokens.length-1] : generated.get(generated.size()-1);
                    float[] logits = forward.forward(lastToken, pos + step, cache, hiddenState);
                    int nextToken = sampler.sample(logits, generated);
                    if (nextToken == tokenizer.eosId) break;
                    generated.add(nextToken);
                    System.out.print(tokenizer.decode(new int[]{nextToken}));
                    System.out.flush();
                }
                System.out.println("\n");
                cache.reset(); // Reset for next prompt
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
