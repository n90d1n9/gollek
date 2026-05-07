package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

@Dependent
@Unremovable
@Command(name = "embed", description = "Generate text embeddings using a specified model")
public class EmbedCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID to use for embeddings", required = true)
    public String modelId;

    @Option(names = { "-p", "--prompt", "--text" }, description = "Input text to embed", required = true)
    public String text;

    @Override
    public void run() {
        try {
            System.out.println("Generating embedding for model: " + modelId);
            EmbeddingResponse response = sdk.embed(modelId, text);
            
            System.out.println("Embedding Result (Count: " + response.embeddings().size() + "):");
            if (!response.embeddings().isEmpty()) {
                System.out.println("Dimension: " + response.dimension());
            }
            
        } catch (Exception e) {
            System.err.println("Embedding failed: " + e.getMessage());
        }
    }
}
