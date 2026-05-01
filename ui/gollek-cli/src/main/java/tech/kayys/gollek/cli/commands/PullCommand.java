package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.sdk.model.ModelPullRequest;
import tech.kayys.gollek.sdk.core.GollekSdk;

/**
 * Pull model using GollekSdk.
 * Usage: gollek pull <model-spec>
 */
@Dependent
@Unremovable
@Command(name = "pull", description = "Pull a model from a provider")
public class PullCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Parameters(index = "0", description = "Model name to pull (e.g. llama3, hf:user/repo)")
    public String modelSpec;

    @Option(names = { "--insecure" }, description = "Allow insecure connections", defaultValue = "false")
    public boolean insecure;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Override
    public void run() {
        try {
            boolean convert = !"off".equalsIgnoreCase(convertMode);
            
            ModelPullRequest request = ModelPullRequest.builder()
                    .modelSpec(modelSpec)
                    .convertIfNecessary(convert)
                    .quantization(ggufOutType)
                    .outType(ggufOutType)
                    .build();

            System.out.println("Pulling model: " + modelSpec);
            System.out.println();

            sdk.pullModel(request, progress -> {
                if (progress.getTotal() > 0) {
                    String bar = progress.getProgressBar(30);
                    System.out.printf("\r%s [%s] %3d%% (%d/%d MB)",
                            progress.getStatus(),
                            bar,
                            progress.getPercentComplete(),
                            progress.getCompleted() / 1024 / 1024,
                            progress.getTotal() / 1024 / 1024);
                } else {
                    System.out.printf("\r%s...", progress.getStatus());
                }
            });

            System.out.println("\nPull complete: " + modelSpec);

        } catch (Exception e) {
            System.err.println("\nFailed to pull model: " + e.getMessage());
            if (e.getCause() != null) {
                System.err.println("Detail: " + e.getCause().getMessage());
            }
        }
    }
}
