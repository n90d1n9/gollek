package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.model.download.DownloadProgressListener;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceRepository;
import tech.kayys.gollek.sdk.model.ModelPullRequest;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.stream.Stream;

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

    @Inject
    Instance<HuggingFaceRepository> huggingFaceRepository;

    @Inject
    Instance<HuggingFaceClient> huggingFaceClient;

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
            String effectiveModelSpec = normalizeModelSpec(modelSpec);
            
            ModelPullRequest request = ModelPullRequest.builder()
                    .modelSpec(effectiveModelSpec)
                    .convertIfNecessary(convert)
                    .quantization(ggufOutType)
                    .outType(ggufOutType)
                    .build();

            System.out.println("Pulling model: " + effectiveModelSpec);
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

            if (!hasLocalSafetensorArtifacts(effectiveModelSpec)) {
                if (!tryHfRepositoryFallback(effectiveModelSpec, new RuntimeException("No local artifacts after SDK pull"))) {
                    throw new RuntimeException("Pull reported success, but no local model artifacts were found for " + effectiveModelSpec);
                }
            }

            System.out.println("\nPull complete: " + effectiveModelSpec);

        } catch (Exception e) {
            String effectiveModelSpec = normalizeModelSpec(modelSpec);
            if (tryHfRepositoryFallback(effectiveModelSpec, e)) {
                System.out.println("\nPull complete: " + effectiveModelSpec);
                return;
            }
            System.err.println("\nFailed to pull model: " + e.getMessage());
            if (e.getCause() != null) {
                System.err.println("Detail: " + e.getCause().getMessage());
            }
        }
    }

    private boolean tryHfRepositoryFallback(String effectiveModelSpec, Exception original) {
        if (effectiveModelSpec == null || !effectiveModelSpec.startsWith("hf:")) {
            return false;
        }
        String message = original.getMessage() == null ? "" : original.getMessage();
        if (!message.contains("requires pre-downloaded models")
                && !message.contains("No local artifacts after SDK pull")
                && !message.contains("Pull reported success")) {
            return false;
        }
        if (huggingFaceRepository == null || huggingFaceRepository.isUnsatisfied()) {
            return false;
        }

        String repoId = effectiveModelSpec.substring("hf:".length()).trim();
        if (repoId.isEmpty()) {
            return false;
        }

        System.out.println("Falling back to HuggingFace repository downloader...");
        var manifest = huggingFaceRepository.get()
                .findById(repoId, "community")
                .await()
                .atMost(Duration.ofMinutes(30));
        if (manifest == null) {
            // Some build profiles can resolve to null even when HF access is valid.
            // Fall back to direct HF client download into the local safetensor cache.
            if (huggingFaceClient == null || huggingFaceClient.isUnsatisfied()) {
                throw new RuntimeException("HuggingFace pull fallback returned no model manifest for " + repoId, original);
            }
            Path targetDir = Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", repoId);
            try {
                Files.createDirectories(targetDir);
                HuggingFaceClient client = huggingFaceClient.get();
                List<String> files = client.listFiles(repoId);
                try (HfProgressRenderer progress = new HfProgressRenderer(files.size())) {
                    client.downloadRepository(repoId, targetDir, progress);
                }
            } catch (Exception ex) {
                throw new RuntimeException("Direct HuggingFace download fallback failed for " + repoId, ex);
            }
        }
        return hasLocalSafetensorArtifacts(effectiveModelSpec);
    }

    private String normalizeModelSpec(String raw) {
        if (raw == null) {
            return null;
        }
        String trimmed = raw.trim();
        if (trimmed.startsWith("hf:") || trimmed.startsWith("huggingface:")) {
            return trimmed;
        }
        // Treat owner/repo as HuggingFace shorthand in CLI pull path.
        if (trimmed.contains("/") && !trimmed.contains("://")) {
            return "hf:" + trimmed;
        }
        return trimmed;
    }

    private boolean hasLocalSafetensorArtifacts(String effectiveModelSpec) {
        if (effectiveModelSpec == null || effectiveModelSpec.isBlank()) {
            return false;
        }
        String repoId = effectiveModelSpec.startsWith("hf:")
                ? effectiveModelSpec.substring("hf:".length()).trim()
                : effectiveModelSpec.trim();
        if (repoId.isBlank()) {
            return false;
        }

        Path base = Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors");
        Path direct = base.resolve(repoId);
        Path normalized = base.resolve(repoId.replace("/", "--"));
        return hasFiles(direct) || hasFiles(normalized);
    }

    private boolean hasFiles(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) {
            return false;
        }
        try (Stream<Path> stream = Files.walk(dir)) {
            return stream.anyMatch(Files::isRegularFile);
        } catch (Exception ignored) {
            return false;
        }
    }

    private static final class HfProgressRenderer implements DownloadProgressListener, AutoCloseable {
        private static final String[] SPINNER = { "|", "/", "-", "\\" };
        private static final String CYAN = "\u001B[36m";
        private static final String GREEN = "\u001B[32m";
        private static final String DIM = "\u001B[2m";
        private static final String RESET = "\u001B[0m";
        private static final int BAR_WIDTH = 34;
        private static final long MIN_REDRAW_NS = 70_000_000L;

        private final int totalFiles;
        private int completedFiles = 0;
        private int spinnerTick = 0;
        private long fileStartNanos = System.nanoTime();
        private long lastRedrawNanos = 0L;

        private HfProgressRenderer(int totalFiles) {
            this.totalFiles = Math.max(1, totalFiles);
        }

        @Override
        public synchronized void onProgress(long downloadedBytes, long totalBytes, double progress) {
            long now = System.nanoTime();
            if (now - lastRedrawNanos < MIN_REDRAW_NS && downloadedBytes < totalBytes) {
                return;
            }
            lastRedrawNanos = now;

            double pct = Math.max(0.0, Math.min(1.0, progress));
            int filled = (int) Math.round(BAR_WIDTH * pct);
            if (filled > BAR_WIDTH) {
                filled = BAR_WIDTH;
            }
            StringBuilder bar = new StringBuilder(BAR_WIDTH);
            for (int i = 0; i < BAR_WIDTH; i++) {
                bar.append(i < filled ? "=" : ".");
            }

            double elapsedSec = Math.max(0.001, (now - fileStartNanos) / 1_000_000_000.0);
            double mbDone = downloadedBytes / 1024.0 / 1024.0;
            double mbTotal = totalBytes > 0 ? totalBytes / 1024.0 / 1024.0 : 0.0;
            double speed = mbDone / elapsedSec;

            String spin = SPINNER[spinnerTick++ % SPINNER.length];
            String line = String.format(
                    "\r%s%s%s %s[%-34s]%s %3d%% %s%.1f/%.1f MB%s  %s%.1f MB/s%s  %sfile %d/%d%s",
                    CYAN, spin, RESET,
                    GREEN, bar, RESET,
                    (int) Math.round(pct * 100.0),
                    DIM, mbDone, mbTotal, RESET,
                    CYAN, speed, RESET,
                    DIM, Math.min(completedFiles + 1, totalFiles), totalFiles, RESET
            );
            System.out.print(line);
            System.out.flush();
        }

        @Override
        public synchronized void onComplete(long totalBytes) {
            completedFiles++;
            fileStartNanos = System.nanoTime();
            String line = String.format(
                    "\r%s%s%s %s[%-34s]%s %3d%% %sfile %d/%d done%s",
                    GREEN, "*", RESET,
                    GREEN, "==================================", RESET,
                    100,
                    DIM, Math.min(completedFiles, totalFiles), totalFiles, RESET
            );
            System.out.print(line);
            System.out.print("\n");
            System.out.flush();
        }

        @Override
        public void close() {
            System.out.flush();
        }
    }
}
