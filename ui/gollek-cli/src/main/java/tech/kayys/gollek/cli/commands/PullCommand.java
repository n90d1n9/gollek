package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.model.download.DownloadProgressListener;
import tech.kayys.gollek.model.repo.local.GollekManifest;
import tech.kayys.gollek.model.repo.local.ManifestStore;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceRepository;
import tech.kayys.gollek.sdk.model.ModelPullRequest;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.util.GollekHome;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
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

    @Inject
    Instance<ManifestStore> manifestStore;

    @Parameters(index = "0", arity = "0..1", description = "Model name to pull (e.g. llama3, hf:user/repo)")
    public String modelSpec;

    @Option(names = { "--insecure" }, description = "Allow insecure connections", defaultValue = "false")
    public boolean insecure;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Option(names = { "-f", "--file" }, description = "Copy model from local file path to internal model repo")
    String file;

    @Option(names = { "-m", "--move" }, description = "Move model from local file path instead of copying")
    boolean move;

    @Override
    public void run() {
        try {
            boolean convert = !"off".equalsIgnoreCase(convertMode);
            String effectiveModelSpec = normalizeModelSpec(modelSpec);

            if (file != null && !file.isBlank()) {
                handleLocalFileCopy(effectiveModelSpec);
                return;
            }
            
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

            if (!hasLocalArtifacts(effectiveModelSpec)) {
                if (!tryHfRepositoryFallback(effectiveModelSpec, new RuntimeException("No local artifacts after SDK pull"))) {
                    throw new RuntimeException("Pull reported success, but no local model artifacts were found for " + effectiveModelSpec);
                }
            }

            System.out.println("\nPull complete: " + effectiveModelSpec);

        } catch (Exception e) {
            String effectiveModelSpec = normalizeModelSpec(modelSpec);
            try {
                if (tryHfRepositoryFallback(effectiveModelSpec, e)) {
                    System.out.println("\nPull complete: " + effectiveModelSpec);
                    return;
                }
            } catch (Exception fallbackEx) {
                System.err.println("\nFailed to pull model via fallback: " + fallbackEx.getMessage());
                if (fallbackEx.getCause() != null) {
                    System.err.println("Detail: " + fallbackEx.getCause().getMessage());
                }
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
                .findById(effectiveModelSpec, "community")
                .await()
                .atMost(Duration.ofMinutes(30));
        if (manifest == null) {
            // Some build profiles can resolve to null even when HF access is valid.
            // Fall back to a direct HF client download and register it in the local manifest store.
            if (huggingFaceClient == null || huggingFaceClient.isUnsatisfied()) {
                throw new RuntimeException("HuggingFace pull fallback returned no model manifest for " + repoId, original);
            }
            String manifestName = GollekManifest.computeName(repoId, "main");
            Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
            try {
                Files.createDirectories(targetDir);
                HuggingFaceClient client = huggingFaceClient.get();
                List<String> files = client.listFiles(repoId);
                try (HfProgressRenderer progress = new HfProgressRenderer(files.size())) {
                    client.downloadRepository(repoId, targetDir, progress);
                }
                saveDirectDownloadManifest(repoId, targetDir, files, manifestName);
                LocalModelIndex.refreshFromDisk();
            } catch (Exception ex) {
                throw new RuntimeException("Direct HuggingFace download fallback failed for " + repoId, ex);
            }
        }
        return hasLocalArtifacts(effectiveModelSpec);
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

    private void saveDirectDownloadManifest(String repoId, Path targetDir, List<String> files, String manifestName)
            throws Exception {
        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();

        String format = ManifestStore.detectFormat(targetDir);
        GollekManifest manifest = new GollekManifest();
        manifest.setId(manifestName);
        manifest.setModelId(repoId);
        manifest.setName(manifestName);
        manifest.setFormat(format);
        manifest.setPipeline(files != null && files.stream().anyMatch(name -> name.endsWith("model_index.json")));
        manifest.setSource("huggingface");
        manifest.setRepository(repoId);
        manifest.setBranch("main");
        manifest.setBlobPath(targetDir.toAbsolutePath().toString());
        manifest.setFiles(ManifestStore.listBlobFiles(targetDir));
        manifest.setCreatedAt(Instant.now());
        manifest.setSizeBytes(computeSize(targetDir));
        manifest.setArchitecture(ManifestStore.detectArchitecture(manifest));
        manifest.setMetadata(Map.of(
                "fallback", "direct-huggingface-repository",
                "format", format != null ? format : "unknown"));

        store.save(manifest);
    }

    private long computeSize(Path path) {
        if (path == null || !Files.exists(path)) {
            return 0L;
        }
        try {
            if (Files.isRegularFile(path)) {
                return Files.size(path);
            }
            try (Stream<Path> walk = Files.walk(path)) {
                return walk.filter(Files::isRegularFile)
                        .mapToLong(file -> {
                            try {
                                return Files.size(file);
                            } catch (Exception ignored) {
                                return 0L;
                            }
                        })
                        .sum();
            }
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private boolean hasLocalArtifacts(String effectiveModelSpec) {
        if (effectiveModelSpec == null || effectiveModelSpec.isBlank()) {
            return false;
        }
        String repoId = effectiveModelSpec.startsWith("hf:")
                ? effectiveModelSpec.substring("hf:".length()).trim()
                : effectiveModelSpec.trim();
        if (repoId.isBlank()) {
            return false;
        }

        if (LocalModelIndex.find(repoId).isPresent() || LocalModelIndex.find(effectiveModelSpec).isPresent()) {
            return true;
        }

        String normalized = repoId.replace("/", "--");
        String manifestName = GollekManifest.computeName(repoId, "main");
        List<Path> candidates = List.of(
                ManifestStore.resolveBlobDir(repoId, manifestName),
                GollekHome.path("models", "safetensors").resolve(repoId),
                GollekHome.path("models", "safetensors").resolve(normalized),
                GollekHome.path("models", "onnx").resolve(repoId),
                GollekHome.path("models", "onnx").resolve(normalized),
                GollekHome.path("models", "litert").resolve(repoId),
                GollekHome.path("models", "litert").resolve(normalized),
                GollekHome.path("models", "gguf").resolve(repoId),
                GollekHome.path("models", "gguf").resolve(normalized),
                GollekHome.path("models", "torchscript").resolve(repoId),
                GollekHome.path("models", "torchscript").resolve(normalized),
                GollekHome.path("models", "libtorchscript").resolve(repoId),
                GollekHome.path("models", "libtorchscript").resolve(normalized));
        return candidates.stream().anyMatch(this::hasFiles);
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

    private void handleLocalFileCopy(String effectiveModelSpec) throws Exception {
        Path sourcePath = Path.of(file);
        if (!Files.exists(sourcePath)) {
            throw new RuntimeException("Source file does not exist: " + file);
        }

        String repoId = effectiveModelSpec != null && effectiveModelSpec.startsWith("hf:")
                ? effectiveModelSpec.substring("hf:".length()).trim()
                : (effectiveModelSpec != null ? effectiveModelSpec.trim() : sourcePath.getFileName().toString());
        
        String manifestName = GollekManifest.computeName(repoId, "main");
        Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
        Files.createDirectories(targetDir);
        Path targetPath = targetDir.resolve(sourcePath.getFileName());

        if (move) {
            System.out.println("Moving " + sourcePath + " to " + targetPath + " ...");
            Files.move(sourcePath, targetPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } else {
            System.out.println("Copying " + sourcePath + " to " + targetPath + " ...");
            Files.copy(sourcePath, targetPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }

        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();
        store.registerLocal(repoId, targetPath);
        LocalModelIndex.refreshFromDisk();

        System.out.println("\nPull complete: " + repoId + " (imported from local file)");
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
