package tech.kayys.gollek.sdk.hub;

import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.ml.nn.StateDict;

import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.Map;

/**
 * Model Hub — downloads and caches pre-trained models from HuggingFace Hub
 * or a local cache directory.
 *
 * <p>Models are cached in {@code ~/.gollek/models/} by default.
 * Subsequent calls with the same model ID return the cached version without
 * re-downloading.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Load weights into an existing model
 * Module model = ResNet.resnet18(1000);
 * ModelHub.loadWeights(model, "kayys/resnet18-imagenet");
 *
 * // Download a SafeTensors file
 * Path weights = ModelHub.download("bert-base-uncased", "model.safetensors");
 *
 * // Load state dict directly
 * Map<String, GradTensor> state = ModelHub.loadStateDict("bert-base-uncased");
 * }</pre>
 */
public final class ModelHub {

    /** Default local cache directory: {@code ~/.gollek/models/}. */
    public static final Path DEFAULT_CACHE = Path.of(System.getProperty("user.home"), ".gollek", "models");

    /** HuggingFace Hub base URL. */
    private static final String HF_BASE = "https://huggingface.co";

    private static Path cacheDir = DEFAULT_CACHE;

    private ModelHub() {}

    /**
     * Sets a custom cache directory for downloaded models.
     *
     * @param dir path to the cache directory (created if it does not exist)
     */
    public static void setCacheDir(Path dir) { cacheDir = dir; }

    /**
     * Returns the current cache directory.
     *
     * @return cache directory path
     */
    public static Path getCacheDir() { return cacheDir; }

    /**
     * Downloads a specific file from a HuggingFace model repository.
     *
     * <p>If the file is already cached locally, the cached version is returned
     * without making a network request.
     *
     * @param modelId  HuggingFace model ID, e.g. {@code "bert-base-uncased"}
     * @param filename filename within the repository, e.g. {@code "model.safetensors"}
     * @return local path to the downloaded (or cached) file
     * @throws IOException if the download fails
     */
    public static Path download(String modelId, String filename) throws IOException {
        Path localDir  = cacheDir.resolve(modelId.replace("/", "--"));
        Path localFile = localDir.resolve(filename);

        if (Files.exists(localFile)) return localFile; // cache hit

        Files.createDirectories(localDir);
        String url = HF_BASE + "/" + modelId + "/resolve/main/" + filename;

        HttpClient http = HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.ALWAYS)
            .connectTimeout(Duration.ofSeconds(30))
            .build();

        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofMinutes(10))
            .GET().build();

        try {
            HttpResponse<Path> resp = http.send(req,
                HttpResponse.BodyHandlers.ofFile(localFile));
            if (resp.statusCode() != 200) {
                Files.deleteIfExists(localFile);
                throw new IOException("HuggingFace Hub returned HTTP " + resp.statusCode() + " for " + url);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Download interrupted", e);
        }
        return localFile;
    }

    /**
     * Downloads and loads a SafeTensors state dict from HuggingFace Hub.
     *
     * @param modelId HuggingFace model ID
     * @return state dict map of parameter name → tensor
     * @throws IOException if download or parsing fails
     */
    public static Map<String, tech.kayys.gollek.ml.autograd.GradTensor> loadStateDict(
            String modelId) throws IOException {
        Path file = download(modelId, "model.safetensors");
        return StateDict.load(file);
    }

    /**
     * Downloads weights from HuggingFace Hub and loads them into an existing model.
     *
     * <p>Uses non-strict loading ({@code strict=false}) so partial weight loading
     * works for fine-tuning scenarios.
     *
     * @param model   target model to load weights into
     * @param modelId HuggingFace model ID
     * @throws IOException if download or loading fails
     */
    public static void loadWeights(Module model, String modelId) throws IOException {
        Map<String, tech.kayys.gollek.ml.autograd.GradTensor> state = loadStateDict(modelId);
        model.loadStateDict(state, false);
    }

    /**
     * Checks whether a model is already cached locally.
     *
     * @param modelId  HuggingFace model ID
     * @param filename filename to check
     * @return {@code true} if the file exists in the local cache
     */
    public static boolean isCached(String modelId, String filename) {
        return Files.exists(cacheDir.resolve(modelId.replace("/", "--")).resolve(filename));
    }

    /**
     * Clears the local cache for a specific model.
     *
     * @param modelId HuggingFace model ID
     * @throws IOException if deletion fails
     */
    public static void clearCache(String modelId) throws IOException {
        Path dir = cacheDir.resolve(modelId.replace("/", "--"));
        if (Files.exists(dir)) {
            try (var walk = Files.walk(dir)) {
                walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> { try { Files.delete(p); } catch (IOException ignored) {} });
            }
        }
    }
}
