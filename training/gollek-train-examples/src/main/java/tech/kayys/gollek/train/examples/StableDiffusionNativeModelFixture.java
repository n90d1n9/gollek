package tech.kayys.gollek.train.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves and validates the repository layout expected by the safetensor-
 * native Stable Diffusion runner.
 */
public final class StableDiffusionNativeModelFixture {
    private static final String ENV_MODEL_DIR = "GOLLEK_SD_NATIVE_MODEL_DIR";

    private final Path baseDir;
    private final Map<String, Path> requiredPaths;
    private final List<String> missingComponents;

    private StableDiffusionNativeModelFixture(
            Path baseDir,
            Map<String, Path> requiredPaths,
            List<String> missingComponents) {
        this.baseDir = baseDir;
        this.requiredPaths = Map.copyOf(requiredPaths);
        this.missingComponents = List.copyOf(missingComponents);
    }

    public static StableDiffusionNativeModelFixture detect() {
        String envPath = System.getenv(ENV_MODEL_DIR);
        if (envPath != null && !envPath.isBlank()) {
            return fromBaseDir(Path.of(envPath));
        }

        List<Path> candidates = List.of(
                Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", "CompVis", "stable-diffusion-v1-4"),
                Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", "stable-diffusion-v1-4"),
                Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs", "CompVis", "stable-diffusion-v1-4"));
        for (Path candidate : candidates) {
            if (Files.isDirectory(candidate)) {
                return fromBaseDir(candidate);
            }
        }
        return fromBaseDir(candidates.get(0));
    }

    public static StableDiffusionNativeModelFixture fromBaseDir(Path baseDir) {
        Objects.requireNonNull(baseDir, "baseDir must not be null");
        Path normalized = baseDir.toAbsolutePath().normalize();
        Map<String, Path> required = new LinkedHashMap<>();
        required.put("textEncoderDir", normalized.resolve("text_encoder"));
        required.put("unetDir", normalized.resolve("unet"));
        required.put("vaeDir", normalized.resolve("vae"));
        required.put("tokenizerDir", normalized.resolve("tokenizer"));

        List<String> missing = new ArrayList<>();
        required.forEach((key, path) -> {
            if (!Files.isDirectory(path)) {
                missing.add(key + "=" + path);
            }
        });

        return new StableDiffusionNativeModelFixture(normalized, required, missing);
    }

    public Path baseDir() {
        return baseDir;
    }

    public boolean isUsable() {
        return missingComponents.isEmpty();
    }

    public List<String> missingComponents() {
        return missingComponents;
    }

    public String summary() {
        if (isUsable()) {
            return "real-fixture-ready baseDir=" + baseDir;
        }
        return "stub-fallback missing=" + missingComponents + " baseDir=" + baseDir;
    }

    public String overrideHint() {
        return "Set " + ENV_MODEL_DIR + " to a Stable Diffusion repo containing text_encoder/, unet/, vae/, and tokenizer/.";
    }

    public Map<String, Path> requiredPaths() {
        return requiredPaths;
    }
}
