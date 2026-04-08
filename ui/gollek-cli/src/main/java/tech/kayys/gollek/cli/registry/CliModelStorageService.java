package tech.kayys.gollek.cli.registry;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.spi.storage.ModelStorageService;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Minimal local storage service for CLI native builds.
 * Ensures CDI wiring is satisfied without requiring external plugins.
 */
@ApplicationScoped
public class CliModelStorageService implements ModelStorageService {

    private static final Logger log = LoggerFactory.getLogger(CliModelStorageService.class);
    private static final String STORAGE_ENV = "GOLLEK_REGISTRY_STORAGE_DIR";
    private static final Path DEFAULT_BASE = Paths.get(System.getProperty("user.home"),
            ".gollek", "registry-storage");

    private Path resolveBase() {
        String override = System.getenv(STORAGE_ENV);
        if (override != null && !override.isBlank()) {
            return Paths.get(override);
        }
        return DEFAULT_BASE;
    }

    private Path resolveModelPath(String apiKey, String modelId, String version) {
        return resolveBase()
                .resolve(sanitize(apiKey))
                .resolve(sanitize(modelId))
                .resolve(sanitize(version))
                .resolve("model.bin");
    }

    private String sanitize(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        return value.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    @Override
    public Uni<String> uploadModel(String apiKey, String modelId, String version, byte[] data) {
        return Uni.createFrom().item(() -> {
            Path target = resolveModelPath(apiKey, modelId, version);
            try {
                Files.createDirectories(target.getParent());
                Files.write(target, data);
            } catch (IOException e) {
                throw new RuntimeException("Failed to store model artifact", e);
            }
            log.debug("Stored model artifact at {}", target);
            return target.toUri().toString();
        });
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            Path path = resolvePath(storageUri);
            try {
                return Files.readAllBytes(path);
            } catch (IOException e) {
                throw new RuntimeException("Failed to read model artifact", e);
            }
        });
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            Path path = resolvePath(storageUri);
            try {
                Files.deleteIfExists(path);
            } catch (IOException e) {
                log.warn("Failed to delete model artifact at {}", path, e);
            }
            return null;
        });
    }

    private Path resolvePath(String storageUri) {
        if (storageUri == null || storageUri.isBlank()) {
            throw new IllegalArgumentException("storageUri is required");
        }
        URI uri = URI.create(storageUri);
        if (uri.getScheme() == null || "file".equalsIgnoreCase(uri.getScheme())) {
            return uri.getScheme() == null ? Paths.get(storageUri) : Paths.get(uri);
        }
        throw new IllegalArgumentException("Unsupported storage URI: " + storageUri);
    }
}
