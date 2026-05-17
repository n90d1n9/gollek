package tech.kayys.gollek.ml.hub;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.model.core.ModelRepositoryProvider;
import tech.kayys.gollek.model.core.RepositoryContext;
import tech.kayys.gollek.model.local.LocalModelRepository;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceConfig;
import tech.kayys.gollek.model.repo.hf.HuggingFaceRepository;
import tech.kayys.gollek.model.repo.local.ManifestStore;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for resolving model repositories in standalone ML usage.
 */
public final class ModelHubFactory {

    private static final Map<String, ModelRepository> CACHE = new ConcurrentHashMap<>();

    private ModelHubFactory() {
    }

    public static ModelRepository getRepository(String scheme, HubConfig config) {
        String cacheKey = scheme + ":" + config.cacheDir();
        return CACHE.computeIfAbsent(cacheKey, ignored -> createRepository(scheme, config));
    }

    private static ModelRepository createRepository(String scheme, HubConfig config) {
        ServiceLoader<ModelRepositoryProvider> loader = ServiceLoader.load(ModelRepositoryProvider.class);
        for (ModelRepositoryProvider provider : loader) {
            if (provider.scheme().equalsIgnoreCase(scheme)) {
                return provider.create(toContext(config));
            }
        }

        return switch (scheme.toLowerCase()) {
            case "hf", "huggingface" -> createStandaloneHF(config);
            case "local" -> new LocalModelRepository(config.cacheDir().toString());
            default -> throw new IllegalArgumentException("Unsupported repository scheme: " + scheme);
        };
    }

    private static ModelRepository createStandaloneHF(HubConfig config) {
        HuggingFaceConfig hfConfig = new HuggingFaceConfig() {
            @Override
            public String baseUrl() {
                return "https://huggingface.co";
            }

            @Override
            public Optional<String> token() {
                return Optional.ofNullable(config.token());
            }

            @Override
            public int timeoutSeconds() {
                return config.timeoutSeconds();
            }

            @Override
            public int maxRetries() {
                return 3;
            }

            @Override
            public boolean parallelDownload() {
                return true;
            }

            @Override
            public int parallelChunks() {
                return 4;
            }

            @Override
            public int chunkSizeMB() {
                return 10;
            }

            @Override
            public String userAgent() {
                return "gollek-sdk/" + revision();
            }

            @Override
            public boolean autoDownload() {
                return !config.forceDownload();
            }

            @Override
            public String revision() {
                return config.revision() != null ? config.revision() : "main";
            }
        };

        HuggingFaceClient client = new HuggingFaceClient();
        try {
            var objectMapperField = HuggingFaceClient.class.getDeclaredField("objectMapper");
            objectMapperField.setAccessible(true);
            objectMapperField.set(client,
                    new ObjectMapper().registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule()));

            var configField = HuggingFaceClient.class.getDeclaredField("config");
            configField.setAccessible(true);
            configField.set(client, hfConfig);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize standalone HuggingFaceClient", e);
        }

        return new HuggingFaceRepository(config.cacheDir(), client, hfConfig, new ManifestStore());
    }

    private static RepositoryContext toContext(HubConfig config) {
        return new RepositoryContext(
                config.cacheDir(),
                Duration.ofSeconds(config.timeoutSeconds()),
                Map.of("token", config.token() != null ? config.token() : ""));
    }
}
