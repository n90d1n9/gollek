package tech.kayys.gollek.model.repo.hf;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.Optional;

@ConfigMapping(prefix = "wayang.inference.repository.huggingface")
public interface HuggingFaceConfig {

    @WithDefault("https://huggingface.co")
    String baseUrl();

    Optional<String> token();

    @WithDefault("30")
    int timeoutSeconds();

    @WithDefault("3")
    int maxRetries();

    @WithDefault("true")
    boolean parallelDownload();

    @WithDefault("4")
    int parallelChunks();

    @WithDefault("10")
    int chunkSizeMB();

    @WithDefault("wayang-inference/1.0")
    String userAgent();

    @WithDefault("true")
    boolean autoDownload();

    @WithDefault("main")
    String revision();
}
