package tech.kayys.gollek.provider.openai;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

/**
 * OpenAI provider configuration.
 */
@ConfigMapping(prefix = "gollek.providers.openai")
public interface OpenAiConfig {

    /**
     * Enable/disable OpenAI provider.
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * OpenAI API key.
     */
    @WithName("api-key")
    @WithDefault("dummy")
    String apiKey();

    /**
     * OpenAI API base URL.
     */
    @WithName("base-url")
    @WithDefault("https://api.openai.com")
    String baseUrl();

    /**
     * Default model to use.
     */
    @WithName("default-model")
    @WithDefault("gpt-3.5-turbo")
    String defaultModel();

    /**
     * Request timeout in seconds.
     */
    @WithName("timeout-seconds")
    @WithDefault("30")
    int timeoutSeconds();

    /**
     * Maximum retries for failed requests.
     */
    @WithName("max-retries")
    @WithDefault("3")
    int maxRetries();
}
