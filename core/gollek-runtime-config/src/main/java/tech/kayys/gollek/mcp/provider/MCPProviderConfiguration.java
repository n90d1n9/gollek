package tech.kayys.gollek.mcp.provider;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@ConfigMapping(prefix = "wayang.inference.mcp")
public interface MCPProviderConfiguration {

    @WithDefault("false")
    boolean enabled();

    @WithName("servers")
    Map<String, MCPServerConfig> servers();

    @WithDefault("30s")
    String defaultTimeout();

    @WithDefault("true")
    boolean toolsEnabled();

    @WithDefault("true")
    boolean resourcesEnabled();

    @WithDefault("true")
    boolean promptsEnabled();

    @WithName("cache")
    CacheConfig cache();

    interface MCPServerConfig {
        String name();

        @WithDefault("stdio")
        String transport();

        Optional<String> command();

        @WithDefault("")
        List<String> args();

        @WithDefault("")
        Map<String, String> env();

        Optional<String> url();

        @WithDefault("true")
        boolean autoReconnect();

        @WithDefault("3")
        int maxReconnectAttempts();
    }

    interface CacheConfig {
        @WithDefault("true")
        boolean enabled();

        @WithDefault("1000")
        int maxSize();

        @WithDefault("15m")
        String ttl();
    }
}
