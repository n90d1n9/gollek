package tech.kayys.gollek.model.core;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;

/**
 * Compatibility facade for the legacy repository context package.
 */
public record RepositoryContext(
        Path cacheDir,
        Duration timeout,
        Map<String, Object> attributes) {

    public tech.kayys.gollek.spi.model.RepositoryContext toSpiContext() {
        return new tech.kayys.gollek.spi.model.RepositoryContext(cacheDir, timeout, attributes);
    }
}
