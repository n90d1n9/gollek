package tech.kayys.gollek.cli.registry;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelRegistry;
import tech.kayys.gollek.spi.model.ModelStatsProvider;

import java.time.LocalDateTime;

/**
 * Minimal stats provider for CLI builds.
 * Returns basic zeroed stats to satisfy registry dependencies.
 */
@ApplicationScoped
public class CliModelStatsProvider implements ModelStatsProvider {

    @Override
    public Uni<ModelRegistry.ModelStats> getModelStats(String requestId, String modelId) {
        return Uni.createFrom().item(new ModelRegistry.ModelStats(
                modelId,
                "UNKNOWN",
                0L,
                0L,
                (LocalDateTime) null,
                (LocalDateTime) null));
    }
}
