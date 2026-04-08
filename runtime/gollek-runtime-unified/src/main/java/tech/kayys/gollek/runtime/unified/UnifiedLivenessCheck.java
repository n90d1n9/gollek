package tech.kayys.gollek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Liveness;

/**
 * Liveness probe for the unified runtime.
 *
 * <p>Indicates whether the application container is up and running.
 * Exposed at {@code /health/live}.
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Liveness
public class UnifiedLivenessCheck implements HealthCheck {

    /**
     * Executes the liveness check.
     *
     * @return A {@link HealthCheckResponse} indicating the application is live.
     */
    @Override
    public HealthCheckResponse call() {
        return HealthCheckResponse.up("gollek-unified-runtime-liveness");
    }
}
