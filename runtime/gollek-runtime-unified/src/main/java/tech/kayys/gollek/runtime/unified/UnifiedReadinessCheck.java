package tech.kayys.gollek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;

/**
 * Readiness probe for the unified runtime.
 *
 * <p>Indicates whether the application is fully initialized and 
 * ready to serve traffic. Exposed at {@code /health/ready}.
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Readiness
public class UnifiedReadinessCheck implements HealthCheck {

    /**
     * Executes the readiness check.
     *
     * @return A {@link HealthCheckResponse} indicating the application is ready.
     */
    @Override
    public HealthCheckResponse call() {
        return HealthCheckResponse.up("gollek-unified-runtime-readiness");
    }
}
