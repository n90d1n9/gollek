package tech.kayys.gollek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import java.util.Map;

/**
 * REST endpoint for checking the unified runtime status.
 *
 * <p>Exposes general system health and version information in a standard 
 * JSON format.
 *
 * @author Bhangun
 * @since 1.0.0
 */
@Path("/status")
@ApplicationScoped
public class StatusResource {

    /**
     * Retrieves the current runtime status.
     *
     * @return A map containing status information.
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Map<String, Object> getStatus() {
        return Map.of(
            "status", "UP",
            "runtime", "Gollek Unified Runtime",
            "version", "1.0.0",
            "timestamp", System.currentTimeMillis()
        );
    }
}
