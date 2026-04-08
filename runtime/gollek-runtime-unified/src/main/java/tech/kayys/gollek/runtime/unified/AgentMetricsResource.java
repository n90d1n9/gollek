package tech.kayys.gollek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import org.jboss.logging.Logger;
import tech.kayys.gollek.agent.core.ManagedAgentService;

import java.util.*;

/**
 * REST endpoints for agent metrics and monitoring.
 */
@Path("/api/agents/metrics")
@ApplicationScoped
public class AgentMetricsResource {
    private static final Logger LOGGER = Logger.getLogger(AgentMetricsResource.class);

    @Inject
    ManagedAgentService managedAgentService;

    /**
     * Get overall service metrics.
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response getServiceMetrics() {
        try {
            Map<String, Object> metrics = managedAgentService.getServiceMetrics();
            return Response.ok(metrics).build();
        } catch (Exception e) {
            LOGGER.errorf(e, "Error getting service metrics: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to get metrics: " + e.getMessage()))
                    .build();
        }
    }

    /**
     * Get metrics for a specific agent.
     */
    @GET
    @Path("/{agentName}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getAgentMetrics(@PathParam("agentName") String agentName) {
        try {
            Map<String, Object> metrics = managedAgentService.getMetrics(agentName);
            return Response.ok(metrics).build();
        } catch (Exception e) {
            LOGGER.errorf(e, "Error getting agent metrics: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to get metrics: " + e.getMessage()))
                    .build();
        }
    }

    /**
     * Get execution history for an agent.
     */
    @GET
    @Path("/{agentName}/history")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getExecutionHistory(@PathParam("agentName") String agentName,
                                       @QueryParam("limit") @DefaultValue("20") int limit) {
        try {
            var history = managedAgentService.getExecutionHistory(agentName, limit);
            return Response.ok(Map.of(
                "agentName", agentName,
                "recordCount", history.size(),
                "records", history
            )).build();
        } catch (Exception e) {
            LOGGER.errorf(e, "Error getting execution history: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to get history: " + e.getMessage()))
                    .build();
        }
    }
}
