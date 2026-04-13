package tech.kayys.gollek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import org.jboss.logging.Logger;
import tech.kayys.wayang.agent.spi.AgentOrchestrator;
import tech.kayys.wayang.agent.spi.AgentResponse;
import tech.kayys.wayang.agent.spi.AgentRequest;
import tech.kayys.wayang.agent.core.skills.SkillsLoader;
import tech.kayys.wayang.agent.core.skills.loader.SkillExecutor;
import tech.kayys.wayang.agent.orchestration.ReActOrchestrator;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * REST endpoints for agent management and execution.
 */
@Path("/api/agents")
@ApplicationScoped
public class AgentResource {
    private static final Logger LOGGER = Logger.getLogger(AgentResource.class);

    private SkillExecutor skillExecutor;
    private Map<String, SkillsLoader.SkillMetadata> cachedSkills;

    @Inject
    ReActOrchestrator reActOrchestrator;

    /**
     * List all available agents.
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listAgents() {
        try {
            initializeSkillExecutor();

            return Response.ok(Map.of(
                "agents", List.of(
                    Map.of(
                        "name", "default-agent",
                        "type", "single-agent",
                        "description", "Single agent with all available tools",
                        "reasoning", "llm-based-chain-of-thought",
                        "skillCount", cachedSkills.size()
                    )
                ),
                "totalAgents", 1
            )).build();

        } catch (Exception e) {
            LOGGER.errorf(e, "Error listing agents: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to list agents: " + e.getMessage()))
                    .build();
        }
    }

    /**
     * Get details about a specific agent.
     */
    @GET
    @Path("/{agentName}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response describeAgent(@PathParam("agentName") String agentName) {
        try {
            initializeSkillExecutor();

            if (!"default-agent".equals(agentName)) {
                return Response.status(Response.Status.NOT_FOUND)
                        .entity(Map.of("error", "Agent not found: " + agentName))
                        .build();
            }

            List<Map<String, Object>> skills = new ArrayList<>();
            for (String skillName : cachedSkills.keySet()) {
                SkillsLoader.SkillMetadata metadata = cachedSkills.get(skillName);
                skills.add(Map.of(
                    "name", skillName,
                    "description", metadata.getDescription(),
                    "license", metadata.getLicense()
                ));
            }

            return Response.ok(Map.of(
                "name", agentName,
                "type", "single-agent",
                "description", "Single agent with dynamic tool selection",
                "reasoning", "llm-based-chain-of-thought",
                "toolSelection", "intelligent",
                "skills", skills
            )).build();

        } catch (Exception e) {
            LOGGER.errorf(e, "Error describing agent: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to describe agent: " + e.getMessage()))
                    .build();
        }
    }

    /**
     * Execute an agent with a query.
     */
    @POST
    @Path("/{agentName}/execute")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response executeAgent(@PathParam("agentName") String agentName, 
                                Map<String, Object> request) {
        try {
            initializeSkillExecutor();

            String query = (String) request.get("query");
            if (query == null || query.isEmpty()) {
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Query is required"))
                        .build();
            }

            // Create context
            AgentRequest context = AgentRequest.builder()
                    .userId((String) request.getOrDefault("userId", "api-user"))
                    .sessionId((String) request.getOrDefault("sessionId", UUID.randomUUID().toString()))
                    .prompt(query)
                    .build();

            // Execute
            long startTime = System.currentTimeMillis();
            AgentResponse response = reActOrchestrator.execute(context).await().indefinitely();
            long elapsed = System.currentTimeMillis() - startTime;

            // Format result
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("agentName", agentName);
            result.put("query", query);
            result.put("answer", response.answer());
            result.put("executionTimeMs", elapsed);
            result.put("reasoningSteps", response.steps());
            result.put("toolCount", response.steps().size());

            return Response.ok(result).build();

        } catch (Exception e) {
            LOGGER.errorf(e, "Error executing agent: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Agent execution failed: " + e.getMessage()))
                    .build();
        }
    }

    /**
     * Get skills available to an agent.
     */
    @GET
    @Path("/{agentName}/skills")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getAgentSkills(@PathParam("agentName") String agentName) {
        try {
            initializeSkillExecutor();

            List<Map<String, Object>> skills = new ArrayList<>();
            for (String skillName : cachedSkills.keySet()) {
                SkillsLoader.SkillMetadata metadata = cachedSkills.get(skillName);
                skills.add(Map.of(
                    "name", skillName,
                    "description", metadata.getDescription(),
                    "license", metadata.getLicense()
                ));
            }

            return Response.ok(Map.of(
                "agentName", agentName,
                "skillCount", skills.size(),
                "skills", skills
            )).build();

        } catch (Exception e) {
            LOGGER.errorf(e, "Error getting agent skills: %s", e.getMessage());
            return Response.serverError()
                    .entity(Map.of("error", "Failed to get skills: " + e.getMessage()))
                    .build();
        }
    }

    // createAgent removed as it's no longer needed with ReActOrchestrator injection

    private synchronized void initializeSkillExecutor() throws Exception {
        if (skillExecutor == null) {
            String skillsDir = System.getProperty("gollek.skills.dir",
                    System.getProperty("user.home") + "/.gollek/skills");
            Path skillsPath = Paths.get(skillsDir);
            skillExecutor = new SkillExecutor(skillsPath);
            cachedSkills = skillExecutor.loadAllSkills();
            LOGGER.infof("Initialized skill executor with %d skills", cachedSkills.size());
        }
    }
}
