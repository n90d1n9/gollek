package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;
import tech.kayys.gollek.client.agent.AgentServingPreflightPolicy;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

/**
 * Prints local-only starter route templates that external agent runtimes can validate before publishing.
 */
@Command(
        name = "template-example",
        aliases = { "route-example", "example-template" },
        description = "Print a starter saved agent route template without HTTP, model invocation, tools, or retrieval")
public class AgentTemplateExampleCommand implements Callable<Integer> {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String OBJECT = "gollek.agent_route_template";
    private static final String VERSION = "v1";
    private static final String DEFAULT_MODEL = "demo-model";
    private static final String DEFAULT_PREFLIGHT_POLICY = AgentServingPreflightPolicy.PROFILE_REQUESTED_ROUTE;

    @ParentCommand
    AgentCommand parentCommand;

    @Option(names = { "-m", "--model" }, description = "Model id to place in the example route template")
    String modelId;

    @Option(names = { "--surface" }, description = "Example serving surface: chat, responses, or embeddings")
    String surface;

    @Option(names = { "--feature-profile" }, description = "Example serving feature profile")
    String featureProfile;

    @Option(names = { "--preflight-policy" }, description = "Example preflight policy profile")
    String preflightPolicy;

    @Option(names = { "--id", "--example-id" }, description = "Starter example id from --list")
    String exampleId;

    @Option(names = { "--list", "--catalog" }, description = "Print the available route template example catalog")
    boolean list;

    @Option(names = { "-o", "--output" }, description = "Write the generated JSON to a file")
    Path outputPath;

    @Option(names = { "--overwrite" }, description = "Allow --output to replace an existing file")
    boolean overwrite;

    @Option(names = { "--json" }, description = "Print JSON output")
    boolean json;

    @Override
    public Integer call() throws IOException {
        if (parentCommand != null && parentCommand.root() != null) {
            parentCommand.root().bootstrapInheritedEnvironment();
        }

        if (list) {
            String jsonOutput = MAPPER.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(AgentRouteTemplateExampleCatalog.toMetadata());
            Path writtenOutput = writeOutputIfRequested(jsonOutput);
            if (json) {
                System.out.println(jsonOutput);
            } else {
                printCatalog();
                printOutputPath(writtenOutput);
            }
            return 0;
        }

        Map<String, Object> template = template();
        String jsonOutput = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(template);
        Path writtenOutput = writeOutputIfRequested(jsonOutput);
        if (json) {
            System.out.println(jsonOutput);
        } else {
            printHuman(template, writtenOutput);
        }
        return 0;
    }

    private Map<String, Object> template() {
        AgentRouteTemplateExampleCatalog.Entry selectedExample = selectedExample();
        String effectiveSurface = surface(selectedExample);
        String effectiveFeatureProfile = featureProfile(selectedExample, effectiveSurface);
        String effectiveModel = firstNonBlank(modelId, DEFAULT_MODEL);
        AgentServingPreflightPolicy policy = AgentServingPreflightPolicy.requireProfile(
                firstNonBlank(preflightPolicy, DEFAULT_PREFLIGHT_POLICY));

        Map<String, Object> template = new LinkedHashMap<>();
        template.put("object", OBJECT);
        template.put("version", VERSION);
        template.put("model", effectiveModel);
        template.put("surface", effectiveSurface);
        template.put("feature_profile", effectiveFeatureProfile);
        template.put("preflight_policy", policy.toMetadata());
        template.put("discover_mcp_tools", shouldDiscoverMcp(effectiveSurface, effectiveFeatureProfile));
        template.put("mcp_discovery_required", false);
        template.put("validate_tools", shouldValidateTools(effectiveSurface));
        template.put("tool_validation_required", shouldValidateTools(effectiveSurface));
        template.put("validate_request", true);
        template.put("request_validation_required", true);
        template.put("openai_tool_compatibility", true);
        template.put("enabled_only", true);
        template.put("request", request(effectiveModel, effectiveSurface, effectiveFeatureProfile));
        return template;
    }

    private AgentRouteTemplateExampleCatalog.Entry selectedExample() {
        if (!hasText(exampleId)) {
            return null;
        }
        return AgentRouteTemplateExampleCatalog.find(exampleId)
                .orElseThrow(() -> new IllegalArgumentException("Unsupported example id: " + exampleId.trim()
                        + ". Use one of: " + String.join(", ", AgentRouteTemplateExampleCatalog.supportedIds())));
    }

    private String surface(AgentRouteTemplateExampleCatalog.Entry selectedExample) {
        String requestedSurface = AgentRequestShape.normalizeSurface(surface);
        if (hasText(surface)) {
            String supportedSurface = requireSupportedSurface(requestedSurface);
            if (selectedExample != null && !selectedExample.surface().equals(supportedSurface)) {
                throw new IllegalArgumentException("Example id " + selectedExample.id()
                        + " requires surface " + selectedExample.surface()
                        + ", but --surface was " + supportedSurface);
            }
            return supportedSurface;
        }
        if (selectedExample != null) {
            return selectedExample.surface();
        }
        if (hasText(featureProfile)) {
            return AgentRequestShape.defaultSurfaceForProfile(
                    requireSupportedFeatureProfile(AgentServingFeatureProfile.normalizeName(featureProfile)),
                    AgentRequestShape.DEFAULT_SURFACE);
        }
        return AgentRequestShape.DEFAULT_SURFACE;
    }

    private String featureProfile(AgentRouteTemplateExampleCatalog.Entry selectedExample, String effectiveSurface) {
        if (hasText(featureProfile)) {
            String supportedFeatureProfile =
                    requireSupportedFeatureProfile(AgentServingFeatureProfile.normalizeName(featureProfile));
            if (selectedExample != null && !selectedExample.featureProfile().equals(supportedFeatureProfile)) {
                throw new IllegalArgumentException("Example id " + selectedExample.id()
                        + " requires feature profile " + selectedExample.featureProfile()
                        + ", but --feature-profile was " + supportedFeatureProfile);
            }
            return supportedFeatureProfile;
        }
        if (selectedExample != null) {
            return selectedExample.featureProfile();
        }
        return switch (effectiveSurface) {
            case "embeddings" -> AgentServingFeatureProfile.EMBEDDING_RAG;
            case "responses" -> AgentServingFeatureProfile.RESPONSES_AGENT;
            default -> AgentServingFeatureProfile.CHAT_AGENT;
        };
    }

    private static String requireSupportedSurface(String value) {
        if (List.of("chat", "responses", "embeddings").contains(value)) {
            return value;
        }
        throw new IllegalArgumentException("Unsupported example surface: " + value
                + ". Use one of: chat, responses, embeddings");
    }

    private static String requireSupportedFeatureProfile(String value) {
        if (AgentServingFeatureProfile.find(value).isPresent()) {
            return value;
        }
        throw new IllegalArgumentException("Unsupported example feature profile: " + value
                + ". Use one of: " + String.join(", ", AgentServingFeatureProfile.supportedProfileNames()));
    }

    private static Map<String, Object> request(String model, String surface, String featureProfile) {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("model", model);
        request.put("surface", surface);
        request.put("feature_profile", featureProfile);
        if ("embeddings".equals(surface)) {
            request.put("input", List.of(
                    "Index this caller-owned document for retrieval outside Gollek.",
                    "Gollek generates embeddings; the agent platform owns vector storage and retrieval."));
            request.put("dimensions", 768);
            request.put("encoding_format", "float");
            request.put("metadata", Map.of("tenant", "agent-project"));
            return request;
        }
        if ("responses".equals(surface)) {
            request.put("input", "Use the supplied context and return a concise answer.");
        } else {
            request.put("messages", List.of(
                    Map.of("role", "system", "content", "Use supplied context. Do not execute tools directly."),
                    Map.of("role", "user", "content", "Can this route support my agent request?")));
        }
        request.put("tools", List.of(searchTool()));
        request.put("rag_context", List.of(Map.of(
                "source", "builder-example",
                "text", "Caller-owned retrieval context injected by the agent platform.")));
        return request;
    }

    private static Map<String, Object> searchTool() {
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", Map.of("query", Map.of("type", "string")));
        parameters.put("required", List.of("query"));

        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "search_context");
        function.put("description", "Search caller-owned knowledge. Gollek validates this schema but does not execute it.");
        function.put("parameters", parameters);

        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        tool.put("x_gollek", Map.of(
                "mcp_server", "knowledge",
                "mcp_tool_name", "search",
                "boundary", "definition_only"));
        return tool;
    }

    private static boolean shouldValidateTools(String surface) {
        return !"embeddings".equals(surface);
    }

    private static boolean shouldDiscoverMcp(String surface, String featureProfile) {
        return shouldValidateTools(surface)
                && (AgentServingFeatureProfile.MCP_TOOLS.equals(featureProfile)
                || AgentServingFeatureProfile.CHAT_AGENT.equals(featureProfile)
                || AgentServingFeatureProfile.RESPONSES_AGENT.equals(featureProfile));
    }

    private void printHuman(Map<String, Object> template, Path writtenOutput) {
        System.out.println("Gollek agent route template example");
        System.out.println("Route: model=" + template.get("model")
                + ", surface=" + template.get("surface")
                + ", feature_profile=" + template.get("feature_profile"));
        Map<?, ?> policy = template.get("preflight_policy") instanceof Map<?, ?> map ? map : Map.of();
        System.out.println("Preflight policy: " + policy.get("profile"));
        if (writtenOutput != null) {
            System.out.println("Output file: " + writtenOutput);
            System.out.println("Template check: gollek agent template-check --request-file "
                    + shellArgument(writtenOutput) + " --json");
        } else {
            System.out.println("Template check: gollek agent template-check --request-file <file> --json");
        }
        System.out.println("Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no");
        System.out.println("Use --json to print the route template.");
    }

    private void printCatalog() {
        System.out.println("Gollek agent route template examples");
        for (AgentRouteTemplateExampleCatalog.Entry entry : AgentRouteTemplateExampleCatalog.entries()) {
            System.out.println("- " + entry.id() + ": surface=" + entry.surface()
                    + ", feature_profile=" + entry.featureProfile());
            System.out.println("  " + entry.description());
            System.out.println("  command: " + entry.command());
        }
        System.out.println("Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no");
        System.out.println("Use --list --json to print the raw catalog.");
    }

    private Path writeOutputIfRequested(String jsonOutput) throws IOException {
        if (outputPath == null) {
            return null;
        }
        Path target = outputPath.toAbsolutePath().normalize();
        if (Files.exists(target) && !overwrite) {
            throw new IllegalArgumentException("Output file already exists: " + target
                    + ". Use --overwrite to replace it.");
        }
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.writeString(target, jsonOutput + System.lineSeparator(), StandardCharsets.UTF_8);
        return target;
    }

    private static void printOutputPath(Path writtenOutput) {
        if (writtenOutput != null) {
            System.out.println("Output file: " + writtenOutput);
        }
    }

    private static String shellArgument(Path path) {
        String value = path.toString();
        if (value.chars().allMatch(AgentTemplateExampleCommand::isShellSafeChar)) {
            return value;
        }
        return "'" + value.replace("'", "'\"'\"'") + "'";
    }

    private static boolean isShellSafeChar(int value) {
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-:+".indexOf(value) >= 0;
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
