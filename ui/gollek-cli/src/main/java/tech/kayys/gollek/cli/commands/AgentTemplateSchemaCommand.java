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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Command(
        name = "template-schema",
        aliases = { "route-schema", "schema" },
        description = "Print the local JSON Schema for saved agent route templates")
public class AgentTemplateSchemaCommand implements Callable<Integer> {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String OBJECT = "gollek.agent_route_template_schema";
    private static final String SCHEMA_VERSION = "v1";
    private static final String SCHEMA_ID =
            "https://gollek-ai.github.io/schemas/agent-route-template.schema.json";
    private static final List<String> SUPPORTED_SURFACES = List.of("chat", "responses", "embeddings");

    @ParentCommand
    AgentCommand parentCommand;

    @Option(names = { "--json" }, description = "Print the raw route-template JSON Schema")
    boolean json;

    @Override
    public Integer call() throws IOException {
        if (parentCommand != null && parentCommand.root() != null) {
            parentCommand.root().bootstrapInheritedEnvironment();
        }

        Map<String, Object> schema = schema();
        if (json) {
            System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(schema));
        } else {
            printHuman(schema);
        }
        return 0;
    }

    static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("$schema", "https://json-schema.org/draft/2020-12/schema");
        schema.put("$id", SCHEMA_ID);
        schema.put("title", "Gollek Agent Route Template");
        schema.put("description",
                "Saved route template accepted by Gollek agent template-check and agent-serving preflight commands.");
        schema.put("type", "object");
        schema.put("required", List.of("request"));
        schema.put("additionalProperties", true);
        schema.put("anyOf", List.of(
                object("required", List.of("model")),
                object("required", List.of("model_id")),
                object("properties", object("request", object("required", List.of("model"))))));
        schema.put("properties", rootProperties());
        schema.put("$defs", definitions());
        schema.put("x-gollek", metadata());
        return schema;
    }

    private static Map<String, Object> rootProperties() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("model", string("Gollek model id for the route."));
        properties.put("model_id", string("Alias for model."));
        properties.put("surface", stringEnum(
                "OpenAI-compatible serving surface for this route.",
                SUPPORTED_SURFACES));
        properties.put("feature_profile", stringEnum(
                "Gollek serving feature bundle for this route.",
                AgentServingFeatureProfile.supportedProfileNames()));
        properties.put("featureProfile", stringEnum(
                "CamelCase alias for feature_profile.",
                AgentServingFeatureProfile.supportedProfileNames()));
        properties.put("profile", stringEnum(
                "Builder-facing alias for feature_profile.",
                AgentServingFeatureProfile.supportedProfileNames()));
        properties.put("preflight_policy", preflightPolicy());
        properties.put("preflightPolicy", preflightPolicy());
        properties.put("preflight_gate", preflightPolicy());
        properties.put("preflightGate", preflightPolicy());
        properties.put("discover_mcp_tools", bool("Whether Gollek should discover MCP tool definitions for validation."));
        properties.put("mcp_discovery_required", bool("Whether missing MCP discovery blocks the route."));
        properties.put("validate_tools", bool("Whether Gollek should validate OpenAI-compatible tool schemas."));
        properties.put("tool_validation_required", bool("Whether tool validation warnings block the route."));
        properties.put("validate_request", bool("Whether Gollek should dry-run validate the normalized request."));
        properties.put("request_validation_required", bool("Whether request validation warnings block the route."));
        properties.put("openai_tool_compatibility", bool("Request OpenAI-compatible MCP tool schemas."));
        properties.put("enabled_only", bool("Only discover enabled MCP registrations."));
        properties.put("route_match_required", bool("Require the selected model and surface to match the requested route."));
        properties.put("warnings_blocking", bool("Treat preflight warnings as blocking."));
        properties.put("request", ref("#/$defs/openai_request"));
        return properties;
    }

    private static Map<String, Object> definitions() {
        Map<String, Object> defs = new LinkedHashMap<>();
        defs.put("openai_request", openAiRequest());
        defs.put("message", object(
                "type", "object",
                "additionalProperties", true,
                "properties", object(
                        "role", stringEnum("OpenAI-compatible message role.", List.of("system", "user", "assistant", "tool")),
                        "content", object(
                                "description", "Message content.",
                                "oneOf", List.of(
                                        object("type", "string"),
                                        object("type", "array"))))));
        defs.put("tool", object(
                "type", "object",
                "additionalProperties", true,
                "properties", object(
                        "type", stringEnum("Tool type.", List.of("function")),
                        "function", object(
                                "type", "object",
                                "additionalProperties", true,
                                "properties", object(
                                        "name", string("Tool function name."),
                                        "description", string("Tool function description."),
                                        "parameters", object("type", "object", "additionalProperties", true))))));
        defs.put("rag_context_item", object(
                "type", "object",
                "additionalProperties", true,
                "properties", object(
                        "source", string("Caller-owned context source."),
                        "text", string("Caller-owned context text."))));
        return defs;
    }

    private static Map<String, Object> openAiRequest() {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("type", "object");
        request.put("additionalProperties", true);
        request.put("properties", object(
                "model", string("Gollek model id used by the nested request."),
                "surface", stringEnum("Nested route surface.", SUPPORTED_SURFACES),
                "feature_profile", stringEnum(
                        "Nested feature profile.",
                        AgentServingFeatureProfile.supportedProfileNames()),
                "featureProfile", stringEnum(
                        "CamelCase nested feature profile alias.",
                        AgentServingFeatureProfile.supportedProfileNames()),
                "messages", object(
                        "type", "array",
                        "items", ref("#/$defs/message")),
                "input", object(
                        "description", "Responses or embeddings input.",
                        "oneOf", List.of(
                                object("type", "string"),
                                object("type", "array"))),
                "tools", object(
                        "type", "array",
                        "items", ref("#/$defs/tool")),
                "rag_context", object(
                        "type", "array",
                        "items", ref("#/$defs/rag_context_item")),
                "ragContext", object(
                        "type", "array",
                        "items", ref("#/$defs/rag_context_item")),
                "context", object(
                        "description", "Alias for caller-owned RAG context.",
                        "oneOf", List.of(
                                object("type", "string"),
                                object("type", "array"))),
                "stream", bool("Whether to request streaming."),
                "temperature", object("type", "number"),
                "max_tokens", object("type", "integer", "minimum", 1),
                "dimensions", object("type", "integer", "minimum", 1),
                "encoding_format", stringEnum("Embedding encoding format.", List.of("float", "base64")),
                "metadata", object("type", "object", "additionalProperties", true)));
        return request;
    }

    private static Map<String, Object> preflightPolicy() {
        return object(
                "description", "Caller-owned policy for deciding whether a serving preflight is acceptable.",
                "oneOf", List.of(
                        stringEnum("Named preflight policy.", AgentServingPreflightPolicy.supportedProfileNames()),
                        object(
                                "type", "object",
                                "additionalProperties", true,
                                "properties", object(
                                        "profile", stringEnum(
                                                "Named preflight policy.",
                                                AgentServingPreflightPolicy.supportedProfileNames()),
                                        "policy", stringEnum(
                                                "Alias for profile.",
                                                AgentServingPreflightPolicy.supportedProfileNames()),
                                        "route_match_required", bool("Require selected route match."),
                                        "warnings_blocking", bool("Treat warnings as blocking.")))));
    }

    private static Map<String, Object> metadata() {
        return object(
                "object", OBJECT,
                "version", SCHEMA_VERSION,
                "schema_id", SCHEMA_ID,
                "template_check_command", "gollek agent template-check --request-file <file> --json",
                "supported_surfaces", SUPPORTED_SURFACES,
                "supported_feature_profiles", AgentServingFeatureProfile.supportedProfileNames(),
                "supported_preflight_policies", AgentServingPreflightPolicy.supportedProfileNames(),
                "boundary", object(
                        "local_only", true,
                        "http_request", false,
                        "model_invoked", false,
                        "tool_execution", false,
                        "retrieval_execution", false,
                        "workflow_state", false,
                        "agent_orchestrator_owns", List.of(
                                "planning",
                                "memory",
                                "tool authorization",
                                "tool execution",
                                "retrieval policy",
                                "workflow state")));
    }

    private void printHuman(Map<String, Object> schema) {
        Map<?, ?> metadata = schema.get("x-gollek") instanceof Map<?, ?> map ? map : Map.of();
        System.out.println("Gollek agent route template schema");
        System.out.println("Schema: " + metadata.get("schema_id"));
        System.out.println("Version: " + metadata.get("version"));
        System.out.println("Surfaces: " + String.join(",", SUPPORTED_SURFACES));
        System.out.println("Feature profiles: " + String.join(",", AgentServingFeatureProfile.supportedProfileNames()));
        System.out.println("Preflight policies: " + String.join(",", AgentServingPreflightPolicy.supportedProfileNames()));
        System.out.println("Template check: " + metadata.get("template_check_command"));
        System.out.println("Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no");
        System.out.println("Use --json to print the raw JSON Schema.");
    }

    private static Map<String, Object> string(String description) {
        return object("type", "string", "description", description);
    }

    private static Map<String, Object> stringEnum(String description, List<String> values) {
        return object("type", "string", "description", description, "enum", List.copyOf(values));
    }

    private static Map<String, Object> bool(String description) {
        return object("type", "boolean", "description", description);
    }

    private static Map<String, Object> ref(String value) {
        return object("$ref", value);
    }

    private static Map<String, Object> object(Object... entries) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (entries == null) {
            return out;
        }
        for (int i = 0; i + 1 < entries.length; i += 2) {
            out.put(entries[i].toString(), entries[i + 1]);
        }
        return out;
    }
}
