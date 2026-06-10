package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Stable machine-readable codes for Gollek agent-serving readiness issues.
 */
public final class AgentReadinessIssueCodes {
    private static final String BLOCKING = "blocking";
    private static final String WARNING = "warning";
    private static final String UNKNOWN = "unknown";

    private static final List<CatalogEntry> CATALOG = List.of(
            entry("CAPABILITIES_RESPONSE_MISSING", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "Capabilities discovery did not return a response.",
                    "Check that the Gollek serving endpoint exposes /v1/agent/capabilities."),
            entry("SERVICE_ROLE_MISMATCH", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "The service role is not inference_serving_engine.",
                    "Run against a Gollek serving deployment, not an orchestrator or training-only endpoint."),
            entry("CAPABILITY_COMPATIBILITY_MISSING", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "A required agent-serving compatibility flag is missing.",
                    "Enable or upgrade the serving feature advertised by /v1/agent/capabilities."),
            entry("ENDPOINT_MISSING", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "A required agent-facing endpoint is missing.",
                    "Expose the missing endpoint or route the orchestrator to a newer Gollek serving build."),
            entry("MODEL_SERVING_BOUNDARY_MISSING", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "Gollek does not advertise ownership of model serving.",
                    "Use a serving profile where Gollek owns inference and provider routing."),
            entry("TOOL_EXECUTION_LOOP_BOUNDARY_MISSING", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "The orchestrator-owned tool loop boundary is missing.",
                    "Keep planning, tool execution loops, and workflow state outside Gollek."),
            entry("AUTH_X_API_KEY_HEADER_MISMATCH", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "The X-API-Key authentication hint does not match the contract.",
                    "Align the serving auth configuration with the agent contract."),
            entry("AUTHORIZATION_HEADER_MISMATCH", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "The Authorization bearer-token hint does not match the contract.",
                    "Align the serving auth configuration with the agent contract."),
            entry("TRACE_REQUEST_ID_HEADER_MISMATCH", AgentServingReadinessReport.AREA_CAPABILITIES, BLOCKING,
                    "The trace request-id header hint does not match the contract.",
                    "Expose X-Gollek-Request-Id so orchestrators can correlate requests."),

            entry("CONTRACT_RESPONSE_MISSING", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "Serving contract discovery did not return a response.",
                    "Check that the Gollek serving endpoint exposes /v1/agent/contract."),
            entry("GOLLEK_RESPONSIBILITY_MISSING", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "A required Gollek-owned serving responsibility is missing.",
                    "Update the contract so Gollek owns only serving, validation, discovery, embeddings, and context injection."),
            entry("ORCHESTRATOR_RESPONSIBILITY_MISSING", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "A required orchestrator-owned responsibility is missing.",
                    "Keep planning, memory, retrieval policy, tool execution, and workflow state in the agent runtime."),
            entry("TOOL_EXECUTION_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "The contract enables tool execution inside Gollek.",
                    "Disable tool execution in Gollek and execute tools in the external orchestrator."),
            entry("RETRIEVAL_EXECUTION_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "The contract enables retrieval execution inside Gollek.",
                    "Keep retrieval policy and vector search in the external orchestrator."),
            entry("STREAMING_CONTRACT_INCOMPLETE", AgentServingReadinessReport.AREA_CONTRACT, BLOCKING,
                    "The streaming contract is incomplete.",
                    "Expose the expected stream event and done-sentinel metadata."),

            entry("FEATURE_NEGOTIATION_METADATA_MISSING",
                    AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                    BLOCKING,
                    "Feature negotiation metadata is missing.",
                    "Upgrade Gollek or call an agent-serving endpoint that exposes feature_negotiation metadata."),
            entry("FEATURE_PROFILE_UNSUPPORTED",
                    AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                    BLOCKING,
                    "The requested agent-serving feature profile is unsupported.",
                    "Use a supported serving profile or send explicit required_features for the integration."),
            entry("CONTRACT_VERSION_UNSUPPORTED",
                    AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                    BLOCKING,
                    "The requested agent-serving contract version is unsupported.",
                    "Use one of supported_contract_versions or upgrade the Gollek serving endpoint."),
            entry("REQUIRED_AGENT_FEATURE_UNSUPPORTED",
                    AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                    BLOCKING,
                    "A required agent-serving feature is unsupported.",
                    "Disable that orchestrator feature or route to a Gollek serving build that advertises it."),
            entry("OPTIONAL_AGENT_FEATURE_UNSUPPORTED",
                    AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                    WARNING,
                    "An optional agent-serving feature is unsupported.",
                    "Keep the integration running but disable the optional path in the orchestrator."),

            entry("MODEL_REQUIRED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "No model id was supplied for the preflight.",
                    "Pass model/model_id in the request or CLI option before validating the route."),
            entry("MODEL_CAPABILITY_RESPONSE_MISSING", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "Model capability discovery did not return a response.",
                    "Check /v1/models/{id}/capabilities for the selected model."),
            entry("MODEL_ROUTE_UNAVAILABLE", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model is not available through Gollek serving.",
                    "Load the model or choose a model route that Gollek can serve."),
            entry("EMBEDDINGS_ENDPOINT_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not advertise the embeddings endpoint.",
                    "Choose an embedding-capable model or disable embedding-route preflight."),
            entry("EMBEDDING_GENERATION_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not support embedding generation.",
                    "Choose an embedding model for caller-owned RAG indexing."),
            entry("EMBEDDING_RETRIEVAL_POLICY_BOUNDARY", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "Embedding route ownership includes retrieval policy.",
                    "Keep retrieval policy outside Gollek and owned by the orchestrator."),
            entry("EMBEDDING_VECTOR_STORE_BOUNDARY", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "Embedding route ownership includes vector-store state.",
                    "Keep vector storage outside Gollek and owned by the orchestrator."),
            entry("COMPLETION_SERVING_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not advertise completion serving.",
                    "Choose a model with chat or responses serving support."),
            entry("CHAT_COMPLETIONS_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not support Chat Completions compatibility.",
                    "Use a chat-compatible route or switch the surface to responses when supported."),
            entry("RESPONSES_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not support Responses compatibility.",
                    "Use a responses-compatible route or switch the surface to chat when supported."),
            entry("SYSTEM_PROMPT_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not advertise system prompt mapping.",
                    "Use a route that maps system prompts or avoid system instructions for that model."),
            entry("TOOL_DEFINITIONS_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not advertise tool definition ingestion.",
                    "Choose a tool-capable model route or disable tool validation for this route."),
            entry("RAG_CONTEXT_UNSUPPORTED", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The selected model does not advertise caller-supplied RAG context injection.",
                    "Choose a route that accepts context documents or omit RAG context for this route."),
            entry("MODEL_ROUTE_TOOL_EXECUTION_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The model route enables tool execution in Gollek.",
                    "Keep tool execution in the orchestrator and return only tool-call declarations from Gollek."),
            entry("RETRIEVAL_POLICY_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The model route claims retrieval-policy ownership.",
                    "Keep retrieval policy in the external agent runtime."),
            entry("VECTOR_STORE_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MODEL_ROUTE, BLOCKING,
                    "The model route claims vector-store ownership.",
                    "Keep vector storage in the external agent runtime."),

            entry("MCP_DISCOVERY_NOT_REQUESTED", AgentServingReadinessReport.AREA_MCP_DISCOVERY, WARNING,
                    "MCP discovery was intentionally skipped.",
                    "Only skip MCP discovery for routes that do not need tools from Gollek."),
            entry("MCP_DISCOVERY_SKIPPED_BY_CALLER", AgentServingReadinessReport.AREA_MCP_DISCOVERY, WARNING,
                    "MCP discovery was skipped by the caller.",
                    "Confirm the route does not require MCP tools, or request discovery in preflight."),
            entry("MCP_DISCOVERY_UNAVAILABLE", AgentServingReadinessReport.AREA_MCP_DISCOVERY, BLOCKING,
                    "MCP discovery is unavailable.",
                    "Enable the MCP registry or mark MCP discovery optional for routes that do not need tools."),
            entry("MCP_DISCOVERY_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MCP_DISCOVERY, BLOCKING,
                    "The MCP discovery endpoint crossed the serving boundary.",
                    "Expose MCP discovery only; execute MCP tools in the orchestrator."),
            entry("MCP_TOOL_EXECUTION_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MCP_DISCOVERY, BLOCKING,
                    "MCP discovery enables tool execution.",
                    "Disable tool execution in Gollek's MCP discovery surface."),
            entry("MCP_TOOL_EXECUTABLE_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_MCP_DISCOVERY, BLOCKING,
                    "A discovered MCP tool is executable from Gollek.",
                    "Return tool definitions only and execute the tool from the external runtime."),

            entry("TOOL_VALIDATION_NOT_REQUESTED", AgentServingReadinessReport.AREA_TOOL_VALIDATION, WARNING,
                    "Tool validation was intentionally skipped.",
                    "Only skip tool validation when the route does not provide tools."),
            entry("TOOL_VALIDATION_RESPONSE_MISSING", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool validation did not return a response.",
                    "Check that /v1/agent/tools/validate is available."),
            entry("TOOL_DEFINITIONS_INVALID", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool definitions are invalid.",
                    "Fix the OpenAI-compatible tool schema before serving the route."),
            entry("TOOL_VALIDATION_MODEL_INVOKED", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool validation invoked a model.",
                    "Keep tool validation dry-run only and model-free."),
            entry("TOOL_VALIDATION_NOT_VALIDATION_ONLY", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool validation is not validation-only.",
                    "Ensure the endpoint validates schema shape without side effects."),
            entry("TOOL_VALIDATION_TOOL_EXECUTION_BOUNDARY", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool validation enables tool execution.",
                    "Do not execute tools from Gollek's validation endpoint."),
            entry("TOOL_AUTHORIZATION_BOUNDARY_DRIFT", AgentServingReadinessReport.AREA_TOOL_VALIDATION, BLOCKING,
                    "Tool validation authorizes tools.",
                    "Keep tool authorization policy in the external agent runtime."),
            entry("TOOL_SCHEMA_PORTABILITY_WARNING", AgentServingReadinessReport.AREA_TOOL_VALIDATION, WARNING,
                    "Tool schema may not be portable across serving providers.",
                    "Simplify the tool schema or record the portability warning in the orchestrator."),

            entry("REQUEST_VALIDATION_NOT_REQUESTED", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, WARNING,
                    "Request validation was intentionally skipped.",
                    "Only skip request validation when another caller validates the request shape."),
            entry("REQUEST_VALIDATION_RESPONSE_MISSING", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "Request validation did not return a response.",
                    "Check that /v1/agent/validate is available."),
            entry("AGENT_REQUEST_INVALID", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "The agent request is invalid.",
                    "Fix the request model, messages/input, tools, or RAG context before serving."),
            entry("REQUEST_VALIDATION_MODEL_INVOKED", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "Request validation invoked a model.",
                    "Keep request validation dry-run only and model-free."),
            entry("REQUEST_VALIDATION_NOT_VALIDATION_ONLY", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "Request validation is not validation-only.",
                    "Ensure validation maps the request without execution side effects."),
            entry("REQUEST_VALIDATION_TOOL_EXECUTION_BOUNDARY", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "Request validation enables tool execution.",
                    "Keep tool execution in the external agent runtime."),
            entry("REQUEST_VALIDATION_RETRIEVAL_BOUNDARY", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "Request validation enables retrieval execution.",
                    "Keep retrieval execution in the external agent runtime."),
            entry("EMBEDDED_TOOL_CONTRACT_INVALID", AgentServingReadinessReport.AREA_REQUEST_VALIDATION, BLOCKING,
                    "The embedded tool contract is invalid.",
                    "Fix tool definitions inside the request before serving."),

            entry("PREFLIGHT_RESPONSE_MISSING", "preflight", BLOCKING,
                    "Server preflight did not return a response.",
                    "Check that /v1/agent/preflight is available."),
            entry("SERVER_PREFLIGHT_BLOCKED", "preflight", BLOCKING,
                    "Server preflight reported blocked without detailed issues.",
                    "Inspect the server-side preflight response and check_results for details."));

    private static final Map<String, CatalogEntry> CATALOG_BY_CODE = catalogByCode(CATALOG);

    private AgentReadinessIssueCodes() {
    }

    public static List<CatalogEntry> catalog() {
        return CATALOG;
    }

    public static Map<String, CatalogEntry> catalogByCode() {
        return CATALOG_BY_CODE;
    }

    public static Optional<CatalogEntry> find(String code) {
        return Optional.ofNullable(CATALOG_BY_CODE.get(normalize(code)));
    }

    public static CatalogEntry describe(String code) {
        String normalized = normalize(code);
        if (normalized == null) {
            return unknownEntry(null);
        }
        CatalogEntry known = CATALOG_BY_CODE.get(normalized);
        if (known != null) {
            return known;
        }
        if (normalized.startsWith("TOOL_SCHEMA_")) {
            return new CatalogEntry(
                    normalized,
                    AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                    WARNING,
                    "Tool schema portability warning.",
                    "Inspect the warning path and simplify the tool schema when needed.");
        }
        return unknownEntry(normalized);
    }

    public static List<CatalogEntry> catalogForArea(String area) {
        String normalizedArea = normalizeArea(area);
        if (normalizedArea == null) {
            return List.of();
        }
        List<CatalogEntry> out = new ArrayList<>();
        for (CatalogEntry entry : CATALOG) {
            if (entry.area().equals(normalizedArea)) {
                out.add(entry);
            }
        }
        return List.copyOf(out);
    }

    public static List<Map<String, Object>> catalogMaps() {
        return CATALOG.stream().map(CatalogEntry::toMap).toList();
    }

    public static String resolve(
            String suppliedCode,
            String area,
            AgentServingReadinessReport.Severity severity,
            String message) {
        return resolve(suppliedCode, area, severityName(severity), message);
    }

    public static String resolve(String suppliedCode, String area, String severity, String message) {
        String normalized = normalize(suppliedCode);
        if (normalized != null) {
            return normalized;
        }
        String explicit = explicitCode(message);
        if (explicit != null) {
            return explicit;
        }
        return fallbackCode(area, severity);
    }

    public static String codeFor(String area, AgentServingReadinessReport.Severity severity, String message) {
        return resolve(null, area, severity, message);
    }

    public static String codeFor(String area, String severity, String message) {
        return resolve(null, area, severity, message);
    }

    public static String toolSchemaWarningCode(String warningCode) {
        String normalized = normalize(warningCode);
        if (normalized == null) {
            return "TOOL_SCHEMA_PORTABILITY_WARNING";
        }
        if (normalized.startsWith("SCHEMA_")) {
            return "TOOL_" + normalized;
        }
        return normalized.startsWith("TOOL_SCHEMA_") ? normalized : "TOOL_SCHEMA_" + normalized;
    }

    public static String normalize(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value.trim()
                .toUpperCase(Locale.ROOT)
                .replaceAll("[^A-Z0-9]+", "_")
                .replaceAll("^_|_$", "");
        return normalized.isBlank() ? null : normalized;
    }

    private static String explicitCode(String message) {
        String normalized = normalizeMessage(message);
        if (normalized == null) {
            return null;
        }
        if (normalized.equals("capabilities response is missing")) {
            return "CAPABILITIES_RESPONSE_MISSING";
        }
        if (normalized.equals("serving contract response is missing")) {
            return "CONTRACT_RESPONSE_MISSING";
        }
        if (normalized.equals("model capability response is missing")) {
            return "MODEL_CAPABILITY_RESPONSE_MISSING";
        }
        if (normalized.equals("tool validation response is missing")) {
            return "TOOL_VALIDATION_RESPONSE_MISSING";
        }
        if (normalized.equals("request validation response is missing")) {
            return "REQUEST_VALIDATION_RESPONSE_MISSING";
        }
        if (normalized.equals("preflight response is missing")) {
            return "PREFLIGHT_RESPONSE_MISSING";
        }
        if (normalized.equals("server preflight reported blocked")) {
            return "SERVER_PREFLIGHT_BLOCKED";
        }
        if (normalized.equals("service_role must be inference_serving_engine")) {
            return "SERVICE_ROLE_MISMATCH";
        }
        if (normalized.startsWith("missing compatibility:")) {
            return "CAPABILITY_COMPATIBILITY_MISSING";
        }
        if (normalized.startsWith("missing endpoint:")) {
            return "ENDPOINT_MISSING";
        }
        if (normalized.contains("model serving in gollek")) {
            return "MODEL_SERVING_BOUNDARY_MISSING";
        }
        if (normalized.contains("tool execution loops in the orchestrator")) {
            return "TOOL_EXECUTION_LOOP_BOUNDARY_MISSING";
        }
        if (normalized.contains("auth.x_api_key_header")) {
            return "AUTH_X_API_KEY_HEADER_MISMATCH";
        }
        if (normalized.contains("auth.authorization_header")) {
            return "AUTHORIZATION_HEADER_MISMATCH";
        }
        if (normalized.contains("traceability.request_id_header")) {
            return "TRACE_REQUEST_ID_HEADER_MISMATCH";
        }
        if (normalized.startsWith("missing gollek-owned responsibility:")) {
            return "GOLLEK_RESPONSIBILITY_MISSING";
        }
        if (normalized.startsWith("missing orchestrator-owned responsibility:")) {
            return "ORCHESTRATOR_RESPONSIBILITY_MISSING";
        }
        if (normalized.contains("contract must not enable tool execution")) {
            return "TOOL_EXECUTION_BOUNDARY_DRIFT";
        }
        if (normalized.contains("contract must not enable retrieval execution")) {
            return "RETRIEVAL_EXECUTION_BOUNDARY_DRIFT";
        }
        if (normalized.contains("streaming contract")) {
            return "STREAMING_CONTRACT_INCOMPLETE";
        }
        if (normalized.equals("feature negotiation metadata is missing")) {
            return "FEATURE_NEGOTIATION_METADATA_MISSING";
        }
        if (normalized.startsWith("feature profile is not supported:")) {
            return "FEATURE_PROFILE_UNSUPPORTED";
        }
        if (normalized.startsWith("required contract version is not supported:")) {
            return "CONTRACT_VERSION_UNSUPPORTED";
        }
        if (normalized.startsWith("required agent feature is not supported:")) {
            return "REQUIRED_AGENT_FEATURE_UNSUPPORTED";
        }
        if (normalized.startsWith("optional agent feature is not supported:")) {
            return "OPTIONAL_AGENT_FEATURE_UNSUPPORTED";
        }
        if (normalized.equals("model is required for preflight")) {
            return "MODEL_REQUIRED";
        }
        if (normalized.equals("model is not available through gollek serving")) {
            return "MODEL_ROUTE_UNAVAILABLE";
        }
        if (normalized.contains("does not advertise the embeddings endpoint")) {
            return "EMBEDDINGS_ENDPOINT_UNSUPPORTED";
        }
        if (normalized.contains("does not support embedding generation")) {
            return "EMBEDDING_GENERATION_UNSUPPORTED";
        }
        if (normalized.contains("must not own retrieval policy")) {
            return "EMBEDDING_RETRIEVAL_POLICY_BOUNDARY";
        }
        if (normalized.contains("must not own vector store state")) {
            return "EMBEDDING_VECTOR_STORE_BOUNDARY";
        }
        if (normalized.contains("does not advertise completion serving")) {
            return "COMPLETION_SERVING_UNSUPPORTED";
        }
        if (normalized.contains("chat completions compatibility")) {
            return "CHAT_COMPLETIONS_UNSUPPORTED";
        }
        if (normalized.contains("responses compatibility")) {
            return "RESPONSES_UNSUPPORTED";
        }
        if (normalized.contains("system prompt mapping")) {
            return "SYSTEM_PROMPT_UNSUPPORTED";
        }
        if (normalized.contains("tool definition ingestion")) {
            return "TOOL_DEFINITIONS_UNSUPPORTED";
        }
        if (normalized.contains("rag context injection")) {
            return "RAG_CONTEXT_UNSUPPORTED";
        }
        if (normalized.contains("model capability must not enable tool execution")) {
            return "MODEL_ROUTE_TOOL_EXECUTION_BOUNDARY_DRIFT";
        }
        if (normalized.contains("retrieval policy must stay with the agent orchestrator")) {
            return "RETRIEVAL_POLICY_BOUNDARY_DRIFT";
        }
        if (normalized.contains("vector store ownership must stay with the agent orchestrator")) {
            return "VECTOR_STORE_BOUNDARY_DRIFT";
        }
        if (normalized.equals("mcp discovery was not requested")) {
            return "MCP_DISCOVERY_NOT_REQUESTED";
        }
        if (normalized.equals("mcp discovery is not available")) {
            return "MCP_DISCOVERY_UNAVAILABLE";
        }
        if (normalized.contains("mcp discovery endpoint crossed the serving boundary")) {
            return "MCP_DISCOVERY_BOUNDARY_DRIFT";
        }
        if (normalized.contains("mcp discovery must not enable tool execution")) {
            return "MCP_TOOL_EXECUTION_BOUNDARY_DRIFT";
        }
        if (normalized.contains("mcp tool must not be executable from gollek")) {
            return "MCP_TOOL_EXECUTABLE_BOUNDARY_DRIFT";
        }
        if (normalized.equals("tool validation was not requested")) {
            return "TOOL_VALIDATION_NOT_REQUESTED";
        }
        if (normalized.equals("tool definitions are not valid")) {
            return "TOOL_DEFINITIONS_INVALID";
        }
        if (normalized.equals("tool validation must not invoke a model")) {
            return "TOOL_VALIDATION_MODEL_INVOKED";
        }
        if (normalized.equals("tool validation must be validation-only")) {
            return "TOOL_VALIDATION_NOT_VALIDATION_ONLY";
        }
        if (normalized.equals("tool validation must not enable tool execution")) {
            return "TOOL_VALIDATION_TOOL_EXECUTION_BOUNDARY";
        }
        if (normalized.equals("tool validation must not authorize tools")) {
            return "TOOL_AUTHORIZATION_BOUNDARY_DRIFT";
        }
        if (normalized.startsWith("tool schema warning")) {
            String embeddedCode = parenthesizedToken(normalized);
            if (embeddedCode != null) {
                return toolSchemaWarningCode(embeddedCode);
            }
            return "TOOL_SCHEMA_PORTABILITY_WARNING";
        }
        if (normalized.equals("schema portability warning")
                || normalized.equals("tool validation returned portability warnings")) {
            return "TOOL_SCHEMA_PORTABILITY_WARNING";
        }
        if (normalized.equals("request validation was not requested")) {
            return "REQUEST_VALIDATION_NOT_REQUESTED";
        }
        if (normalized.equals("agent request is not valid")) {
            return "AGENT_REQUEST_INVALID";
        }
        if (normalized.equals("request validation must not invoke a model")) {
            return "REQUEST_VALIDATION_MODEL_INVOKED";
        }
        if (normalized.equals("request validation must be validation-only")) {
            return "REQUEST_VALIDATION_NOT_VALIDATION_ONLY";
        }
        if (normalized.equals("request validation must not enable tool execution")) {
            return "REQUEST_VALIDATION_TOOL_EXECUTION_BOUNDARY";
        }
        if (normalized.equals("request validation must not enable retrieval execution")) {
            return "REQUEST_VALIDATION_RETRIEVAL_BOUNDARY";
        }
        if (normalized.equals("embedded tool contract is not valid")) {
            return "EMBEDDED_TOOL_CONTRACT_INVALID";
        }
        return null;
    }

    private static String fallbackCode(String area, String severity) {
        String normalized = normalize(firstNonBlank(area, "preflight")
                + "_"
                + firstNonBlank(severity, "blocking")
                + "_issue");
        return normalized == null ? "PREFLIGHT_ISSUE" : normalized;
    }

    private static CatalogEntry entry(
            String code,
            String area,
            String defaultSeverity,
            String summary,
            String remediation) {
        return new CatalogEntry(code, area, defaultSeverity, summary, remediation);
    }

    private static Map<String, CatalogEntry> catalogByCode(List<CatalogEntry> entries) {
        Map<String, CatalogEntry> out = new LinkedHashMap<>();
        for (CatalogEntry entry : entries) {
            out.put(entry.code(), entry);
        }
        return Collections.unmodifiableMap(out);
    }

    private static CatalogEntry unknownEntry(String code) {
        return new CatalogEntry(
                firstNonBlank(code, "UNKNOWN_READINESS_ISSUE"),
                UNKNOWN,
                UNKNOWN,
                "Unknown Gollek readiness issue code.",
                "Use the raw issue message for details and keep the code for routing or audit logs.");
    }

    private static String normalizeArea(String area) {
        if (area == null || area.isBlank()) {
            return null;
        }
        String normalized = area.trim().toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_|_$", "");
        return normalized.isBlank() ? null : normalized;
    }

    private static String normalizeMessage(String message) {
        if (message == null || message.isBlank()) {
            return null;
        }
        return message.trim().toLowerCase(Locale.ROOT);
    }

    private static String parenthesizedToken(String value) {
        if (value == null) {
            return null;
        }
        int open = value.indexOf('(');
        int close = value.indexOf(')', open + 1);
        if (open < 0 || close <= open + 1) {
            return null;
        }
        return value.substring(open + 1, close);
    }

    private static String severityName(AgentServingReadinessReport.Severity severity) {
        return severity == null ? "blocking" : severity.name().toLowerCase(Locale.ROOT);
    }

    private static String firstNonBlank(String first, String fallback) {
        return first == null || first.isBlank() ? fallback : first;
    }

    public record CatalogEntry(
            String code,
            String area,
            String defaultSeverity,
            String summary,
            String remediation) {

        public CatalogEntry {
            code = firstNonBlank(normalize(code), "UNKNOWN_READINESS_ISSUE");
            area = firstNonBlank(normalizeArea(area), UNKNOWN);
            defaultSeverity = firstNonBlank(normalizeArea(defaultSeverity), UNKNOWN);
            summary = firstNonBlank(summary, "Gollek readiness issue.");
            remediation = firstNonBlank(remediation, "Inspect the readiness issue message for details.");
        }

        public boolean blockingByDefault() {
            return BLOCKING.equals(defaultSeverity);
        }

        public boolean warningByDefault() {
            return WARNING.equals(defaultSeverity);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("code", code);
            out.put("area", area);
            out.put("default_severity", defaultSeverity);
            out.put("summary", summary);
            out.put("remediation", remediation);
            return Collections.unmodifiableMap(out);
        }
    }
}
