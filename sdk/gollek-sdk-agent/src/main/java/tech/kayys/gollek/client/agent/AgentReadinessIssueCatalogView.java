package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed view over {@code /v1/agent/readiness/issues}.
 *
 * <p>The catalog is serving metadata only. It helps dashboards and CI explain
 * stable readiness codes; it does not run validation, execute tools, run
 * retrieval, invoke a model, or make orchestration decisions.
 */
public final class AgentReadinessIssueCatalogView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();
    private static final String SERVICE_ROLE_INFERENCE_SERVING_ENGINE = "inference_serving_engine";

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentReadinessIssueCatalogView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentReadinessIssueCatalogView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentReadinessIssueCatalogView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentReadinessIssueCatalogView(mapper, node);
    }

    public static AgentReadinessIssueCatalogView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentReadinessIssueCatalogView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentReadinessIssueCatalogView(objectMapper, response);
    }

    public String object() {
        return text(raw.path("object"));
    }

    public String version() {
        return text(raw.path("version"));
    }

    public String serviceRole() {
        return text(raw.path("service_role"));
    }

    public boolean isInferenceServingEngine() {
        return SERVICE_ROLE_INFERENCE_SERVING_ENGINE.equals(serviceRole());
    }

    public int count() {
        int advertised = raw.path("count").asInt(-1);
        return advertised >= 0 ? advertised : entries().size();
    }

    public boolean validationOnly() {
        return raw.path("boundary").path("validation_only").asBoolean(true);
    }

    public boolean modelInvoked() {
        return raw.path("boundary").path("model_invoked").asBoolean(false);
    }

    public boolean toolExecutionEnabled() {
        return raw.path("boundary").path("tool_execution").asBoolean(false);
    }

    public boolean retrievalExecutionEnabled() {
        return raw.path("boundary").path("retrieval_execution").asBoolean(false);
    }

    public boolean toolAuthorizationEnabled() {
        return raw.path("boundary").path("tool_authorization").asBoolean(false);
    }

    public List<AgentReadinessIssueCodes.CatalogEntry> entries() {
        List<AgentReadinessIssueCodes.CatalogEntry> out = new ArrayList<>();
        JsonNode items = raw.path("items");
        if (items.isArray()) {
            for (JsonNode item : items) {
                entry(item).ifPresent(out::add);
            }
        }
        if (out.isEmpty() && raw.path("by_code").isObject()) {
            raw.path("by_code").fields().forEachRemaining(item -> entry(item.getValue()).ifPresent(out::add));
        }
        return List.copyOf(out);
    }

    public Map<String, AgentReadinessIssueCodes.CatalogEntry> byCode() {
        Map<String, AgentReadinessIssueCodes.CatalogEntry> out = new LinkedHashMap<>();
        for (AgentReadinessIssueCodes.CatalogEntry entry : entries()) {
            out.put(entry.code(), entry);
        }
        return Collections.unmodifiableMap(out);
    }

    public Map<String, List<AgentReadinessIssueCodes.CatalogEntry>> byArea() {
        Map<String, List<AgentReadinessIssueCodes.CatalogEntry>> grouped = new LinkedHashMap<>();
        for (AgentReadinessIssueCodes.CatalogEntry entry : entries()) {
            grouped.computeIfAbsent(entry.area(), ignored -> new ArrayList<>()).add(entry);
        }
        Map<String, List<AgentReadinessIssueCodes.CatalogEntry>> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<AgentReadinessIssueCodes.CatalogEntry>> entry : grouped.entrySet()) {
            out.put(entry.getKey(), List.copyOf(entry.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }

    public Optional<AgentReadinessIssueCodes.CatalogEntry> find(String code) {
        String normalized = AgentReadinessIssueCodes.normalize(code);
        return normalized == null ? Optional.empty() : Optional.ofNullable(byCode().get(normalized));
    }

    public boolean hasCode(String code) {
        return find(code).isPresent();
    }

    public List<AgentReadinessIssueCodes.CatalogEntry> entriesForArea(String area) {
        String normalized = normalizeArea(area);
        if (normalized == null) {
            return List.of();
        }
        return byArea().getOrDefault(normalized, List.of());
    }

    public boolean servingBoundaryValid() {
        return servingBoundaryIssues().isEmpty();
    }

    public List<String> servingBoundaryIssues() {
        List<String> issues = new ArrayList<>();
        if (!isInferenceServingEngine()) {
            issues.add("service_role must be inference_serving_engine");
        }
        if (!validationOnly()) {
            issues.add("readiness issue catalog must be validation-only");
        }
        if (modelInvoked()) {
            issues.add("readiness issue catalog must not invoke a model");
        }
        if (toolExecutionEnabled()) {
            issues.add("readiness issue catalog must not execute tools");
        }
        if (retrievalExecutionEnabled()) {
            issues.add("readiness issue catalog must not run retrieval");
        }
        if (toolAuthorizationEnabled()) {
            issues.add("readiness issue catalog must not authorize tools");
        }
        return List.copyOf(issues);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> raw() {
        return objectMapper.convertValue(raw, Map.class);
    }

    public JsonNode rawNode() {
        return raw;
    }

    private static Optional<AgentReadinessIssueCodes.CatalogEntry> entry(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Optional.empty();
        }
        String code = text(node.path("code"));
        if (code == null) {
            return Optional.empty();
        }
        return Optional.of(new AgentReadinessIssueCodes.CatalogEntry(
                code,
                text(node.path("area")),
                text(node.path("default_severity")),
                text(node.path("summary")),
                text(node.path("remediation"))));
    }

    private static String normalizeArea(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value.trim()
                .toLowerCase(java.util.Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_|_$", "");
        return normalized.isBlank() ? null : normalized;
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
