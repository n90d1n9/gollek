package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Read-only view over contract-version and feature-flag negotiation metadata.
 */
public final class AgentFeatureNegotiation {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();

    private final ObjectMapper objectMapper;
    private final JsonNode raw;
    private final JsonNode negotiation;

    private AgentFeatureNegotiation(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
        JsonNode nested = this.raw.path("feature_negotiation");
        this.negotiation = nested.isObject() ? nested : this.raw;
    }

    public static AgentFeatureNegotiation from(Map<String, Object> payload) {
        return from(payload, null);
    }

    public static AgentFeatureNegotiation from(Map<String, Object> payload, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = payload == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(payload, JsonNode.class);
        return new AgentFeatureNegotiation(mapper, node);
    }

    public static AgentFeatureNegotiation from(JsonNode payload) {
        return from(payload, null);
    }

    public static AgentFeatureNegotiation from(JsonNode payload, ObjectMapper objectMapper) {
        return new AgentFeatureNegotiation(objectMapper, payload);
    }

    public String mode() {
        return text(negotiation.path("mode"));
    }

    public String featureNamespace() {
        return text(negotiation.path("feature_namespace"));
    }

    public String contractVersion() {
        return firstText(
                negotiation.path("contract_version"),
                raw.path("contract_version"),
                raw.path("version"));
    }

    public String minimumClientContractVersion() {
        return firstText(
                negotiation.path("minimum_client_contract_version"),
                raw.path("minimum_client_contract_version"),
                negotiation.path("contract_version"),
                raw.path("contract_version"),
                raw.path("version"));
    }

    public List<String> supportedContractVersions() {
        List<String> versions = union(
                strings(negotiation.path("supported_contract_versions")),
                strings(raw.path("supported_contract_versions")));
        if (!versions.isEmpty()) {
            return versions;
        }
        String version = contractVersion();
        return version == null ? List.of() : List.of(version);
    }

    public boolean supportsContractVersion(String version) {
        return contains(supportedContractVersions(), version);
    }

    public List<String> requiredFeatures() {
        return strings(negotiation.path("required_features"));
    }

    public List<String> optionalFeatures() {
        return strings(negotiation.path("optional_features"));
    }

    public List<String> allFeatures() {
        return union(
                strings(raw.path("compatibility")),
                strings(negotiation.path("all_features")),
                requiredFeatures(),
                optionalFeatures());
    }

    public boolean supportsFeature(String feature) {
        return contains(allFeatures(), feature);
    }

    public String defaultFeatureProfile() {
        String profile = text(negotiation.path("default_feature_profile"));
        return profile == null ? AgentServingFeatureProfile.DEFAULT_PROFILE : profile;
    }

    public List<String> supportedFeatureProfiles() {
        List<String> profiles = strings(negotiation.path("supported_feature_profiles"));
        return profiles.isEmpty() ? AgentServingFeatureProfile.supportedProfileNames() : profiles;
    }

    public boolean supportsFeatureProfile(String profile) {
        return contains(supportedFeatureProfiles(), AgentServingFeatureProfile.normalizeName(profile));
    }

    public Optional<Map<String, Object>> featureProfile(String profile) {
        String normalized = AgentServingFeatureProfile.normalizeName(profile);
        for (Map<String, Object> item : featureProfiles()) {
            Object name = item.get("name");
            if (normalized.equals(name == null ? null : name.toString())) {
                return Optional.of(item);
            }
        }
        return Optional.empty();
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> featureProfiles() {
        JsonNode profiles = negotiation.path("feature_profiles");
        if (!profiles.isArray()) {
            return AgentServingFeatureProfile.catalogMaps();
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (JsonNode item : profiles) {
            if (item.isObject()) {
                out.add(objectMapper.convertValue(item, LinkedHashMap.class));
            }
        }
        return List.copyOf(out);
    }

    public boolean supportsAll(Collection<String> features) {
        return unsupportedFeatures(features).isEmpty();
    }

    public List<String> unsupportedFeatures(Collection<String> features) {
        if (features == null || features.isEmpty()) {
            return List.of();
        }
        List<String> available = allFeatures();
        List<String> missing = new ArrayList<>();
        for (String feature : features) {
            if (!contains(available, feature)) {
                missing.add(feature);
            }
        }
        return List.copyOf(missing);
    }

    public boolean stableV1Supported() {
        return supportsContractVersion(AgentServingFeatureCatalog.CONTRACT_VERSION);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> raw() {
        return objectMapper.convertValue(negotiation, Map.class);
    }

    public JsonNode rawNode() {
        return negotiation;
    }

    @SafeVarargs
    private static List<String> union(List<String>... lists) {
        Set<String> out = new LinkedHashSet<>();
        for (List<String> list : lists) {
            if (list != null) {
                out.addAll(list);
            }
        }
        return List.copyOf(out);
    }

    private static boolean contains(List<String> values, String value) {
        return value != null && values.contains(value);
    }

    private static List<String> strings(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (JsonNode item : node) {
            String value = text(item);
            if (value != null) {
                out.add(value);
            }
        }
        return List.copyOf(out);
    }

    private static String firstText(JsonNode... nodes) {
        for (JsonNode node : nodes) {
            String value = text(node);
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
