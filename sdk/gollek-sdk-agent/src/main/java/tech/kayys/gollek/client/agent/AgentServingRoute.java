package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Selected Gollek serving route for an agent-facing integration.
 *
 * <p>This is descriptive metadata only. It does not choose an agent plan,
 * execute tools, run retrieval, update memory, or invoke a model.
 */
public record AgentServingRoute(String surface, String model, String featureProfile) {

    public AgentServingRoute {
        surface = normalizeSurface(surface);
        model = text(model);
        featureProfile = normalizeFeatureProfile(featureProfile);
    }

    public static AgentServingRoute empty() {
        return new AgentServingRoute(null, null, null);
    }

    public static AgentServingRoute of(String surface, String model) {
        return new AgentServingRoute(surface, model, null);
    }

    public static AgentServingRoute of(String surface, String model, String featureProfile) {
        return new AgentServingRoute(surface, model, featureProfile);
    }

    public static AgentServingRoute fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return empty();
        }
        return of(
                text(metadata.get("surface")),
                text(metadata.get("model")),
                text(metadata.get("feature_profile")));
    }

    public boolean known() {
        return hasSurface() || hasModel() || hasFeatureProfile();
    }

    public boolean hasSurface() {
        return surface != null;
    }

    public boolean hasModel() {
        return model != null;
    }

    public boolean hasFeatureProfile() {
        return featureProfile != null;
    }

    public String featureProfileOrDefault() {
        return featureProfile == null ? AgentServingFeatureProfile.DEFAULT_PROFILE : featureProfile;
    }

    public boolean matchesSurface(String expectedSurface) {
        String normalized = normalizeSurface(expectedSurface);
        return surface != null && surface.equals(normalized);
    }

    public boolean matches(AgentServingRoute expected) {
        return mismatchFields(expected).isEmpty();
    }

    public AgentServingRouteComparison comparison(AgentServingRoute expected) {
        return AgentServingRouteComparison.compare(expected, this);
    }

    public List<String> mismatchFields(AgentServingRoute expected) {
        if (expected == null || !expected.known()) {
            return List.of();
        }
        List<String> mismatches = new ArrayList<>();
        if (expected.hasSurface() && !expected.surface().equals(surface)) {
            mismatches.add("surface");
        }
        if (expected.hasModel() && !expected.model().equals(model)) {
            mismatches.add("model");
        }
        if (expected.hasFeatureProfile()
                && !expected.featureProfileOrDefault().equals(featureProfileOrDefault())) {
            mismatches.add("feature_profile");
        }
        return List.copyOf(mismatches);
    }

    public AgentServingRoute withFeatureProfile(String featureProfile) {
        return new AgentServingRoute(surface, model, featureProfile);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        putIfPresent(out, "feature_profile", featureProfile);
        putIfPresent(out, "surface", surface);
        putIfPresent(out, "model", model);
        return Collections.unmodifiableMap(out);
    }

    public static String normalizeSurface(String surface) {
        if (surface == null || surface.isBlank()) {
            return null;
        }
        String normalized = surface.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "chat", "chat_completions", "chat-completions" -> "chat";
            case "response", "responses" -> "responses";
            case "embedding", "embeddings" -> "embeddings";
            default -> normalized;
        };
    }

    private static String normalizeFeatureProfile(String featureProfile) {
        if (featureProfile == null || featureProfile.isBlank()) {
            return null;
        }
        return AgentServingFeatureProfile.normalizeName(featureProfile);
    }

    private static void putIfPresent(Map<String, Object> out, String key, String value) {
        if (value != null) {
            out.put(key, value);
        }
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }
}
