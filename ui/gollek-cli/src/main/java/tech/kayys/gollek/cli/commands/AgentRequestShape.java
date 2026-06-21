package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Local request-shaping helpers for agent CLI commands.
 *
 * <p>This class only builds validation/preflight payload shapes. It does not
 * execute tools, run retrieval, or model an agent workflow.
 */
final class AgentRequestShape {
    static final String DEFAULT_SURFACE = "chat";

    private AgentRequestShape() {
    }

    static String resolveSurface(
            String defaultSurface,
            String explicitSurface,
            String inputSurface,
            String requestSurface,
            Map<String, Object> request,
            String featureProfile) {
        return normalizeSurface(firstNonBlank(
                explicitSurface,
                inputSurface,
                requestSurface,
                inferSurface(request),
                defaultSurfaceForProfile(featureProfile, defaultSurface),
                defaultSurface));
    }

    static Map<String, Object> generatedRequest(
            String surface,
            String featureProfile,
            String prompt,
            String defaultPrompt,
            String systemPrompt,
            String defaultSystemPrompt) {
        String effectiveSurface = normalizeSurface(firstNonBlank(
                surface,
                defaultSurfaceForProfile(featureProfile, DEFAULT_SURFACE),
                DEFAULT_SURFACE));
        String input = firstNonBlank(prompt, defaultPrompt);
        Map<String, Object> request = new LinkedHashMap<>();
        if ("embeddings".equals(effectiveSurface)) {
            request.put("input", List.of(input));
        } else if ("responses".equals(effectiveSurface)) {
            request.put("input", input);
        } else {
            List<Map<String, Object>> messages = new ArrayList<>();
            messages.add(Map.of("role", "system", "content", firstNonBlank(systemPrompt, defaultSystemPrompt)));
            messages.add(Map.of("role", "user", "content", input));
            request.put("messages", messages);
        }
        return request;
    }

    static String normalizeSurface(String value) {
        String normalized = firstNonBlank(value, DEFAULT_SURFACE).toLowerCase(Locale.ROOT).trim();
        if ("response".equals(normalized)) {
            return "responses";
        }
        if ("embedding".equals(normalized)) {
            return "embeddings";
        }
        return normalized;
    }

    static String inferSurface(Map<String, Object> request) {
        if (request == null) {
            return null;
        }
        if (request.containsKey("messages")) {
            return "chat";
        }
        if (request.containsKey("input")) {
            Object input = request.get("input");
            return input instanceof List<?> ? "embeddings" : "responses";
        }
        return null;
    }

    static String defaultSurfaceForProfile(String profileName, String defaultSurface) {
        String normalized = AgentServingFeatureProfile.normalizeName(profileName);
        return AgentServingFeatureProfile.find(normalized)
                .filter(profile -> !profile.surfaces().isEmpty())
                .map(profile -> profile.surfaces().get(0))
                .orElse(firstNonBlank(defaultSurface, DEFAULT_SURFACE));
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return null;
    }
}
