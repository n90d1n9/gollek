package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Caller-owned policy for deciding whether a serving preflight is acceptable.
 *
 * <p>The policy describes gate requirements only. It does not choose routes,
 * retry requests, execute tools, run retrieval, or invoke a model.
 */
public record AgentServingPreflightPolicy(
        String profile,
        boolean requireRouteMatch,
        boolean failOnWarnings) {

    public static final String OBJECT = "gollek.agent_preflight_policy";
    public static final String PROFILE_READINESS_ONLY = "readiness_only";
    public static final String PROFILE_REQUESTED_ROUTE = "requested_route";
    public static final String PROFILE_CLEAN = "clean";
    public static final String PROFILE_CUSTOM = "custom";

    public AgentServingPreflightPolicy {
        profile = normalizedProfile(profile, requireRouteMatch, failOnWarnings);
    }

    public static AgentServingPreflightPolicy readinessOnly() {
        return new AgentServingPreflightPolicy(PROFILE_READINESS_ONLY, false, false);
    }

    public static AgentServingPreflightPolicy requestedRoute() {
        return new AgentServingPreflightPolicy(PROFILE_REQUESTED_ROUTE, true, false);
    }

    public static AgentServingPreflightPolicy clean() {
        return new AgentServingPreflightPolicy(PROFILE_CLEAN, true, true);
    }

    public static AgentServingPreflightPolicy of(boolean requireRouteMatch, boolean failOnWarnings) {
        return new AgentServingPreflightPolicy(null, requireRouteMatch, failOnWarnings);
    }

    public static List<String> supportedProfileNames() {
        return List.of(PROFILE_READINESS_ONLY, PROFILE_REQUESTED_ROUTE, PROFILE_CLEAN);
    }

    public static boolean supportsProfile(String profile) {
        String normalized = normalizeProfileName(profile);
        if (normalized == null || normalized.isBlank()) {
            return true;
        }
        return switch (normalized) {
            case "default", PROFILE_READINESS_ONLY, "readiness-only",
                    PROFILE_REQUESTED_ROUTE, "requested-route", "route", "strict_route", "strict-route",
                    PROFILE_CLEAN, "ci", "strict" -> true;
            default -> false;
        };
    }

    public static AgentServingPreflightPolicy requireProfile(String profile) {
        if (!supportsProfile(profile)) {
            throw new IllegalArgumentException("Unsupported preflight policy profile: " + profile
                    + ". Use one of: " + String.join(", ", supportedProfileNames()));
        }
        return profile(profile);
    }

    public static AgentServingPreflightPolicy profile(String profile) {
        String normalized = normalizeProfileName(profile);
        if (normalized == null || normalized.isBlank()) {
            return readinessOnly();
        }
        return switch (normalized) {
            case "default", PROFILE_READINESS_ONLY, "readiness-only" -> readinessOnly();
            case PROFILE_REQUESTED_ROUTE, "requested-route", "route", "strict_route", "strict-route" -> requestedRoute();
            case PROFILE_CLEAN, "ci", "strict" -> clean();
            default -> new AgentServingPreflightPolicy(normalized, false, false);
        };
    }

    public static AgentServingPreflightPolicy fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return readinessOnly();
        }
        String profile = text(metadata.get("profile"));
        boolean requireRouteMatch = bool(metadata.get("route_match_required"));
        boolean failOnWarnings = bool(metadata.get("warnings_blocking"));
        if (!hasText(profile) && !requireRouteMatch && !failOnWarnings) {
            return readinessOnly();
        }
        if (hasText(profile)) {
            AgentServingPreflightPolicy named = profile(profile);
            boolean effectiveRequireRouteMatch = requireRouteMatch || named.requireRouteMatch();
            boolean effectiveFailOnWarnings = failOnWarnings || named.failOnWarnings();
            if (effectiveRequireRouteMatch == named.requireRouteMatch()
                    && effectiveFailOnWarnings == named.failOnWarnings()) {
                return named;
            }
            return of(effectiveRequireRouteMatch, effectiveFailOnWarnings);
        }
        return of(requireRouteMatch, failOnWarnings);
    }

    public AgentServingPreflightGate evaluate(AgentServingPreflightResult result) {
        return AgentServingPreflightGate.evaluate(result, this);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", OBJECT);
        out.put("profile", profile);
        out.put("route_match_required", requireRouteMatch);
        out.put("warnings_blocking", failOnWarnings);
        return Collections.unmodifiableMap(out);
    }

    private static String normalizedProfile(
            String profile,
            boolean requireRouteMatch,
            boolean failOnWarnings) {
        String normalized = normalizeProfileName(profile);
        if (!hasText(normalized)) {
            if (requireRouteMatch && failOnWarnings) {
                return PROFILE_CLEAN;
            }
            if (requireRouteMatch) {
                return PROFILE_REQUESTED_ROUTE;
            }
            if (failOnWarnings) {
                return PROFILE_CUSTOM;
            }
            return PROFILE_READINESS_ONLY;
        }
        return normalized;
    }

    private static String normalizeProfileName(String profile) {
        if (profile == null || profile.isBlank()) {
            return null;
        }
        return profile.trim().toLowerCase(Locale.ROOT);
    }

    private static boolean bool(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        return value != null && Boolean.parseBoolean(String.valueOf(value));
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
