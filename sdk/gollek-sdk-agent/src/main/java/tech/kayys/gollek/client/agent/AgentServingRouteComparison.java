package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Describes whether a Gollek-selected serving route matches a requested route.
 *
 * <p>This is comparison metadata for callers and telemetry. It does not choose
 * routes, retry requests, execute tools, or invoke a model.
 */
public record AgentServingRouteComparison(
        AgentServingRoute expected,
        AgentServingRoute actual,
        List<String> mismatchFields) {

    public AgentServingRouteComparison {
        expected = expected == null ? AgentServingRoute.empty() : expected;
        actual = actual == null ? AgentServingRoute.empty() : actual;
        mismatchFields = mismatchFields == null
                ? actual.mismatchFields(expected)
                : List.copyOf(mismatchFields);
    }

    public static AgentServingRouteComparison compare(
            AgentServingRoute expected,
            AgentServingRoute actual) {
        return new AgentServingRouteComparison(expected, actual, null);
    }

    public boolean matches() {
        return mismatchFields.isEmpty();
    }

    public boolean hasMismatches() {
        return !matches();
    }

    public List<AgentServingRouteMismatch> mismatches() {
        return mismatchFields.stream()
                .map(field -> new AgentServingRouteMismatch(
                        field,
                        valueFor(expected, field),
                        valueFor(actual, field)))
                .toList();
    }

    public List<Map<String, Object>> mismatchMetadata() {
        return mismatches().stream()
                .map(AgentServingRouteMismatch::toMetadata)
                .toList();
    }

    public List<String> mismatchMessages() {
        return mismatches().stream()
                .map(AgentServingRouteMismatch::message)
                .toList();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("matches", matches());
        out.put("mismatch_fields", mismatchFields);
        out.put("mismatches", mismatchMetadata());
        out.put("mismatch_messages", mismatchMessages());
        out.put("expected", expected.toMetadata());
        out.put("actual", actual.toMetadata());
        return Collections.unmodifiableMap(out);
    }

    private static String valueFor(AgentServingRoute route, String field) {
        if (route == null || field == null) {
            return null;
        }
        return switch (field) {
            case "surface" -> route.surface();
            case "model" -> route.model();
            case "feature_profile" -> route.featureProfileOrDefault();
            default -> null;
        };
    }
}
