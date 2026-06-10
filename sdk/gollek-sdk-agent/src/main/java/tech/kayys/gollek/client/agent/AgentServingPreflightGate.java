package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Evaluates whether a typed preflight result satisfies caller-owned gate policy.
 *
 * <p>The gate only summarizes readiness, route-match, and warning policy. It
 * does not select routes, retry requests, execute tools, run retrieval, or
 * invoke a model.
 */
public record AgentServingPreflightGate(
        AgentServingPreflightResult result,
        AgentServingPreflightPolicy policy) {

    public static final String OBJECT = "gollek.agent_preflight_gate";
    public static final String REASON_READINESS_BLOCKED = "readiness_blocked";
    public static final String REASON_ROUTE_MISMATCH = "route_mismatch";
    public static final String REASON_WARNINGS_PRESENT = "warnings_present";

    public AgentServingPreflightGate {
        result = result == null ? AgentServingPreflightResult.from(null, Map.of()) : result;
        policy = policy == null ? AgentServingPreflightPolicy.readinessOnly() : policy;
    }

    public static AgentServingPreflightGate evaluate(AgentServingPreflightResult result) {
        return evaluate(result, AgentServingPreflightPolicy.readinessOnly());
    }

    public static AgentServingPreflightGate evaluate(
            AgentServingPreflightResult result,
            boolean requireRouteMatch,
            boolean failOnWarnings) {
        return evaluate(result, AgentServingPreflightPolicy.of(requireRouteMatch, failOnWarnings));
    }

    public static AgentServingPreflightGate evaluate(
            AgentServingPreflightResult result,
            AgentServingPreflightPolicy policy) {
        return new AgentServingPreflightGate(result, policy);
    }

    public boolean requireRouteMatch() {
        return policy.requireRouteMatch();
    }

    public boolean failOnWarnings() {
        return policy.failOnWarnings();
    }

    public boolean ready() {
        return blockingReasons().isEmpty();
    }

    public String status() {
        return ready() ? "ready" : "blocked";
    }

    public List<String> blockingReasons() {
        List<String> reasons = new ArrayList<>();
        if (!result.ready()) {
            reasons.add(REASON_READINESS_BLOCKED);
        }
        if (requireRouteMatch() && !result.routeMatches()) {
            reasons.add(REASON_ROUTE_MISMATCH);
        }
        if (failOnWarnings() && result.readiness().hasWarnings()) {
            reasons.add(REASON_WARNINGS_PRESENT);
        }
        return List.copyOf(reasons);
    }

    public List<String> blockingMessages() {
        List<String> messages = new ArrayList<>();
        for (String reason : blockingReasons()) {
            messages.add(messageFor(reason));
        }
        return List.copyOf(messages);
    }

    public boolean blocksOnRouteMismatch() {
        return blockingReasons().contains(REASON_ROUTE_MISMATCH);
    }

    public boolean blocksOnWarnings() {
        return blockingReasons().contains(REASON_WARNINGS_PRESENT);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", OBJECT);
        out.put("status", status());
        out.put("ready", ready());
        out.put("policy", policy.toMetadata());
        out.put("route_match_required", requireRouteMatch());
        out.put("warnings_blocking", failOnWarnings());
        out.put("blocking_reasons", blockingReasons());
        out.put("blocking_messages", blockingMessages());
        out.put("route_mismatch_fields", result.routeMismatchFields());
        out.put("route_mismatches", result.routeMismatchDetails());
        out.put("route_mismatch_messages", result.routeMismatchMessages());
        out.put("warning_count", result.readiness().warnings().size());
        return Collections.unmodifiableMap(out);
    }

    private String messageFor(String reason) {
        return switch (reason) {
            case REASON_READINESS_BLOCKED -> "preflight readiness is blocked";
            case REASON_ROUTE_MISMATCH -> routeMismatchMessage();
            case REASON_WARNINGS_PRESENT -> "preflight warnings are present";
            default -> reason;
        };
    }

    private String routeMismatchMessage() {
        List<String> fields = result.routeMismatchFields();
        if (fields.isEmpty()) {
            return "selected route differs from requested route";
        }
        List<String> messages = result.routeMismatchMessages();
        if (messages.isEmpty()) {
            return "selected route differs from requested route: " + String.join(",", fields);
        }
        return "selected route differs from requested route: " + String.join("; ", messages);
    }
}
