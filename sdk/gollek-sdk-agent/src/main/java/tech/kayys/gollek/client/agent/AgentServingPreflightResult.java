package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed result for Gollek's server-side serving preflight.
 *
 * <p>The result preserves the request, raw response, typed readiness report,
 * and route comparison as descriptive metadata. It does not plan, retry,
 * execute tools, run retrieval, update memory, or invoke a model.
 */
public record AgentServingPreflightResult(
        AgentServingPreflightRequest request,
        Map<String, Object> response,
        AgentServingReadinessReport readiness,
        AgentServingRouteComparison routeComparison) {

    public static final String OBJECT = "gollek.agent_preflight_result";

    public AgentServingPreflightResult {
        request = request == null ? AgentServingPreflightRequest.builder().build() : request;
        response = response == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(response));
        readiness = readiness == null ? AgentServingReadinessReport.fromPreflightResponse(response) : readiness;
        routeComparison = routeComparison == null ? readiness.routeComparison(request) : routeComparison;
    }

    public static AgentServingPreflightResult from(
            AgentServingPreflightRequest request,
            Map<String, Object> response) {
        return new AgentServingPreflightResult(request, response, null, null);
    }

    public boolean ready() {
        return readiness.ready();
    }

    public String status() {
        return readiness.status();
    }

    public boolean readyForRequestedRoute() {
        return ready() && routeMatches();
    }

    public String statusForRequestedRoute() {
        return readyForRequestedRoute() ? "ready" : "blocked";
    }

    public AgentServingRoute requestedRoute() {
        return request.route();
    }

    public AgentServingRoute selectedRoute() {
        return readiness.route();
    }

    public boolean routeMatches() {
        return routeComparison.matches();
    }

    public List<String> routeMismatchFields() {
        return routeComparison.mismatchFields();
    }

    public List<AgentServingRouteMismatch> routeMismatches() {
        return routeComparison.mismatches();
    }

    public List<Map<String, Object>> routeMismatchDetails() {
        return routeComparison.mismatchMetadata();
    }

    public List<String> routeMismatchMessages() {
        return routeComparison.mismatchMessages();
    }

    public AgentServingPreflightGate gate() {
        return gate(AgentServingPreflightPolicy.readinessOnly());
    }

    public AgentServingPreflightGate gate(boolean requireRouteMatch, boolean failOnWarnings) {
        return gate(AgentServingPreflightPolicy.of(requireRouteMatch, failOnWarnings));
    }

    public AgentServingPreflightGate gate(AgentServingPreflightPolicy policy) {
        return AgentServingPreflightGate.evaluate(this, policy);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", OBJECT);
        out.put("status", status());
        out.put("ready", ready());
        out.put("status_for_requested_route", statusForRequestedRoute());
        out.put("ready_for_requested_route", readyForRequestedRoute());
        out.put("requested_route", requestedRoute().toMetadata());
        out.put("selected_route", selectedRoute().toMetadata());
        out.put("route_comparison", routeComparison.toMetadata());
        out.put("route_mismatches", routeMismatchDetails());
        out.put("route_mismatch_messages", routeMismatchMessages());
        out.put("readiness_report", readiness.toMetadata());
        return Collections.unmodifiableMap(out);
    }
}
