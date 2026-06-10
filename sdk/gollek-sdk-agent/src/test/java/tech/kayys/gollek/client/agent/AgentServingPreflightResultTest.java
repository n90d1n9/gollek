package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingPreflightResultTest {

    @Test
    void wrapsRequestResponseReadinessAndRouteComparison() {
        AgentServingPreflightRequest request = AgentServingPreflightRequest.builder()
                .route(AgentServingRoute.of("chat-completions", "demo-chat", "chatAgent"))
                .request(Map.of("model", "demo-chat"))
                .build();
        Map<String, Object> response = Map.of(
                "object", "gollek.agent_preflight",
                "status", "ready",
                "ready", true,
                "surface", "chat",
                "model", "demo-chat",
                "feature_profile", AgentServingFeatureProfile.CHAT_AGENT,
                "issues", List.of());

        AgentServingPreflightResult result = AgentServingPreflightResult.from(request, response);

        assertTrue(result.ready());
        assertEquals("ready", result.status());
        assertTrue(result.readyForRequestedRoute());
        assertEquals("ready", result.statusForRequestedRoute());
        assertEquals(request, result.request());
        assertEquals(response, result.response());
        assertEquals(request.route(), result.requestedRoute());
        assertEquals(request.route(), result.selectedRoute());
        assertTrue(result.routeMatches());
        assertEquals(List.of(), result.routeMismatchFields());
        assertEquals(List.of(), result.routeMismatches());
        assertEquals(List.of(), result.routeMismatchDetails());
        assertEquals(List.of(), result.routeMismatchMessages());
        assertTrue(result.routeComparison().matches());
        assertEquals(request.route(), result.routeComparison().expected());
        assertEquals(request.route(), result.routeComparison().actual());
        assertThrows(UnsupportedOperationException.class,
                () -> result.response().put("status", "mutated"));
        assertEquals(Map.ofEntries(
                        Map.entry("object", AgentServingPreflightResult.OBJECT),
                        Map.entry("status", "ready"),
                        Map.entry("ready", true),
                        Map.entry("status_for_requested_route", "ready"),
                        Map.entry("ready_for_requested_route", true),
                        Map.entry("requested_route", request.route().toMetadata()),
                        Map.entry("selected_route", request.route().toMetadata()),
                        Map.entry("route_comparison", result.routeComparison().toMetadata()),
                        Map.entry("route_mismatches", List.of()),
                        Map.entry("route_mismatch_messages", List.of()),
                        Map.entry("readiness_report", result.readiness().toMetadata())),
                result.toMetadata());
    }

    @Test
    void exposesRouteMismatchesWithoutChangingReadiness() {
        AgentServingPreflightRequest request = AgentServingPreflightRequest.builder()
                .route(AgentServingRoute.of("embeddings", "requested-embed", AgentServingFeatureProfile.EMBEDDING_RAG))
                .build();
        Map<String, Object> response = Map.of(
                "status", "ready",
                "ready", true,
                "surface", "embeddings",
                "model", "selected-embed",
                "feature_profile", AgentServingFeatureProfile.EMBEDDING_RAG,
                "issues", List.of());

        AgentServingPreflightResult result = AgentServingPreflightResult.from(request, response);

        assertTrue(result.ready());
        assertFalse(result.readyForRequestedRoute());
        assertEquals("blocked", result.statusForRequestedRoute());
        assertFalse(result.routeMatches());
        assertEquals(List.of("model"), result.routeMismatchFields());
        assertEquals(List.of("model requested=requested-embed selected=selected-embed"),
                result.routeMismatchMessages());
        assertEquals(List.of(Map.of(
                        "field", "model",
                        "requested", "requested-embed",
                        "selected", "selected-embed",
                        "message", "model requested=requested-embed selected=selected-embed")),
                result.routeMismatchDetails());
        assertEquals("requested-embed", result.routeComparison().expected().model());
        assertEquals("selected-embed", result.routeComparison().actual().model());
    }
}
