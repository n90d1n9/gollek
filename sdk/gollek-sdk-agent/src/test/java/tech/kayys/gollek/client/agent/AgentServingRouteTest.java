package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingRouteTest {

    @Test
    void normalizesServingRouteMetadata() {
        AgentServingRoute route = AgentServingRoute.of(
                "chat-completions",
                "demo-chat",
                "chatAgent");

        assertTrue(route.known());
        assertTrue(route.hasSurface());
        assertTrue(route.hasModel());
        assertTrue(route.hasFeatureProfile());
        assertEquals("chat", route.surface());
        assertEquals("demo-chat", route.model());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, route.featureProfile());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, route.featureProfileOrDefault());
        assertTrue(route.matchesSurface("chat_completions"));
        assertFalse(route.matchesSurface("embeddings"));
        assertEquals(Map.of(
                        "feature_profile", AgentServingFeatureProfile.CHAT_AGENT,
                        "surface", "chat",
                        "model", "demo-chat"),
                route.toMetadata());
    }

    @Test
    void keepsEmptyRouteMetadataOutOfMaps() {
        AgentServingRoute route = AgentServingRoute.empty();

        assertFalse(route.known());
        assertFalse(route.hasSurface());
        assertFalse(route.hasModel());
        assertFalse(route.hasFeatureProfile());
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE, route.featureProfileOrDefault());
        assertEquals(Map.of(), route.toMetadata());
    }

    @Test
    void readsRouteFromMetadataMap() {
        AgentServingRoute route = AgentServingRoute.fromMetadata(Map.of(
                "surface", "response",
                "model", "demo-response",
                "feature_profile", "responsesAgent"));

        assertEquals("responses", route.surface());
        assertEquals("demo-response", route.model());
        assertEquals(AgentServingFeatureProfile.RESPONSES_AGENT, route.featureProfile());
        assertTrue(route.matchesSurface("responses"));
    }

    @Test
    void comparesRequestedRouteAgainstSelectedRoute() {
        AgentServingRoute selected = AgentServingRoute.of(
                "chat",
                "demo-chat",
                AgentServingFeatureProfile.CHAT_AGENT);

        assertTrue(selected.matches(AgentServingRoute.of(
                "chat-completions",
                "demo-chat",
                "chatAgent")));
        assertEquals(List.of(), selected.mismatchFields(AgentServingRoute.of(
                "chat-completions",
                "demo-chat",
                "chatAgent")));
        assertEquals(List.of("surface", "model", "feature_profile"),
                selected.mismatchFields(AgentServingRoute.of(
                        "embeddings",
                        "demo-embed",
                        AgentServingFeatureProfile.EMBEDDING_RAG)));
        assertTrue(selected.matches(AgentServingRoute.of(null, null)));
    }

    @Test
    void createsStructuredComparisonMetadata() {
        AgentServingRoute selected = AgentServingRoute.of(
                "chat",
                "demo-chat",
                AgentServingFeatureProfile.CHAT_AGENT);
        AgentServingRoute expected = AgentServingRoute.of(
                "responses",
                "demo-response",
                AgentServingFeatureProfile.RESPONSES_AGENT);

        AgentServingRouteComparison comparison = selected.comparison(expected);

        assertFalse(comparison.matches());
        assertTrue(comparison.hasMismatches());
        assertEquals(expected, comparison.expected());
        assertEquals(selected, comparison.actual());
        assertEquals(List.of("surface", "model", "feature_profile"), comparison.mismatchFields());
        assertEquals(Map.of(
                        "matches", false,
                        "mismatch_fields", List.of("surface", "model", "feature_profile"),
                        "expected", expected.toMetadata(),
                        "actual", selected.toMetadata()),
                comparison.toMetadata());
    }

    @Test
    void comparisonTreatsMissingExpectedRouteAsMatch() {
        AgentServingRoute selected = AgentServingRoute.of("chat", "demo-chat");

        AgentServingRouteComparison comparison = selected.comparison(null);

        assertTrue(comparison.matches());
        assertFalse(comparison.hasMismatches());
        assertEquals(AgentServingRoute.empty(), comparison.expected());
        assertEquals(selected, comparison.actual());
        assertEquals(List.of(), comparison.mismatchFields());
        assertEquals(Map.of(
                        "matches", true,
                        "mismatch_fields", List.of(),
                        "expected", Map.of(),
                        "actual", selected.toMetadata()),
                comparison.toMetadata());
    }

    @Test
    void treatsMissingFeatureProfileAsDefaultForComparison() {
        AgentServingRoute selected = AgentServingRoute.of("responses", "demo-response");

        assertTrue(selected.matches(AgentServingRoute.of(
                "responses",
                "demo-response",
                AgentServingFeatureProfile.DEFAULT_PROFILE)));
        assertEquals(List.of("feature_profile"),
                selected.mismatchFields(AgentServingRoute.of(
                        "responses",
                        "demo-response",
                        AgentServingFeatureProfile.RESPONSES_AGENT)));
    }

    @Test
    void updatesFeatureProfileWithoutChangingSelectedRoute() {
        AgentServingRoute route = AgentServingRoute.of("embedding", "demo-embed")
                .withFeatureProfile("embedding-rag");

        assertEquals("embeddings", route.surface());
        assertEquals("demo-embed", route.model());
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, route.featureProfile());
    }
}
