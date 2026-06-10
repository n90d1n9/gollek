package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingPreflightGateTest {

    @Test
    void defaultsToReadinessOnlyGate() {
        AgentServingPreflightResult result = AgentServingPreflightResult.from(
                request("demo-chat"),
                response("demo-chat", true, List.of(), List.of()));

        AgentServingPreflightGate gate = result.gate();

        assertTrue(gate.ready());
        assertEquals("ready", gate.status());
        assertEquals(AgentServingPreflightPolicy.PROFILE_READINESS_ONLY, gate.policy().profile());
        assertEquals(List.of(), gate.blockingReasons());
        assertEquals(AgentServingPreflightGate.OBJECT, gate.toMetadata().get("object"));
        assertEquals(AgentServingPreflightPolicy.readinessOnly().toMetadata(), gate.toMetadata().get("policy"));
        assertEquals(false, gate.toMetadata().get("route_match_required"));
        assertEquals(false, gate.toMetadata().get("warnings_blocking"));
    }

    @Test
    void routeMismatchBlocksOnlyWhenRequired() {
        AgentServingPreflightResult result = AgentServingPreflightResult.from(
                request("requested-chat"),
                response("selected-chat", true, List.of(), List.of()));

        AgentServingPreflightGate permissive = result.gate();
        AgentServingPreflightGate strict = result.gate(AgentServingPreflightPolicy.requestedRoute());

        assertTrue(permissive.ready());
        assertFalse(strict.ready());
        assertEquals("blocked", strict.status());
        assertEquals(AgentServingPreflightPolicy.PROFILE_REQUESTED_ROUTE, strict.policy().profile());
        assertTrue(strict.blocksOnRouteMismatch());
        assertEquals(List.of(AgentServingPreflightGate.REASON_ROUTE_MISMATCH), strict.blockingReasons());
        assertEquals(List.of("selected route differs from requested route: model requested=requested-chat selected=selected-chat"),
                strict.blockingMessages());
        assertEquals(List.of("model"), strict.toMetadata().get("route_mismatch_fields"));
        assertEquals(List.of("model requested=requested-chat selected=selected-chat"),
                strict.toMetadata().get("route_mismatch_messages"));
    }

    @Test
    void warningsCanBlockCleanCiGate() {
        AgentServingPreflightResult result = AgentServingPreflightResult.from(
                request("demo-chat"),
                response("demo-chat", true, List.of(), List.of("mcp discovery unavailable")));

        AgentServingPreflightGate gate = result.gate(true, true);

        assertFalse(gate.ready());
        assertFalse(gate.blocksOnRouteMismatch());
        assertTrue(gate.blocksOnWarnings());
        assertEquals(List.of(AgentServingPreflightGate.REASON_WARNINGS_PRESENT), gate.blockingReasons());
        assertEquals(List.of("preflight warnings are present"), gate.blockingMessages());
        assertEquals(1, gate.toMetadata().get("warning_count"));
    }

    private static AgentServingPreflightRequest request(String model) {
        return AgentServingPreflightRequest.builder()
                .route(AgentServingRoute.of("chat", model, AgentServingFeatureProfile.CHAT_AGENT))
                .build();
    }

    private static Map<String, Object> response(
            String model,
            boolean ready,
            List<String> blockingMessages,
            List<String> warningMessages) {
        return Map.of(
                "status", ready ? "ready" : "blocked",
                "ready", ready,
                "surface", "chat",
                "model", model,
                "feature_profile", AgentServingFeatureProfile.CHAT_AGENT,
                "blocking_messages", blockingMessages,
                "warning_messages", warningMessages,
                "issues", List.of());
    }
}
