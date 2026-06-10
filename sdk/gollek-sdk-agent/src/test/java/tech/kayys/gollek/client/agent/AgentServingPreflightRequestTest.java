package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingPreflightRequestTest {

    @Test
    void acceptsTypedServingRouteDescriptor() {
        AgentServingRoute route = AgentServingRoute.of(
                "chat-completions",
                "demo-chat",
                "chatAgent");

        AgentServingPreflightRequest request = AgentServingPreflightRequest.builder()
                .route(route)
                .request(Map.of("messages", List.of(Map.of(
                        "role", "user",
                        "content", "hello"))))
                .build();

        assertEquals("demo-chat", request.modelId());
        assertEquals("chat", request.surface());
        assertEquals(route, request.route());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, request.featureProfile());
        assertTrue(request.requiredFeatures().contains("openai_chat_completions"));
        assertTrue(request.optionalFeatures().contains("system_prompt"));
    }

    @Test
    void normalizesRouteFieldsFromLegacyBuilderSetters() {
        AgentServingPreflightRequest request = AgentServingPreflightRequest.builder()
                .modelId(" demo-response ")
                .surface("response")
                .featureProfile("responsesAgent")
                .build();

        assertEquals("demo-response", request.modelId());
        assertEquals("responses", request.surface());
        assertEquals(AgentServingRoute.of(
                        "responses",
                        "demo-response",
                        AgentServingFeatureProfile.RESPONSES_AGENT),
                request.route());
    }
}
