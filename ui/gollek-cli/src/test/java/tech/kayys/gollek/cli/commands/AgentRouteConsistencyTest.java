package tech.kayys.gollek.cli.commands;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentRouteConsistencyTest {

    @Test
    void reportsConflictsWithoutMutatingNestedRoute() {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("model", "other-model");
        request.put("surface", "chat");
        request.put("feature_profile", "embedding_rag");
        AgentRouteInput input = input("demo-model", "responses", "chat_agent", request);

        AgentRouteConsistency.Result result = AgentRouteConsistency.validateAndNormalize(
                input,
                request,
                new AgentRouteConsistency.Overrides(false, false, false),
                new AgentRouteConsistency.Route("demo-model", "responses", "chat_agent"));

        assertFalse(result.ok());
        assertEquals(3, result.conflicts().size());
        assertEquals("model", result.conflicts().get(0).field());
        assertEquals("demo-model", result.conflicts().get(0).topLevel());
        assertEquals("other-model", result.conflicts().get(0).request());
        assertEquals("surface", result.conflictMetadata().get(1).get("field"));
        assertEquals("responses", result.conflictMetadata().get(1).get("top_level"));
        assertEquals("chat", result.conflictMetadata().get(1).get("request"));
        assertTrue(result.errorMessage().contains("model top-level=demo-model request=other-model"));
        assertTrue(result.errorMessage().contains("surface top-level=responses request=chat"));
        assertTrue(result.errorMessage().contains("feature_profile top-level=chat_agent request=embedding_rag"));
        assertEquals("other-model", request.get("model"));
        assertEquals("chat", request.get("surface"));
        assertEquals("embedding_rag", request.get("feature_profile"));
    }

    @Test
    void cliOverridesNormalizeNestedSurfaceAndCamelCaseProfile() {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("surface", "chat");
        request.put("featureProfile", "embedding_rag");
        AgentRouteInput input = input("demo-model", "chat", "embedding_rag", request);

        AgentRouteConsistency.Result result = AgentRouteConsistency.validateAndNormalize(
                input,
                request,
                new AgentRouteConsistency.Overrides(false, true, true),
                new AgentRouteConsistency.Route("demo-model", "responses", "chat_agent"));

        assertTrue(result.ok());
        assertEquals("responses", request.get("surface"));
        assertEquals("chat_agent", request.get("feature_profile"));
        assertFalse(request.containsKey("featureProfile"));
    }

    private static AgentRouteInput input(
            String model,
            String surface,
            String featureProfile,
            Map<String, Object> request) {
        return new AgentRouteInput(
                model,
                surface,
                featureProfile,
                request,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null);
    }
}
