package tech.kayys.gollek.server.api.v1;

import tech.kayys.gollek.client.agent.AgentReadinessIssueCodes;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class AgentReadinessIssueCatalogMapper {

    private AgentReadinessIssueCatalogMapper() {
    }

    static Map<String, Object> catalog() {
        Map<String, Object> payload = new LinkedHashMap<>();
        List<Map<String, Object>> items = entries();
        payload.put("object", "gollek.agent_readiness_issue_catalog");
        payload.put("version", "v1");
        payload.put("service_role", "inference_serving_engine");
        payload.put("boundary", boundary());
        payload.put("count", items.size());
        payload.put("items", items);
        payload.put("by_code", byCode());
        payload.put("by_area", byArea());
        return payload;
    }

    static List<Map<String, Object>> entries() {
        return AgentReadinessIssueCodes.catalogMaps();
    }

    static Map<String, Object> byCode() {
        Map<String, Object> out = new LinkedHashMap<>();
        for (AgentReadinessIssueCodes.CatalogEntry entry : AgentReadinessIssueCodes.catalog()) {
            out.put(entry.code(), entry.toMap());
        }
        return out;
    }

    static Map<String, Object> byArea() {
        Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
        for (AgentReadinessIssueCodes.CatalogEntry entry : AgentReadinessIssueCodes.catalog()) {
            grouped.computeIfAbsent(entry.area(), ignored -> new ArrayList<>()).add(entry.toMap());
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Map<String, Object>>> entry : grouped.entrySet()) {
            out.put(entry.getKey(), List.copyOf(entry.getValue()));
        }
        return out;
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", true);
        boundary.put("model_invoked", false);
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        boundary.put("tool_authorization", false);
        return boundary;
    }
}
