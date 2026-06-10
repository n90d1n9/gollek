package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AgentReadinessMetadataTest {

    @Test
    void derivesStableMetadataFromCheckMaps() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put("capabilities", Map.of(
                "status", "ready",
                "blocking_messages", List.of(),
                "warning_messages", List.of()));
        checks.put("mcp_discovery", Map.of(
                "status", "skipped",
                "blocking_messages", List.of(),
                "warning_messages", List.of("MCP discovery skipped by caller"),
                "warning_codes", List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER")));
        checks.put("tool_validation", Map.of(
                "status", "blocked",
                "blocking_messages", List.of("tool definitions are not valid"),
                "warning_messages", List.of("schema portability warning"),
                "warning_codes", List.of("TOOL_SCHEMA_PORTABILITY_WARNING")));

        Map<String, Object> metadata = AgentReadinessMetadata.fromChecks(checks);

        assertEquals(List.of("capabilities", "mcp_discovery", "tool_validation"),
                metadata.get("checked_areas"));
        assertEquals(1L, metadata.get("ready_check_count"));
        assertEquals(1L, metadata.get("blocked_check_count"));
        assertEquals(1L, metadata.get("skipped_check_count"));
        assertEquals(1, metadata.get("blocking_issue_count"));
        assertEquals(2, metadata.get("warning_count"));
        assertEquals(List.of("tool definitions are not valid"), metadata.get("blocking_messages"));
        assertEquals(List.of(
                        "MCP discovery skipped by caller",
                        "schema portability warning"),
                metadata.get("warning_messages"));
        assertEquals(List.of("TOOL_DEFINITIONS_INVALID"), metadata.get("blocking_codes"));
        assertEquals(List.of(
                        "MCP_DISCOVERY_SKIPPED_BY_CALLER",
                        "TOOL_SCHEMA_PORTABILITY_WARNING"),
                metadata.get("warning_codes"));
        assertEquals(List.of(
                        "tool definitions are not valid",
                        "schema portability warning"),
                map(metadata.get("issues_by_area")).get("tool_validation"));
        assertEquals(List.of(
                        "TOOL_DEFINITIONS_INVALID",
                        "TOOL_SCHEMA_PORTABILITY_WARNING"),
                map(metadata.get("issue_codes_by_area")).get("tool_validation"));
        List<Map<String, Object>> hints = mapList(metadata.get("issue_hints"));
        assertEquals(3, hints.size());
        assertEquals("MCP_DISCOVERY_SKIPPED_BY_CALLER", hints.get(0).get("code"));
        assertEquals("TOOL_DEFINITIONS_INVALID", hints.get(1).get("code"));
        assertEquals("Fix the OpenAI-compatible tool schema before serving the route.",
                hints.get(1).get("remediation"));
        Map<String, List<Map<String, Object>>> hintsByArea = hintGroups(metadata.get("issue_hints_by_area"));
        assertEquals(1, hintsByArea.get("mcp_discovery").size());
        assertEquals(2, hintsByArea.get("tool_validation").size());
        assertEquals("TOOL_DEFINITIONS_INVALID",
                hintsByArea.get("tool_validation").get(0).get("code"));
        Map<String, List<Map<String, Object>>> hintsByCode = hintGroups(metadata.get("issue_hints_by_code"));
        assertEquals("tool_validation",
                hintsByCode.get("TOOL_DEFINITIONS_INVALID").get(0).get("area"));
        assertEquals("MCP discovery skipped by caller",
                hintsByCode.get("MCP_DISCOVERY_SKIPPED_BY_CALLER").get(0).get("message"));
        List<Map<String, Object>> remediationPlan = mapList(metadata.get("remediation_plan"));
        assertEquals(3, remediationPlan.size());
        assertEquals("MCP_DISCOVERY_SKIPPED_BY_CALLER", remediationPlan.get(0).get("code"));
        assertEquals(List.of("MCP discovery skipped by caller"), remediationPlan.get(0).get("messages"));
        assertEquals("TOOL_DEFINITIONS_INVALID", remediationPlan.get(1).get("code"));
        assertEquals("Fix the OpenAI-compatible tool schema before serving the route.",
                remediationPlan.get(1).get("remediation"));
        assertEquals("TOOL_SCHEMA_PORTABILITY_WARNING", remediationPlan.get(2).get("code"));
        assertEquals(List.of(remediationPlan.get(1)), metadata.get("blocking_remediation_plan"));
        assertEquals(List.of(remediationPlan.get(0), remediationPlan.get(2)),
                metadata.get("warning_remediation_plan"));
        assertEquals(remediationPlan.subList(1, 3),
                hintGroups(metadata.get("remediation_plan_by_area")).get("tool_validation"));
        assertEquals(List.of(remediationPlan.get(1)),
                hintGroups(metadata.get("remediation_plan_by_code")).get("TOOL_DEFINITIONS_INVALID"));
        assertEquals(List.of(remediationPlan.get(0)),
                AgentReadinessMetadata.remediationPlanByCode(checks).get("MCP_DISCOVERY_SKIPPED_BY_CALLER"));
    }

    @Test
    void canPrefixFlatMessagesForOperatorFacingCliReports() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put("preflight", Map.of(
                "status", "blocked",
                "blocking_messages", List.of("agent request is not valid"),
                "warning_messages", List.of("MCP discovery is not available")));

        Map<String, Object> metadata = AgentReadinessMetadata.fromChecks(checks, true);

        assertEquals(List.of("preflight: agent request is not valid"),
                metadata.get("blocking_messages"));
        assertEquals(List.of("preflight: MCP discovery is not available"),
                metadata.get("warning_messages"));
        assertEquals(List.of("AGENT_REQUEST_INVALID"), metadata.get("blocking_codes"));
        assertEquals(List.of("MCP_DISCOVERY_UNAVAILABLE"), metadata.get("warning_codes"));
        assertEquals(List.of(
                        "agent request is not valid",
                        "MCP discovery is not available"),
                map(metadata.get("issues_by_area")).get("preflight"));
        assertEquals(List.of(
                        "AGENT_REQUEST_INVALID",
                        "MCP_DISCOVERY_UNAVAILABLE"),
                map(metadata.get("issue_codes_by_area")).get("preflight"));
        List<Map<String, Object>> hints = mapList(metadata.get("issue_hints"));
        assertEquals(2, hints.size());
        assertEquals("AGENT_REQUEST_INVALID", hints.get(0).get("code"));
        assertEquals("Fix the request model, messages/input, tools, or RAG context before serving.",
                hints.get(0).get("remediation"));
        assertEquals(2, hintGroups(metadata.get("issue_hints_by_area")).get("preflight").size());
        assertEquals("preflight",
                hintGroups(metadata.get("issue_hints_by_code")).get("MCP_DISCOVERY_UNAVAILABLE").get(0).get("area"));
        assertEquals(2, mapList(metadata.get("remediation_plan")).size());
        assertEquals("AGENT_REQUEST_INVALID",
                mapList(metadata.get("remediation_plan")).get(0).get("code"));
        assertEquals(List.of(mapList(metadata.get("remediation_plan")).get(0)),
                metadata.get("blocking_remediation_plan"));
        assertEquals(List.of(mapList(metadata.get("remediation_plan")).get(1)),
                metadata.get("warning_remediation_plan"));
        assertEquals("preflight",
                hintGroups(metadata.get("remediation_plan_by_code")).get("AGENT_REQUEST_INVALID").get(0).get("area"));
    }

    @Test
    void groupsRemediationPlanFromSuppliedIssueHints() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put("tool_validation", Map.of(
                "status", "blocked",
                "blocking_messages", List.of("tools[0] has no function schema", "tools[1] has no parameters"),
                "warning_messages", List.of(),
                "issue_hints", List.of(
                        Map.of(
                                "area", "tool_validation",
                                "severity", "blocking",
                                "message", "tools[0] has no function schema",
                                "code", "TOOL_DEFINITIONS_INVALID",
                                "summary", "Tool definitions failed service validation.",
                                "remediation", "Regenerate the tool contract."),
                        Map.of(
                                "area", "tool_validation",
                                "severity", "blocking",
                                "message", "tools[1] has no parameters",
                                "code", "tool-definitions-invalid",
                                "summary", "Tool definitions failed service validation.",
                                "remediation", "Regenerate the tool contract."))));

        Map<String, Object> metadata = AgentReadinessMetadata.fromChecks(checks);

        List<Map<String, Object>> remediationPlan = mapList(metadata.get("remediation_plan"));
        assertEquals(1, remediationPlan.size());
        assertEquals("TOOL_DEFINITIONS_INVALID", remediationPlan.get(0).get("code"));
        assertEquals(List.of("tools[0] has no function schema", "tools[1] has no parameters"),
                remediationPlan.get(0).get("messages"));
        assertEquals(remediationPlan, metadata.get("blocking_remediation_plan"));
        assertEquals(List.of(), metadata.get("warning_remediation_plan"));
        assertEquals(remediationPlan,
                hintGroups(metadata.get("remediation_plan_by_area")).get("tool_validation"));
        assertEquals(remediationPlan,
                hintGroups(metadata.get("remediation_plan_by_code")).get("TOOL_DEFINITIONS_INVALID"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, List<String>> map(Object value) {
        return (Map<String, List<String>>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> mapList(Object value) {
        return (List<Map<String, Object>>) value;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, List<Map<String, Object>>> hintGroups(Object value) {
        return (Map<String, List<Map<String, Object>>>) value;
    }
}
