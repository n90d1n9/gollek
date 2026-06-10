package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentReadinessIssueCodesTest {

    @Test
    void describesKnownCodesWithStableCatalogMetadata() {
        AgentReadinessIssueCodes.CatalogEntry entry =
                AgentReadinessIssueCodes.describe("model-route-unavailable");

        assertEquals("MODEL_ROUTE_UNAVAILABLE", entry.code());
        assertEquals(AgentServingReadinessReport.AREA_MODEL_ROUTE, entry.area());
        assertEquals("blocking", entry.defaultSeverity());
        assertTrue(entry.blockingByDefault());
        assertFalse(entry.warningByDefault());
        assertTrue(entry.summary().contains("model"));
        assertTrue(entry.remediation().contains("model"));
        assertEquals(entry, AgentReadinessIssueCodes.find("MODEL_ROUTE_UNAVAILABLE").orElseThrow());
    }

    @Test
    void resolvesUnsupportedFeatureProfileToStableCode() {
        assertEquals("FEATURE_PROFILE_UNSUPPORTED",
                AgentReadinessIssueCodes.resolve(
                        null,
                        AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION,
                        "blocking",
                        "feature profile is not supported: wayang_planner"));
        assertTrue(AgentReadinessIssueCodes.describe("FEATURE_PROFILE_UNSUPPORTED")
                .remediation()
                .contains("serving profile"));
    }

    @Test
    void groupsCatalogEntriesByReadinessArea() {
        List<AgentReadinessIssueCodes.CatalogEntry> toolCodes =
                AgentReadinessIssueCodes.catalogForArea("tool-validation");

        assertTrue(toolCodes.stream().anyMatch(entry -> entry.code().equals("TOOL_DEFINITIONS_INVALID")));
        assertTrue(toolCodes.stream().anyMatch(entry -> entry.code().equals("TOOL_SCHEMA_PORTABILITY_WARNING")));
        assertTrue(toolCodes.stream().allMatch(entry ->
                entry.area().equals(AgentServingReadinessReport.AREA_TOOL_VALIDATION)));
    }

    @Test
    void describesDynamicToolSchemaWarningCodes() {
        AgentReadinessIssueCodes.CatalogEntry entry =
                AgentReadinessIssueCodes.describe("tool_schema_feature_may_be_ignored");

        assertEquals("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED", entry.code());
        assertEquals(AgentServingReadinessReport.AREA_TOOL_VALIDATION, entry.area());
        assertEquals("warning", entry.defaultSeverity());
        assertTrue(entry.warningByDefault());
        assertTrue(entry.summary().contains("Tool schema"));
    }

    @Test
    void mapsCatalogEntriesForDashboards() {
        Map<String, Object> mapped = AgentReadinessIssueCodes.describe("agent_request_invalid").toMap();

        assertEquals(AgentReadinessIssueCodes.catalog().size(), AgentReadinessIssueCodes.catalogByCode().size());
        assertEquals("AGENT_REQUEST_INVALID", mapped.get("code"));
        assertEquals(AgentServingReadinessReport.AREA_REQUEST_VALIDATION, mapped.get("area"));
        assertEquals("blocking", mapped.get("default_severity"));
        assertTrue(AgentReadinessIssueCodes.catalogMaps().stream()
                .anyMatch(entry -> "AGENT_REQUEST_INVALID".equals(entry.get("code"))));
    }

    @Test
    void resolvesPlainSchemaPortabilityWarningToStableCode() {
        assertEquals("TOOL_SCHEMA_PORTABILITY_WARNING",
                AgentReadinessIssueCodes.resolve(
                        null,
                        AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                        "warning",
                        "schema portability warning"));
    }
}
