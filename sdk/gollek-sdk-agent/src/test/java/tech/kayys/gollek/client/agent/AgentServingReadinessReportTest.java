package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static java.util.Map.entry;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingReadinessReportTest {

    @Test
    void readyWhenAllServingBoundaryViewsPass() {
        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .capabilities(capabilitiesView())
                .contract(contractView())
                .modelRoute(modelRouteView())
                .mcpDiscovery(mcpDiscoveryView(), true)
                .toolValidation(toolValidationView(), true)
                .requestValidation(requestValidationView(), true)
                .build();

        assertTrue(report.ready(), report.issues().toString());
        assertFalse(report.hasWarnings());
        assertTrue(report.issues().isEmpty());
        assertTrue(report.issuesByArea().isEmpty());
        assertTrue(report.issueCodesByArea().isEmpty());
        assertTrue(report.blockingCodesByArea().isEmpty());
        assertTrue(report.warningCodesByArea().isEmpty());
        assertTrue(report.issueHints().isEmpty());
        assertTrue(report.issueHintsByArea().isEmpty());
        assertTrue(report.remediationPlan().isEmpty());
        assertTrue(report.remediationPlanByArea().isEmpty());
        assertTrue(report.remediationPlanByCode().isEmpty());
        assertEquals(List.of(), report.toMetadata().get("remediation_plan"));
        assertEquals(List.of(), report.toMetadata().get("blocking_remediation_plan"));
        assertEquals(List.of(), report.toMetadata().get("warning_remediation_plan"));
        assertEquals(Map.of(), report.toMetadata().get("remediation_plan_by_area"));
        assertEquals(Map.of(), report.toMetadata().get("remediation_plan_by_code"));
        assertEquals("ready", report.status());
        assertEquals("Gollek serving preflight ready", report.summary());
        assertEquals(true, report.toMetadata().get("ready"));
        assertEquals("ready", report.toMetadata().get("status"));
        assertEquals(AgentServingReadinessReport.OBJECT, report.toMetadata().get("object"));
        assertEquals("chat", report.surface().orElseThrow());
        assertEquals("demo", report.model().orElseThrow());
        assertEquals(AgentServingRoute.of("chat", "demo"), report.route());
        assertTrue(report.route().matchesSurface("chat_completions"));
        assertEquals("chat", report.toMetadata().get("surface"));
        assertEquals("demo", report.toMetadata().get("model"));
        assertEquals(6L, report.toMetadata().get("ready_check_count"));
        assertEquals(0L, report.toMetadata().get("blocked_check_count"));
        assertEquals(0L, report.toMetadata().get("skipped_check_count"));
        assertEquals(List.of(), report.toMetadata().get("blocking_codes"));
        assertEquals(List.of(), report.toMetadata().get("warning_codes"));
        assertEquals(List.of(), report.toMetadata().get("issue_hints"));
        assertEquals(List.of(
                        AgentServingReadinessReport.AREA_CAPABILITIES,
                        AgentServingReadinessReport.AREA_CONTRACT,
                        AgentServingReadinessReport.AREA_MODEL_ROUTE,
                        AgentServingReadinessReport.AREA_MCP_DISCOVERY,
                        AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                        AgentServingReadinessReport.AREA_REQUEST_VALIDATION),
                report.checkedAreas());
        assertEquals(report.checkedAreas(), report.toMetadata().get("checked_areas"));

        Map<String, Object> readinessReport = report.toReport();
        assertEquals(AgentServingReadinessReport.OBJECT, readinessReport.get("object"));
        assertEquals(false, map(readinessReport.get("boundary")).get("model_invoked"));
        assertEquals(true, map(map(readinessReport.get("checks"))
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)).get("ready"));
        assertEquals(List.of(), map(map(readinessReport.get("checks"))
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)).get("remediation_plan"));
        assertEquals(List.of(), map(map(readinessReport.get("checks"))
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)).get("blocking_remediation_plan"));
        assertEquals(List.of(), map(map(readinessReport.get("checks"))
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)).get("warning_remediation_plan"));
        assertEquals(Map.of(), map(map(readinessReport.get("checks"))
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)).get("remediation_plan_by_code"));
        assertTrue(report.hasCheck(AgentServingReadinessReport.AREA_CAPABILITIES));
        assertEquals(
                report.check(AgentServingReadinessReport.AREA_CAPABILITIES),
                report.capabilitiesCheck());
        assertEquals(report.check(AgentServingReadinessReport.AREA_CONTRACT), report.contractCheck());
        assertEquals(report.check(AgentServingReadinessReport.AREA_MODEL_ROUTE), report.modelRouteCheck());
        assertEquals(report.check(AgentServingReadinessReport.AREA_MCP_DISCOVERY), report.mcpDiscoveryCheck());
        assertEquals(report.check(AgentServingReadinessReport.AREA_TOOL_VALIDATION), report.toolValidationCheck());
        assertEquals(
                report.check(AgentServingReadinessReport.AREA_REQUEST_VALIDATION),
                report.requestValidationCheck());
        assertTrue(report.check(AgentServingReadinessReport.AREA_MODEL_ROUTE).ready());
        assertTrue(report.check(AgentServingReadinessReport.AREA_MODEL_ROUTE).requested());
        assertFalse(report.check(AgentServingReadinessReport.AREA_MODEL_ROUTE).hasWarnings());

        AgentServingReadinessReport.Check requestCheck =
                report.requestValidationCheck();
        assertEquals("chat", requestCheck.surface());
        assertEquals("demo", requestCheck.model());
        assertTrue(requestCheck.boundaryDetails().validationOnly());
        assertFalse(requestCheck.boundaryDetails().modelInvoked());
        assertTrue(requestCheck.hasRequestDetails());
        assertFalse(requestCheck.hasEmbeddingDetails());
        AgentServingReadinessReport.RequestDetails requestDetails = requestCheck.requestDetails();
        assertEquals(false, requestDetails.streaming());
        assertEquals(2, requestDetails.messageCount());
        assertEquals(1, requestDetails.toolCount());
        assertEquals(List.of("model", "messages", "tools", "rag_context"), requestDetails.parameterKeys());
        assertEquals(true, requestDetails.rag().injected());
        assertEquals(1, requestDetails.rag().items());
        assertEquals("rag_context", requestDetails.rag().alias());
        assertEquals(true, requestDetails.toolContract().valid());
    }

    @Test
    void featureNegotiationBlocksUnsupportedRequiredFeaturesAndWarnsOnOptionalFeatures() {
        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .featureNegotiation(
                        AgentFeatureNegotiation.from(Map.of(
                                "contract_version", "v1",
                                "supported_contract_versions", List.of("v1"),
                                "compatibility", List.of("rag_context"))),
                        "v2",
                        List.of("rag_context", "wayang_planner"),
                        List.of("wayang_designer"))
                .build();

        assertFalse(report.ready());
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE, report.featureProfile().orElseThrow());
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE, report.toMetadata().get("feature_profile"));
        assertTrue(report.hasWarnings());
        assertTrue(report.hasBlockingCode("CONTRACT_VERSION_UNSUPPORTED"));
        assertTrue(report.hasBlockingCode("REQUIRED_AGENT_FEATURE_UNSUPPORTED"));
        assertTrue(report.hasWarningCode("OPTIONAL_AGENT_FEATURE_UNSUPPORTED"));
        AgentServingReadinessReport.Check check = report.featureNegotiationCheck();
        assertEquals("blocked", check.status());
        assertEquals("v2", check.details().get("required_contract_version"));
        assertEquals(List.of("wayang_planner"), check.details().get("unsupported_required_features"));
        assertEquals(List.of("wayang_designer"), check.details().get("unsupported_optional_features"));
        assertTrue(mapList(check.toMap().get("blocking_remediation_plan")).stream()
                .anyMatch(remediation -> "REQUIRED_AGENT_FEATURE_UNSUPPORTED".equals(remediation.get("code"))));
        assertTrue(mapList(check.toMap().get("warning_remediation_plan")).stream()
                .anyMatch(remediation -> "OPTIONAL_AGENT_FEATURE_UNSUPPORTED".equals(remediation.get("code"))));
    }

    @Test
    void featureNegotiationBlocksUnsupportedProfiles() {
        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .featureNegotiation(
                        AgentFeatureNegotiation.from(AgentServingFeatureCatalog.featureNegotiation()),
                        "wayang_planner",
                        "v1",
                        List.of(),
                        List.of())
                .build();

        assertFalse(report.ready());
        assertEquals("wayang_planner", report.featureProfile().orElseThrow());
        assertEquals("wayang_planner", report.featureProfileOrDefault());
        assertEquals("wayang_planner", report.toReport().get("feature_profile"));
        assertTrue(report.hasBlockingCode("FEATURE_PROFILE_UNSUPPORTED"));
        AgentServingReadinessReport.Check check = report.featureNegotiationCheck();
        assertEquals("wayang_planner", check.details().get("feature_profile"));
        assertEquals(false, check.details().get("feature_profile_supported"));
        assertTrue(((List<?>) check.details().get("supported_feature_profiles"))
                .contains(AgentServingFeatureProfile.EMBEDDING_RAG));
    }

    @Test
    void embeddingSurfaceUsesEmbeddingRouteChecks() {
        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .capabilities(capabilitiesView())
                .contract(contractView())
                .modelRoute("embeddings", embeddingRouteView())
                .requestValidation(AgentValidationView.from(Map.of(
                        "surface", "embeddings",
                        "valid", true,
                        "model_invoked", false,
                        "boundary", Map.of(
                                "validation_only", true,
                                "tool_execution", false,
                                "retrieval_execution", false),
                        "normalized", Map.of(
                                "model", "demo-embed",
                                "input_count", 1,
                                "input_lengths", List.of(19),
                                "requested_dimensions", 384,
                                "encoding_format", "float",
                                "parameter_keys", List.of("dimensions", "encoding_format", "metadata"),
                                "metadata", Map.of("tenant", "agent-project"),
                                "rag", Map.of(
                                        "embedding_generation", true,
                                        "retrieval_execution", false,
                                        "retrieval_policy_owned_by", "agent_orchestrator",
                                        "vector_store_owned_by", "agent_orchestrator")))), true)
                .build();

        assertTrue(report.ready(), report.issues().toString());
        assertEquals("embeddings", report.surface().orElseThrow());
        assertEquals("demo-embed", report.model().orElseThrow());
        assertEquals(AgentServingRoute.of("embeddings", "demo-embed"), report.route());
        assertEquals("embeddings", report.toReport().get("surface"));
        assertEquals("demo-embed", report.toReport().get("model"));
        assertTrue(report.check(AgentServingReadinessReport.AREA_MODEL_ROUTE).ready());
        assertEquals(4L, report.toMetadata().get("ready_check_count"));
        AgentServingReadinessReport.Check typedCheck =
                report.check(AgentServingReadinessReport.AREA_REQUEST_VALIDATION);
        assertEquals("embeddings", typedCheck.surface());
        assertEquals("demo-embed", typedCheck.model());
        assertTrue(typedCheck.hasEmbeddingDetails());
        assertFalse(typedCheck.hasRequestDetails());
        assertTrue(typedCheck.boundaryDetails().validationOnly());
        AgentServingReadinessReport.EmbeddingDetails typedEmbedding = typedCheck.embeddingDetails();
        assertEquals(1, typedEmbedding.inputCount());
        assertEquals(List.of(19), typedEmbedding.inputLengths());
        assertEquals(384, typedEmbedding.requestedDimensions());
        assertEquals("float", typedEmbedding.encodingFormat());
        assertEquals(List.of("dimensions", "encoding_format", "metadata"), typedEmbedding.parameterKeys());
        assertEquals(List.of("tenant"), typedEmbedding.metadataKeys());
        assertEquals(true, typedEmbedding.rag().embeddingGeneration());
        assertEquals(false, typedEmbedding.rag().retrievalExecution());
        assertEquals("agent_orchestrator", typedEmbedding.rag().retrievalPolicyOwnedBy());
        assertEquals("agent_orchestrator", typedEmbedding.rag().vectorStoreOwnedBy());
        assertTrue(typedEmbedding.storageOwnedByOrchestrator());
        Map<String, Object> requestCheck = map(map(report.toReport().get("checks"))
                .get(AgentServingReadinessReport.AREA_REQUEST_VALIDATION));
        Map<String, Object> details = map(requestCheck.get("details"));
        Map<String, Object> embedding = map(details.get("embedding"));
        Map<String, Object> rag = map(embedding.get("rag"));
        assertEquals("embeddings", details.get("surface"));
        assertEquals("demo-embed", details.get("model"));
        assertEquals(1, embedding.get("input_count"));
        assertEquals(List.of(19), embedding.get("input_lengths"));
        assertEquals(384, embedding.get("requested_dimensions"));
        assertEquals("float", embedding.get("encoding_format"));
        assertEquals(List.of("tenant"), embedding.get("metadata_keys"));
        assertEquals(true, rag.get("embedding_generation"));
        assertEquals(false, rag.get("retrieval_execution"));
        assertEquals("agent_orchestrator", rag.get("retrieval_policy_owned_by"));
        assertEquals("agent_orchestrator", rag.get("vector_store_owned_by"));
        assertEquals(true, rag.get("storage_owned_by_orchestrator"));
        assertFalse(report.hasIssue(
                AgentServingReadinessReport.AREA_MODEL_ROUTE,
                "model does not advertise completion serving"));
    }

    @Test
    void collectsBlockingAndWarningIssuesByArea() {
        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .capabilities(null)
                .contract(null)
                .modelRoute(AgentModelCapabilitiesView.from(Map.of(
                        "available", false,
                        "api_contract", Map.of(),
                        "openai_compatibility", Map.of(),
                        "inference", Map.of(),
                        "tooling", Map.of("tool_execution", true),
                        "rag", Map.of(
                                "retrieval_policy", true,
                                "vector_store_ownership", true))))
                .mcpDiscovery(AgentMcpDiscoveryView.from(Map.of(
                        "available", true,
                        "boundary", Map.of(
                                "role", "tool_runtime",
                                "tool_execution", true),
                        "tools", List.of(Map.of(
                                "type", "function",
                                "execution", true,
                                "function", Map.of("name", "lookup"))))), true)
                .toolValidation(AgentToolValidationView.from(Map.of(
                        "valid", false,
                        "model_invoked", true,
                        "boundary", Map.of(
                                "validation_only", false,
                                "tool_execution", true,
                                "tool_authorization", true),
                        "warnings", List.of(Map.of(
                                "code", "schema_feature_may_be_ignored",
                                "path", "tools[0].parameters.anyOf",
                                "message", "anyOf may be ignored")))), true)
                .requestValidation(AgentValidationView.from(Map.of(
                        "valid", false,
                        "model_invoked", true,
                        "boundary", Map.of(
                                "validation_only", false,
                                "tool_execution", true,
                                "retrieval_execution", true),
                        "normalized", Map.of(
                                "tool_contract", Map.of("valid", false)))), true)
                .build();

        assertFalse(report.ready());
        assertTrue(report.hasWarnings());
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_CAPABILITIES,
                "capabilities response is missing"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_CONTRACT,
                "serving contract response is missing"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_MODEL_ROUTE,
                "model is not available through Gollek serving"));
        assertTrue(report.hasIssueCode("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.hasBlockingCode("model-route-unavailable"));
        assertFalse(report.hasWarningCode("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.modelRouteCheck().hasBlockingCode("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_MCP_DISCOVERY,
                "MCP tool must not be executable from Gollek: lookup"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                "tool validation must not authorize tools"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                "TOOL_AUTHORIZATION_BOUNDARY_DRIFT",
                "tool validation must not authorize tools"));
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_REQUEST_VALIDATION,
                "request validation must not enable retrieval execution"));
        assertTrue(report.hasIssueCode("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertTrue(report.hasWarningCode("tool schema feature may be ignored"));
        assertTrue(report.toolValidationCheck().hasIssueCode("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertTrue(report.toolValidationCheck().hasWarningCode("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertTrue(report.warningMessages().stream()
                .anyMatch(message -> message.contains("schema_feature_may_be_ignored")));
        assertTrue(report.warningCodes().contains("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertTrue(report.issuesByArea().containsKey(AgentServingReadinessReport.AREA_MODEL_ROUTE));
        assertTrue(report.blockingCodesByArea()
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)
                .contains("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.issueCodesByArea()
                .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                .contains("TOOL_AUTHORIZATION_BOUNDARY_DRIFT"));
        assertTrue(report.warningCodesByArea()
                .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                .contains("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        AgentServingReadinessReport.IssueHint modelHint = report.blockingIssueHints().stream()
                .filter(hint -> hint.code().equals("MODEL_ROUTE_UNAVAILABLE"))
                .findFirst()
                .orElseThrow();
        assertEquals(AgentServingReadinessReport.AREA_MODEL_ROUTE, modelHint.area());
        assertEquals(AgentServingReadinessReport.Severity.BLOCKING, modelHint.severity());
        assertEquals("blocking", modelHint.defaultSeverity());
        assertTrue(modelHint.summary().contains("model"));
        assertTrue(modelHint.remediation().contains("model"));
        assertEquals("MODEL_ROUTE_UNAVAILABLE", modelHint.toMap().get("code"));
        assertEquals(modelHint, report.issueHint("model-route-unavailable").orElseThrow());
        assertEquals(modelHint, report.issueHintsForCode("MODEL_ROUTE_UNAVAILABLE").get(0));
        assertEquals(modelHint, report.issueHintsByCode().get("MODEL_ROUTE_UNAVAILABLE").get(0));
        assertTrue(report.remediationForCode("MODEL_ROUTE_UNAVAILABLE").orElseThrow().contains("model"));
        AgentReadinessRemediation modelRemediation = report.blockingRemediationPlan().stream()
                .filter(remediation -> remediation.code().equals("MODEL_ROUTE_UNAVAILABLE"))
                .findFirst()
                .orElseThrow();
        assertEquals(AgentServingReadinessReport.AREA_MODEL_ROUTE, modelRemediation.area());
        assertTrue(modelRemediation.blocking());
        assertTrue(modelRemediation.summary().contains("model"));
        assertTrue(modelRemediation.remediation().contains("model"));
        assertTrue(modelRemediation.messages().contains("model is not available through Gollek serving"));
        assertEquals("MODEL_ROUTE_UNAVAILABLE", modelRemediation.toMap().get("code"));
        assertTrue(report.remediationPlanByArea()
                .get(AgentServingReadinessReport.AREA_MODEL_ROUTE)
                .contains(modelRemediation));
        assertEquals(List.of(modelRemediation), report.remediationPlanForCode("model-route-unavailable"));
        assertEquals(List.of(modelRemediation), report.remediationPlanByCode().get("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(mapList(report.toMetadata().get("remediation_plan")).stream()
                .anyMatch(remediation -> "MODEL_ROUTE_UNAVAILABLE".equals(remediation.get("code"))));
        assertEquals(mapList(report.toMetadata().get("remediation_plan")).stream()
                        .filter(remediation -> AgentServingReadinessReport.AREA_MODEL_ROUTE
                                .equals(remediation.get("area")))
                        .toList(),
                hintGroups(report.toMetadata().get("remediation_plan_by_area"))
                        .get(AgentServingReadinessReport.AREA_MODEL_ROUTE));
        assertEquals(List.of(modelRemediation.toMap()),
                hintGroups(report.toMetadata().get("remediation_plan_by_code"))
                        .get("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.warningRemediationPlan().stream()
                .allMatch(AgentReadinessRemediation::warning));
        assertEquals(List.of(), report.issueHintsForCode("missing readiness issue"));
        assertTrue(report.issueHint("missing readiness issue").isEmpty());
        assertEquals("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED",
                report.toolValidationCheck()
                        .issueHint("tool-schema-feature-may-be-ignored")
                        .orElseThrow()
                        .code());
        assertTrue(report.toolValidationCheck()
                .remediationForCode("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED")
                .orElseThrow()
                .contains("tool schema"));
        assertEquals(1, report.toolValidationCheck()
                .issueHintsByCode()
                .get("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED")
                .size());
        assertTrue(mapList(report.toMetadata().get("issue_hints")).stream()
                .anyMatch(hint -> "MODEL_ROUTE_UNAVAILABLE".equals(hint.get("code"))
                        && String.valueOf(hint.get("remediation")).contains("model")));
        assertTrue(mapList(map(report.toolValidationCheck().toMap()).get("issue_hints")).stream()
                .anyMatch(hint -> "TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED".equals(hint.get("code"))
                        && String.valueOf(hint.get("remediation")).contains("tool schema")));
        assertTrue(mapList(map(report.toolValidationCheck().toMap()).get("remediation_plan")).stream()
                .anyMatch(remediation -> "TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED".equals(remediation.get("code"))
                        && String.valueOf(remediation.get("remediation")).contains("tool schema")));
        assertTrue(mapList(map(report.toolValidationCheck().toMap()).get("blocking_remediation_plan")).stream()
                .anyMatch(remediation -> "TOOL_AUTHORIZATION_BOUNDARY_DRIFT".equals(remediation.get("code"))));
        assertTrue(mapList(map(report.toolValidationCheck().toMap()).get("warning_remediation_plan")).stream()
                .anyMatch(remediation -> "TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED".equals(remediation.get("code"))));
        assertTrue(hintGroups(map(report.toolValidationCheck().toMap()).get("remediation_plan_by_code"))
                .containsKey("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertTrue(report.issueHintsByArea()
                .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                .stream()
                .anyMatch(hint -> hint.code().equals("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED")
                        && hint.warningIssue()));
        assertTrue(report.toolValidationCheck().warningIssueHints().stream()
                .anyMatch(hint -> hint.code().equals("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED")
                        && hint.remediation().contains("tool schema")));
        assertEquals("blocked", report.status());
        assertEquals(false, report.toMetadata().get("ready"));
        assertEquals("blocked", report.toMetadata().get("status"));
        assertEquals(report.blockingIssues().size(), report.toMetadata().get("blocking_issue_count"));
        assertEquals(report.warnings().size(), report.toMetadata().get("warning_count"));
        assertTrue(((List<String>) report.toMetadata().get("blocking_codes")).contains("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(((List<String>) report.toMetadata().get("warning_codes"))
                .contains("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertEquals(0L, report.toMetadata().get("ready_check_count"));
        assertEquals(6L, report.toMetadata().get("blocked_check_count"));
        assertTrue(report.summary().contains("Gollek serving preflight blocked"));
    }

    @Test
    void requireReadyReturnsReportOrThrowsIntegrationException() throws Exception {
        AgentServingReadinessReport ready = AgentServingReadinessReport.builder()
                .capabilities(capabilitiesView())
                .contract(contractView())
                .modelRoute(modelRouteView())
                .build();

        assertEquals(ready, ready.requireReady());

        AgentServingReadinessReport blocked = AgentServingReadinessReport.builder()
                .capabilities(null)
                .build();

        AgentIntegrationException exception = assertThrows(
                AgentIntegrationException.class,
                () -> blocked.requireReady("Preflight blocked"));
        assertEquals("SDK_ERR_AGENT_PREFLIGHT", exception.getErrorCode());
        assertTrue(exception.getMessage().contains("Preflight blocked"));
        assertTrue(exception.getMessage().contains("capabilities response is missing"));
    }

    @Test
    void mapsServerPreflightResponseIntoReadinessReport() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "blocked",
                "ready", false,
                "feature_profile", AgentServingFeatureProfile.CHAT_AGENT,
                "surface", "chat",
                "model", "demo-chat",
                "issues", List.of(
                        Map.of(
                                "area", AgentServingReadinessReport.AREA_MODEL_ROUTE,
                                "severity", "blocking",
                                "code", "MODEL_ROUTE_UNAVAILABLE",
                                "message", "model is not available through Gollek serving"),
                        Map.of(
                                "area", AgentServingReadinessReport.AREA_TOOL_VALIDATION,
                                "severity", "warning",
                                "code", "TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED",
                                "message", "tool schema warning (schema_feature_may_be_ignored)"))));

        assertFalse(report.ready());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, report.featureProfile().orElseThrow());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, report.toMetadata().get("feature_profile"));
        assertEquals("chat", report.surface().orElseThrow());
        assertEquals("demo-chat", report.model().orElseThrow());
        assertEquals(AgentServingRoute.of("chat", "demo-chat", AgentServingFeatureProfile.CHAT_AGENT),
                report.route());
        assertEquals("chat", report.toMetadata().get("surface"));
        assertEquals("demo-chat", report.toMetadata().get("model"));
        assertTrue(report.hasWarnings());
        assertTrue(report.hasIssueCode("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.hasBlockingCode("MODEL_ROUTE_UNAVAILABLE"));
        assertTrue(report.hasWarningCode("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"));
        assertEquals("MODEL_ROUTE_UNAVAILABLE", report.issues().get(0).code());
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_MODEL_ROUTE,
                "model is not available through Gollek serving"));
        assertEquals(List.of("tool schema warning (schema_feature_may_be_ignored)"), report.warningMessages());
        assertEquals(List.of("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"), report.warningCodes());
        assertEquals(List.of("MODEL_ROUTE_UNAVAILABLE"),
                report.blockingCodesByArea().get(AgentServingReadinessReport.AREA_MODEL_ROUTE));
        assertEquals(List.of("TOOL_SCHEMA_FEATURE_MAY_BE_IGNORED"),
                report.warningCodesByArea().get(AgentServingReadinessReport.AREA_TOOL_VALIDATION));
        assertEquals("Load the model or choose a model route that Gollek can serve.",
                report.blockingIssueHints().get(0).remediation());
        assertEquals("Tool schema portability warning.",
                report.warningIssueHints().get(0).summary());
        assertEquals("blocked", report.status());
    }

    @Test
    void mapsServerPreflightIssuesByAreaFallback() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "status", "blocked",
                "ready", false,
                "issues_by_area", Map.of(
                        AgentServingReadinessReport.AREA_MCP_DISCOVERY, List.of("MCP discovery is not available")),
                "warning_messages", List.of()));

        assertFalse(report.ready());
        assertTrue(report.hasIssue(
                AgentServingReadinessReport.AREA_MCP_DISCOVERY,
                "MCP discovery is not available"));
    }

    @Test
    void mapsFlatServerPreflightMessagesIntoPreflightIssues() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "status", "blocked",
                "ready", false,
                "blocking_messages", List.of("agent request is not valid"),
                "blocking_codes", List.of("AGENT_REQUEST_INVALID"),
                "warning_messages", List.of("mcp discovery unavailable"),
                "warning_codes", List.of("MCP_DISCOVERY_UNAVAILABLE")));

        assertFalse(report.ready());
        assertTrue(report.hasIssue("preflight", "agent request is not valid"));
        assertTrue(report.hasBlockingCode("AGENT_REQUEST_INVALID"));
        assertTrue(report.hasWarningCode("MCP_DISCOVERY_UNAVAILABLE"));
        assertEquals(List.of("agent request is not valid"), report.blockingMessages());
        assertEquals(List.of("mcp discovery unavailable"), report.warningMessages());
        assertEquals(List.of("AGENT_REQUEST_INVALID"), report.blockingCodes());
        assertEquals(List.of("MCP_DISCOVERY_UNAVAILABLE"), report.warningCodes());
    }

    @Test
    void preservesExplicitCheckedAreasFromCompactServerPreflightResponse() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "status", "blocked",
                "ready", false,
                "checked_areas", List.of(
                        AgentServingReadinessReport.AREA_CAPABILITIES,
                        AgentServingReadinessReport.AREA_MODEL_ROUTE,
                        AgentServingReadinessReport.AREA_REQUEST_VALIDATION),
                "issues_by_area", Map.of(
                        AgentServingReadinessReport.AREA_MODEL_ROUTE,
                        List.of("model is not available through Gollek serving"))));

        assertFalse(report.ready());
        assertEquals(List.of(
                        AgentServingReadinessReport.AREA_CAPABILITIES,
                        AgentServingReadinessReport.AREA_MODEL_ROUTE,
                        AgentServingReadinessReport.AREA_REQUEST_VALIDATION),
                report.checkedAreas());
        assertEquals(2L, report.toMetadata().get("ready_check_count"));
        assertEquals(1L, report.toMetadata().get("blocked_check_count"));
        assertEquals(report.checkedAreas(), report.toMetadata().get("checked_areas"));
    }

    @Test
    void readsNestedReadinessReportChecksWhenTopLevelChecksAreRequestOptions() {
        Map<String, Object> nestedChecks = new LinkedHashMap<>();
        nestedChecks.put(AgentServingReadinessReport.AREA_REQUEST_VALIDATION, Map.of(
                "status", "ready",
                "ready", true,
                "requested", true,
                "blocking_messages", List.of(),
                "warning_messages", List.of(),
                "details", Map.of("surface", "chat", "model", "demo")));
        nestedChecks.put(AgentServingReadinessReport.AREA_MCP_DISCOVERY, Map.of(
                "status", "skipped",
                "ready", true,
                "requested", false,
                "blocking_messages", List.of(),
                "warning_messages", List.of("MCP discovery skipped by caller"),
                "warning_codes", List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER")));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "ready",
                "ready", true,
                "checks", Map.of(
                        "discover_mcp_tools", false,
                        "validate_request", true),
                "readiness_report", Map.of(
                        "feature_profile", AgentServingFeatureProfile.RESPONSES_AGENT,
                        "surface", "responses",
                        "model", "demo-response",
                        "checked_areas", List.of(
                                AgentServingReadinessReport.AREA_REQUEST_VALIDATION,
                                AgentServingReadinessReport.AREA_MCP_DISCOVERY),
                        "checks", nestedChecks)));

        assertTrue(report.ready(), report.issues().toString());
        assertEquals(AgentServingFeatureProfile.RESPONSES_AGENT, report.featureProfile().orElseThrow());
        assertEquals(AgentServingFeatureProfile.RESPONSES_AGENT, report.toReport().get("feature_profile"));
        assertEquals("responses", report.surface().orElseThrow());
        assertEquals("demo-response", report.model().orElseThrow());
        assertEquals(AgentServingRoute.of("responses", "demo-response", AgentServingFeatureProfile.RESPONSES_AGENT),
                report.route());
        assertEquals(List.of(
                        AgentServingReadinessReport.AREA_REQUEST_VALIDATION,
                        AgentServingReadinessReport.AREA_MCP_DISCOVERY),
                report.checkedAreas());
        assertEquals("chat", report.requestValidationCheck().surface());
        assertTrue(report.mcpDiscoveryCheck().skipped());
        assertFalse(report.mcpDiscoveryCheck().requested());
        assertEquals(List.of("MCP discovery skipped by caller"), report.warningMessages());
        assertEquals(List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER"), report.warningCodes());
        assertTrue(report.mcpDiscoveryCheck().hasWarningCode("MCP_DISCOVERY_SKIPPED_BY_CALLER"));
    }

    @Test
    void readsFeatureProfileFromFeatureNegotiationCheckDetailsWhenTopLevelMetadataIsMissing() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put(AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION, Map.of(
                "status", "ready",
                "ready", true,
                "requested", true,
                "blocking_messages", List.of(),
                "warning_messages", List.of(),
                "details", Map.of(
                        "feature_profile", AgentServingFeatureProfile.EMBEDDING_RAG,
                        "feature_profile_supported", true)));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "ready",
                "ready", true,
                "check_results", checks));

        assertTrue(report.ready(), report.issues().toString());
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, report.featureProfile().orElseThrow());
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, report.featureProfileOrDefault());
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, report.toMetadata().get("feature_profile"));
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, report.toReport().get("feature_profile"));
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG,
                report.featureNegotiationCheck().details().get("feature_profile"));
        assertEquals(List.of(AgentServingReadinessReport.AREA_FEATURE_NEGOTIATION), report.checkedAreas());
    }

    @Test
    void builderAcceptsTypedServingRouteDescriptor() {
        AgentServingRoute route = AgentServingRoute.of("chat-completions", "demo-chat", "chatAgent");
        AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                .route(route)
                .build();

        AgentServingReadinessReport report = AgentServingReadinessReport.builder()
                .route(route)
                .build();

        assertTrue(report.ready());
        assertEquals(route, report.route());
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, report.featureProfile().orElseThrow());
        assertEquals("chat", report.surface().orElseThrow());
        assertEquals("demo-chat", report.model().orElseThrow());
        assertEquals(route.toMetadata(), Map.of(
                "feature_profile", report.toMetadata().get("feature_profile"),
                "surface", report.toMetadata().get("surface"),
                "model", report.toMetadata().get("model")));
        assertTrue(report.routeComparison(preflight).matches());
        assertEquals(route, report.routeComparison(preflight).expected());
        assertEquals(route, report.routeComparison(preflight).actual());
        assertEquals(Map.of(
                        "matches", true,
                        "mismatch_fields", List.of(),
                        "mismatches", List.of(),
                        "mismatch_messages", List.of(),
                        "expected", route.toMetadata(),
                        "actual", route.toMetadata()),
                report.routeComparison(preflight).toMetadata());
        assertTrue(report.routeMatches(preflight));
        assertEquals(List.of(), report.routeMismatchFields(preflight));
        AgentServingRoute mismatchedRoute = AgentServingRoute.of(
                "embeddings",
                "demo-embed",
                AgentServingFeatureProfile.EMBEDDING_RAG);
        AgentServingRouteComparison mismatch = report.routeComparison(mismatchedRoute);
        assertFalse(report.routeMatches(mismatchedRoute));
        assertFalse(mismatch.matches());
        assertEquals(mismatchedRoute, mismatch.expected());
        assertEquals(route, mismatch.actual());
        assertEquals(List.of("surface", "model", "feature_profile"),
                report.routeMismatchFields(mismatchedRoute));
        assertEquals(List.of("surface", "model", "feature_profile"), mismatch.mismatchFields());
        assertEquals(List.of(
                        "surface requested=embeddings selected=chat",
                        "model requested=demo-embed selected=demo-chat",
                        "feature_profile requested=embedding_rag selected=chat_agent"),
                mismatch.mismatchMessages());
    }

    @Test
    void mapsReadinessReportChecksIntoTypedReport() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put(AgentServingReadinessReport.AREA_CAPABILITIES, Map.of(
                "status", "ready",
                "ready", true,
                "requested", true,
                "blocking_messages", List.of(),
                "warning_messages", List.of(),
                "details", Map.of("service_role", "inference_serving_engine")));
        checks.put(AgentServingReadinessReport.AREA_MCP_DISCOVERY, Map.of(
                "status", "skipped",
                "ready", true,
                "requested", false,
                "blocking_messages", List.of(),
                "warning_messages", List.of("MCP discovery skipped by caller"),
                "warning_codes", List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER")));
        checks.put("preflight", Map.of(
                "status", "blocked",
                "ready", false,
                "blocking_messages", List.of("agent request is not valid"),
                "blocking_codes", List.of("AGENT_REQUEST_INVALID"),
                "warning_messages", List.of()));
        checks.put(AgentServingReadinessReport.AREA_TOOL_VALIDATION, Map.of(
                "status", "ready",
                "ready", true,
                "requested", true,
                "blocking_messages", List.of(),
                "warning_messages", List.of("schema portability warning"),
                "warning_codes", List.of("TOOL_SCHEMA_PORTABILITY_WARNING")));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", AgentServingReadinessReport.OBJECT,
                "status", "blocked",
                "ready", false,
                "check_results", checks));

        assertFalse(report.ready());
        assertEquals(List.of(
                        AgentServingReadinessReport.AREA_CAPABILITIES,
                        AgentServingReadinessReport.AREA_MCP_DISCOVERY,
                        "preflight",
                        AgentServingReadinessReport.AREA_TOOL_VALIDATION),
                report.checkedAreas());
        assertTrue(report.check(AgentServingReadinessReport.AREA_MCP_DISCOVERY).skipped());
        assertFalse(report.check(AgentServingReadinessReport.AREA_MCP_DISCOVERY).requested());
        assertEquals(List.of("MCP discovery skipped by caller"),
                report.mcpDiscoveryCheck().warningMessages());
        assertEquals(List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER"),
                report.mcpDiscoveryCheck().warningCodes());
        assertEquals(false, map(map(report.checks()).get(AgentServingReadinessReport.AREA_MCP_DISCOVERY))
                .get("requested"));
        Map<String, Object> capabilityCheck = map(map(report.checks())
                .get(AgentServingReadinessReport.AREA_CAPABILITIES));
        assertEquals("inference_serving_engine", map(capabilityCheck.get("details")).get("service_role"));
        assertTrue(report.hasIssue("preflight", "agent request is not valid"));
        assertTrue(report.hasIssue("preflight", "AGENT_REQUEST_INVALID", "agent request is not valid"));
        assertTrue(report.check("preflight").hasBlockingCode("AGENT_REQUEST_INVALID"));
        assertEquals("Fix the request model, messages/input, tools, or RAG context before serving.",
                report.check("preflight").blockingIssueHints().get(0).remediation());
        assertTrue(report.mcpDiscoveryCheck().hasWarningCode("MCP_DISCOVERY_SKIPPED_BY_CALLER"));
        assertEquals(List.of("MCP discovery skipped by caller", "schema portability warning"),
                report.warningMessages());
        assertEquals(List.of("MCP_DISCOVERY_SKIPPED_BY_CALLER", "TOOL_SCHEMA_PORTABILITY_WARNING"),
                report.warningCodes());
        assertEquals(2L, report.toMetadata().get("ready_check_count"));
        assertEquals(1L, report.toMetadata().get("blocked_check_count"));
        assertEquals(1L, report.toMetadata().get("skipped_check_count"));
        assertEquals(report.checkedAreas(), report.toMetadata().get("checked_areas"));
    }

    @Test
    void preservesServerProvidedPreflightIssueHints() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put(AgentServingReadinessReport.AREA_TOOL_VALIDATION, Map.ofEntries(
                entry("status", "blocked"),
                entry("ready", false),
                entry("requested", true),
                entry("blocking_messages", List.of("tool definitions are not valid")),
                entry("blocking_codes", List.of("TOOL_DEFINITIONS_INVALID")),
                entry("warning_messages", List.of()),
                entry("warning_codes", List.of()),
                entry("issue_hints", List.of(Map.ofEntries(
                        entry("area", AgentServingReadinessReport.AREA_TOOL_VALIDATION),
                        entry("severity", "blocking"),
                        entry("message", "tool definitions are not valid"),
                        entry("code", "TOOL_DEFINITIONS_INVALID"),
                        entry("default_severity", "blocking"),
                        entry("summary", "Server-specific tool validation summary."),
                        entry("remediation", "Regenerate the tool contract through the service profile."))))));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "blocked",
                "ready", false,
                "check_results", checks));

        AgentServingReadinessReport.IssueHint checkHint =
                report.toolValidationCheck().blockingIssueHints().get(0);
        assertEquals("Server-specific tool validation summary.", checkHint.summary());
        assertEquals("Regenerate the tool contract through the service profile.", checkHint.remediation());
        assertEquals("Server-specific tool validation summary.", report.blockingIssueHints().get(0).summary());
        assertTrue(report.issueHintsByArea()
                .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                .stream()
                .anyMatch(hint -> "Regenerate the tool contract through the service profile."
                        .equals(hint.remediation())));

        Map<String, Object> checkMap = map(report.toolValidationCheck().toMap());
        assertEquals("Server-specific tool validation summary.",
                mapList(checkMap.get("issue_hints")).get(0).get("summary"));
        assertEquals("Regenerate the tool contract through the service profile.",
                mapList(report.toMetadata().get("issue_hints")).get(0).get("remediation"));
        assertEquals("Regenerate the tool contract through the service profile.",
                hintGroups(report.toMetadata().get("issue_hints_by_code"))
                        .get("TOOL_DEFINITIONS_INVALID")
                        .get(0)
                        .get("remediation"));
    }

    @Test
    void preservesAggregatedPreflightIssueHintsWhenCheckHintsAreMissing() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put(AgentServingReadinessReport.AREA_TOOL_VALIDATION, Map.of(
                "status", "blocked",
                "ready", false,
                "requested", true,
                "blocking_messages", List.of("tool definitions are not valid"),
                "blocking_codes", List.of("TOOL_DEFINITIONS_INVALID"),
                "warning_messages", List.of(),
                "warning_codes", List.of()));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "blocked",
                "ready", false,
                "check_results", checks,
                "issue_hints", List.of(Map.ofEntries(
                        entry("area", AgentServingReadinessReport.AREA_TOOL_VALIDATION),
                        entry("severity", "blocking"),
                        entry("message", "tool definitions are not valid"),
                        entry("code", "TOOL_DEFINITIONS_INVALID"),
                        entry("summary", "Aggregated server tool validation summary."),
                        entry("remediation", "Use the server profile to regenerate normalized tools.")))));

        assertEquals(1, report.issueHints().size());
        assertEquals("Aggregated server tool validation summary.",
                report.toolValidationCheck().blockingIssueHints().get(0).summary());
        assertEquals("Use the server profile to regenerate normalized tools.",
                report.remediationForCode("tool-definitions-invalid").orElseThrow());
        assertEquals("Aggregated server tool validation summary.",
                report.toolValidationCheck().issueHint("TOOL_DEFINITIONS_INVALID").orElseThrow().summary());
        assertEquals("Use the server profile to regenerate normalized tools.",
                mapList(map(report.toolValidationCheck().toMap()).get("issue_hints")).get(0).get("remediation"));
        assertEquals("Use the server profile to regenerate normalized tools.",
                mapList(report.toMetadata().get("issue_hints")).get(0).get("remediation"));
        assertEquals("Use the server profile to regenerate normalized tools.",
                hintGroups(report.toMetadata().get("issue_hints_by_area"))
                        .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                        .get(0)
                        .get("remediation"));
        assertEquals("Use the server profile to regenerate normalized tools.",
                mapList(report.toMetadata().get("remediation_plan")).get(0).get("remediation"));
        assertEquals(mapList(report.toMetadata().get("remediation_plan")),
                report.toMetadata().get("blocking_remediation_plan"));
        assertEquals(List.of(), report.toMetadata().get("warning_remediation_plan"));
    }

    @Test
    void groupsRemediationPlanByStableCodeAndAction() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "blocked",
                "ready", false,
                "issue_hints", List.of(
                        Map.ofEntries(
                                entry("area", AgentServingReadinessReport.AREA_TOOL_VALIDATION),
                                entry("severity", "blocking"),
                                entry("message", "tools[0] has no function schema"),
                                entry("code", "TOOL_DEFINITIONS_INVALID"),
                                entry("summary", "Tool definitions failed service validation."),
                                entry("remediation", "Regenerate the tool contract from the orchestrator profile.")),
                        Map.ofEntries(
                                entry("area", AgentServingReadinessReport.AREA_TOOL_VALIDATION),
                                entry("severity", "blocking"),
                                entry("message", "tools[1] has no parameters object"),
                                entry("code", "tool-definitions-invalid"),
                                entry("summary", "Tool definitions failed service validation."),
                                entry("remediation", "Regenerate the tool contract from the orchestrator profile.")))));

        assertFalse(report.ready());
        assertEquals(2, report.issueHintsForCode("TOOL_DEFINITIONS_INVALID").size());
        assertEquals(1, report.remediationPlan().size());
        AgentReadinessRemediation remediation = report.remediationPlan().get(0);
        assertEquals(AgentServingReadinessReport.AREA_TOOL_VALIDATION, remediation.area());
        assertEquals("TOOL_DEFINITIONS_INVALID", remediation.code());
        assertTrue(remediation.blocking());
        assertEquals(
                List.of("tools[0] has no function schema", "tools[1] has no parameters object"),
                remediation.messages());
        assertEquals("Regenerate the tool contract from the orchestrator profile.",
                remediation.toMap().get("remediation"));
        assertEquals(remediation, report.remediationPlanByArea()
                .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION)
                .get(0));
        assertEquals(List.of(remediation.toMap()), report.toMetadata().get("remediation_plan"));
        assertEquals(List.of(remediation.toMap()), report.toMetadata().get("blocking_remediation_plan"));
        assertEquals(List.of(), report.toMetadata().get("warning_remediation_plan"));
        assertEquals(List.of(remediation.toMap()),
                hintGroups(report.toMetadata().get("remediation_plan_by_area"))
                        .get(AgentServingReadinessReport.AREA_TOOL_VALIDATION));
        assertEquals(List.of(remediation.toMap()),
                hintGroups(report.toMetadata().get("remediation_plan_by_code"))
                        .get("TOOL_DEFINITIONS_INVALID"));
        assertEquals(List.of(remediation.toMap()),
                map(report.toolValidationCheck().toMap()).get("remediation_plan"));
        assertEquals(List.of(remediation.toMap()),
                map(report.toolValidationCheck().toMap()).get("blocking_remediation_plan"));
        assertEquals(List.of(), map(report.toolValidationCheck().toMap()).get("warning_remediation_plan"));
        assertEquals(List.of(remediation.toMap()),
                hintGroups(map(report.toolValidationCheck().toMap()).get("remediation_plan_by_code"))
                        .get("TOOL_DEFINITIONS_INVALID"));
        assertEquals(1, report.toolValidationCheck().remediationPlan().size());
        assertEquals(List.of(remediation),
                report.toolValidationCheck().remediationPlanForCode("tool-definitions-invalid"));
        assertEquals(List.of(remediation),
                report.toolValidationCheck().remediationPlanByCode().get("TOOL_DEFINITIONS_INVALID"));
    }

    @Test
    void readsNestedReadinessReportIssueHintsWhenTopLevelHintsAreMissing() {
        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "blocked",
                "ready", false,
                "readiness_report", Map.of(
                        "object", AgentServingReadinessReport.OBJECT,
                        "status", "blocked",
                        "ready", false,
                        "checked_areas", List.of(AgentServingReadinessReport.AREA_MODEL_ROUTE),
                        "issue_hints", List.of(Map.ofEntries(
                                entry("area", AgentServingReadinessReport.AREA_MODEL_ROUTE),
                                entry("severity", "blocking"),
                                entry("message", "model is not available through Gollek serving"),
                                entry("code", "MODEL_ROUTE_UNAVAILABLE"),
                                entry("summary", "Nested server route summary."),
                                entry("remediation", "Select a served route from the model catalog."))))));

        assertFalse(report.ready());
        assertTrue(report.hasIssueCode("MODEL_ROUTE_UNAVAILABLE"));
        assertEquals(1, report.issueHints().size());
        assertEquals("Nested server route summary.", report.modelRouteCheck().blockingIssueHints().get(0).summary());
        assertEquals("Select a served route from the model catalog.",
                mapList(report.toMetadata().get("issue_hints")).get(0).get("remediation"));
    }

    @Test
    void mapsServerPreflightCheckDetailsIntoTypedAccessors() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checks.put(AgentServingReadinessReport.AREA_REQUEST_VALIDATION, Map.of(
                "status", "ready",
                "ready", true,
                "requested", true,
                "blocking_messages", List.of(),
                "warning_messages", List.of(),
                "details", Map.of(
                        "surface", "embeddings",
                        "model", "demo-embed",
                        "boundary", Map.of(
                                "validation_only", true,
                                "model_invoked", false,
                                "tool_execution", false,
                                "retrieval_execution", false),
                        "embedding", Map.of(
                                "input_count", 2,
                                "input_lengths", List.of(5, 4),
                                "requested_dimensions", 768,
                                "encoding_format", "float",
                                "metadata_keys", List.of("tenant"),
                                "rag", Map.of(
                                        "embedding_generation", true,
                                        "retrieval_execution", false,
                                        "retrieval_policy_owned_by", "agent_orchestrator",
                                        "vector_store_owned_by", "agent_orchestrator",
                                        "storage_owned_by_orchestrator", true)))));

        AgentServingReadinessReport report = AgentServingReadinessReport.fromPreflightResponse(Map.of(
                "object", "gollek.agent_preflight",
                "status", "ready",
                "ready", true,
                "check_results", checks));

        assertTrue(report.ready(), report.issues().toString());
        assertTrue(report.hasCheck(AgentServingReadinessReport.AREA_REQUEST_VALIDATION));
        assertFalse(report.hasCheck(AgentServingReadinessReport.AREA_TOOL_VALIDATION));
        assertEquals(null, report.toolValidationCheck());
        assertEquals(List.of(AgentServingReadinessReport.AREA_REQUEST_VALIDATION),
                report.toMetadata().get("checked_areas"));
        AgentServingReadinessReport.Check check = report.requestValidationCheck();
        assertEquals("embeddings", check.surface());
        assertEquals("demo-embed", check.model());
        assertTrue(check.boundaryDetails().validationOnly());
        assertFalse(check.boundaryDetails().modelInvoked());
        AgentServingReadinessReport.EmbeddingDetails embedding = check.embeddingDetails();
        assertEquals(2, embedding.inputCount());
        assertEquals(List.of(5, 4), embedding.inputLengths());
        assertEquals(768, embedding.requestedDimensions());
        assertEquals("float", embedding.encodingFormat());
        assertEquals(List.of("tenant"), embedding.metadataKeys());
        assertEquals(true, embedding.rag().embeddingGeneration());
        assertEquals(false, embedding.rag().retrievalExecution());
        assertTrue(embedding.storageOwnedByOrchestrator());
    }

    private static AgentCapabilitiesView capabilitiesView() {
        return AgentCapabilitiesView.from(Map.ofEntries(
                entry("object", AgentServingFeatureCatalog.CAPABILITIES_OBJECT),
                entry("version", AgentServingFeatureCatalog.CONTRACT_VERSION),
                entry("contract_version", AgentServingFeatureCatalog.CONTRACT_VERSION),
                entry("supported_contract_versions", AgentServingFeatureCatalog.SUPPORTED_CONTRACT_VERSIONS),
                entry("feature_negotiation", AgentServingFeatureCatalog.featureNegotiation()),
                entry("service_role", "inference_serving_engine"),
                entry("compatibility", AgentServingFeatureCatalog.compatibilityFeatures()),
                entry("endpoints", Map.of(
                        "openai_chat_completions", "/v1/chat/completions",
                        "openai_responses", "/v1/responses",
                        "openai_embeddings", "/v1/embeddings",
                        "model_capabilities", "/v1/models/{id}/capabilities",
                        "agent_contract", "/v1/agent/contract",
                        "agent_readiness_issues", "/v1/agent/readiness/issues",
                        "agent_preflight", "/v1/agent/preflight",
                        "agent_validation", "/v1/agent/validate",
                        "agent_tool_validation", "/v1/agent/tools/validate",
                        "mcp_tools", "/v1/mcp/tools")),
                entry("agent_boundary", Map.of(
                        "gollek_owns", List.of("model serving"),
                        "agent_orchestrator_owns", List.of("tool execution loops"))),
                entry("auth", Map.of(
                        "x_api_key_header", "X-API-Key",
                        "authorization_header", "Bearer token")),
                entry("traceability", Map.of(
                        "request_id_header", AgentRequestOptions.REQUEST_ID_HEADER))));
    }

    private static AgentServingContract contractView() {
        return AgentServingContract.from(Map.ofEntries(
                entry("object", AgentServingFeatureCatalog.CONTRACT_OBJECT),
                entry("version", AgentServingFeatureCatalog.CONTRACT_VERSION),
                entry("contract_version", AgentServingFeatureCatalog.CONTRACT_VERSION),
                entry("supported_contract_versions", AgentServingFeatureCatalog.SUPPORTED_CONTRACT_VERSIONS),
                entry("compatibility", AgentServingFeatureCatalog.compatibilityFeatures()),
                entry("feature_negotiation", AgentServingFeatureCatalog.featureNegotiation()),
                entry("service_role", "inference_serving_engine"),
                entry("boundary", Map.of(
                        "gollek_owns", List.of(
                                "model_serving",
                                "provider_routing",
                                "system_prompt_mapping",
                                "tool_schema_ingestion",
                                "tool_contract_validation",
                                "mcp_registry_discovery",
                                "embedding_generation",
                                "rag_context_injection"),
                        "agent_orchestrator_owns", List.of(
                                "planning",
                                "memory_policy",
                                "retrieval_policy",
                                "vector_store_ownership",
                                "tool_authorization",
                                "tool_execution",
                                "tool_result_loop",
                                "workflow_state"),
                        "tool_execution", false,
                        "retrieval_execution", false)),
                entry("endpoints", Map.ofEntries(
                        entry("chat_completions", Map.of("method", "POST", "path", "/v1/chat/completions")),
                        entry("responses", Map.of("method", "POST", "path", "/v1/responses")),
                        entry("embeddings", Map.of("method", "POST", "path", "/v1/embeddings")),
                        entry("model_capabilities", Map.of("method", "GET", "path", "/v1/models/{id}/capabilities")),
                        entry("mcp_tools", Map.of("method", "GET", "path", "/v1/mcp/tools")),
                        entry("agent_capabilities", Map.of("method", "GET", "path", "/v1/agent/capabilities")),
                        entry("agent_contract", Map.of("method", "GET", "path", "/v1/agent/contract")),
                        entry("agent_readiness_issues", Map.of("method", "GET", "path", "/v1/agent/readiness/issues")),
                        entry("agent_preflight", Map.of("method", "POST", "path", "/v1/agent/preflight")),
                        entry("agent_validation", Map.of("method", "POST", "path", "/v1/agent/validate")),
                        entry("agent_tool_validation", Map.of("method", "POST", "path", "/v1/agent/tools/validate")))),
                entry("streaming", Map.of(
                        "done_sentinel", "[DONE]",
                        "chat_completions_events", List.of("chat.completion.chunk"),
                        "responses_events", List.of("response.output_text.delta")))));
    }

    private static AgentModelCapabilitiesView modelRouteView() {
        return AgentModelCapabilitiesView.from(Map.of(
                "model_id", "demo",
                "available", true,
                "api_contract", Map.of(
                        "chat_completions", true,
                        "responses", true,
                        "system_prompt", true,
                        "tools_request_schema", true,
                        "rag_context_injection", true),
                "openai_compatibility", Map.of(
                        "chat_completions", true,
                        "responses", true),
                "inference", Map.of(
                        "completion", true,
                        "system_prompt", true),
                "tooling", Map.of(
                        "tool_definitions", true,
                        "tool_execution", false),
                "rag", Map.of(
                        "context_injection", true,
                        "retrieval_policy", false,
                        "vector_store_ownership", false)));
    }

    private static AgentModelCapabilitiesView embeddingRouteView() {
        return AgentModelCapabilitiesView.from(Map.of(
                "model_id", "demo-embed",
                "available", true,
                "api_contract", Map.of("embeddings_endpoint", true),
                "openai_compatibility", Map.of("embeddings", true),
                "embeddings", Map.of(
                        "generation", true,
                        "endpoint", "/v1/embeddings",
                        "openai_compatible", true,
                        "dimensions", 768,
                        "encoding_formats", List.of("float", "base64"),
                        "input_aliases", List.of("input", "inputs"),
                        "batch_inputs", true,
                        "metadata_passthrough", true,
                        "retrieval_policy", false,
                        "vector_store_ownership", false),
                "modalities", Map.of("embeddings", true),
                "rag", Map.of(
                        "retrieval_policy", false,
                        "vector_store_ownership", false),
                "tooling", Map.of("tool_execution", false)));
    }

    private static AgentMcpDiscoveryView mcpDiscoveryView() {
        return AgentMcpDiscoveryView.from(Map.of(
                "available", true,
                "boundary", Map.of(
                        "role", "discovery_only",
                        "tool_execution", false),
                "tools", List.of(Map.of(
                        "type", "function",
                        "execution", false,
                        "function", Map.of(
                                "name", "lookup",
                                "parameters", Map.of("type", "object"))))));
    }

    private static AgentToolValidationView toolValidationView() {
        return AgentToolValidationView.from(Map.of(
                "valid", true,
                "model_invoked", false,
                "boundary", Map.of(
                        "validation_only", true,
                        "tool_execution", false,
                        "tool_authorization", false),
                "warnings", List.of(),
                "normalized", List.of(Map.of(
                        "index", 0,
                        "name", "lookup",
                        "type", "function"))));
    }

    private static AgentValidationView requestValidationView() {
        return AgentValidationView.from(Map.of(
                "surface", "chat",
                "valid", true,
                "model_invoked", false,
                "boundary", Map.of(
                        "validation_only", true,
                        "tool_execution", false,
                        "retrieval_execution", false),
                "normalized", Map.of(
                        "model", "demo",
                        "streaming", false,
                        "message_count", 2,
                        "tool_count", 1,
                        "parameter_keys", List.of("model", "messages", "tools", "rag_context"),
                        "rag", Map.of(
                                "injected", true,
                                "items", 1,
                                "alias", "rag_context"),
                        "tool_contract", Map.of("valid", true))));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
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
