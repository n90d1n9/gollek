package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Aggregates Gollek serving-boundary checks for an external agent runtime.
 *
 * <p>The report is intentionally a read-only preflight helper. Callers provide
 * discovery and validation views that were already fetched from Gollek; the
 * report does not choose a workflow, execute tools, run retrieval, or invoke a
 * model.
 */
public final class AgentServingReadinessReport {
    public static final String OBJECT = AgentReadinessMetadata.OBJECT_READINESS_REPORT;
    public static final String AREA_CAPABILITIES = "capabilities";
    public static final String AREA_CONTRACT = "contract";
    public static final String AREA_FEATURE_NEGOTIATION = "feature_negotiation";
    public static final String AREA_MODEL_ROUTE = "model_route";
    public static final String AREA_MCP_DISCOVERY = "mcp_discovery";
    public static final String AREA_TOOL_VALIDATION = "tool_validation";
    public static final String AREA_REQUEST_VALIDATION = "request_validation";

    private final List<Issue> issues;
    private final List<String> checkedAreas;
    private final Map<String, Check> sourceChecks;
    private final List<IssueHint> sourceIssueHints;
    private final AgentServingRoute route;

    private AgentServingReadinessReport(List<Issue> issues, List<String> checkedAreas) {
        this(issues, checkedAreas, null, null, null, null, null);
    }

    private AgentServingReadinessReport(
            List<Issue> issues,
            List<String> checkedAreas,
            Map<String, Check> sourceChecks) {
        this(issues, checkedAreas, sourceChecks, null, null, null, null);
    }

    private AgentServingReadinessReport(
            List<Issue> issues,
            List<String> checkedAreas,
            Map<String, Check> sourceChecks,
            List<IssueHint> sourceIssueHints,
            String featureProfile,
            String surface,
            String model) {
        this.issues = issues == null ? List.of() : List.copyOf(issues);
        this.checkedAreas = checkedAreas == null ? areasFromIssues(this.issues) : List.copyOf(checkedAreas);
        this.sourceChecks = sourceChecks == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(sourceChecks));
        this.sourceIssueHints = sourceIssueHints == null ? List.of() : List.copyOf(sourceIssueHints);
        this.route = AgentServingRoute.of(surface, model, featureProfile);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static AgentServingReadinessReport from(
            AgentCapabilitiesView capabilities,
            AgentServingContract contract,
            AgentModelCapabilitiesView modelRoute) {
        return builder()
                .capabilities(capabilities)
                .contract(contract)
                .modelRoute(modelRoute)
                .build();
    }

    /**
     * Builds the typed readiness report from Gollek's server-side
     * {@code /v1/agent/preflight} response.
     */
    public static AgentServingReadinessReport fromPreflightResponse(Map<String, Object> response) {
        if (response == null || response.isEmpty()) {
            return builder()
                    .blocking("preflight", "preflight response is missing")
                    .build();
        }
        List<IssueHint> responseHints = responseIssueHints(response);
        List<Issue> parsed = parseIssues(response, responseHints);
        if (parsed.isEmpty() && blocked(response)) {
            parsed.add(new Issue("preflight", Severity.BLOCKING, "server preflight reported blocked"));
        }
        Map<String, Check> checks = parseChecks(response, responseHints);
        return new AgentServingReadinessReport(
                parsed,
                checkedAreas(response, parsed),
                checks,
                responseHints,
                featureProfile(response, checks),
                routeSurface(response, checks),
                routeModel(response, checks));
    }

    public boolean ready() {
        return blockingIssues().isEmpty();
    }

    public String status() {
        return ready() ? "ready" : "blocked";
    }

    public boolean hasWarnings() {
        return !warnings().isEmpty();
    }

    public List<Issue> issues() {
        return issues;
    }

    public List<String> checkedAreas() {
        return checkedAreas;
    }

    public Optional<String> featureProfile() {
        return Optional.ofNullable(route.featureProfile());
    }

    public String featureProfileOrDefault() {
        return route.featureProfileOrDefault();
    }

    public AgentServingRoute route() {
        return route;
    }

    public boolean routeMatches(AgentServingRoute expected) {
        return route.matches(expected);
    }

    public boolean routeMatches(AgentServingPreflightRequest preflight) {
        return preflight == null || routeMatches(preflight.route());
    }

    public AgentServingRouteComparison routeComparison(AgentServingRoute expected) {
        return route.comparison(expected);
    }

    public AgentServingRouteComparison routeComparison(AgentServingPreflightRequest preflight) {
        return routeComparison(preflight == null ? null : preflight.route());
    }

    public List<String> routeMismatchFields(AgentServingRoute expected) {
        return route.mismatchFields(expected);
    }

    public List<String> routeMismatchFields(AgentServingPreflightRequest preflight) {
        return preflight == null ? List.of() : routeMismatchFields(preflight.route());
    }

    public Optional<String> surface() {
        return Optional.ofNullable(route.surface());
    }

    public Optional<String> model() {
        return Optional.ofNullable(route.model());
    }

    public List<Issue> blockingIssues() {
        return issues.stream()
                .filter(issue -> issue.severity() == Severity.BLOCKING)
                .toList();
    }

    public List<Issue> warnings() {
        return issues.stream()
                .filter(issue -> issue.severity() == Severity.WARNING)
                .toList();
    }

    public List<String> blockingMessages() {
        return messages(blockingIssues());
    }

    public List<String> warningMessages() {
        return messages(warnings());
    }

    public List<String> blockingCodes() {
        return codes(blockingIssues());
    }

    public List<String> warningCodes() {
        return codes(warnings());
    }

    public Map<String, List<String>> issueCodesByArea() {
        return codesByArea(issues);
    }

    public Map<String, List<String>> blockingCodesByArea() {
        return codesByArea(blockingIssues());
    }

    public Map<String, List<String>> warningCodesByArea() {
        return codesByArea(warnings());
    }

    public List<IssueHint> issueHints() {
        return checkIssueHints(null);
    }

    public List<IssueHint> blockingIssueHints() {
        return checkIssueHints(Severity.BLOCKING);
    }

    public List<IssueHint> warningIssueHints() {
        return checkIssueHints(Severity.WARNING);
    }

    public Map<String, List<IssueHint>> issueHintsByArea() {
        return checkIssueHintsByArea();
    }

    public Map<String, List<IssueHint>> issueHintsByCode() {
        return hintsByCode(issueHints());
    }

    public List<IssueHint> issueHintsForCode(String code) {
        return hintsForCode(issueHints(), code);
    }

    public Optional<IssueHint> issueHint(String code) {
        return firstHintForCode(issueHints(), code);
    }

    public Optional<String> remediationForCode(String code) {
        return issueHint(code).map(IssueHint::remediation);
    }

    public List<AgentReadinessRemediation> remediationPlan() {
        return AgentReadinessRemediation.fromIssueHints(issueHints());
    }

    public List<AgentReadinessRemediation> blockingRemediationPlan() {
        return AgentReadinessRemediation.fromIssueHints(blockingIssueHints());
    }

    public List<AgentReadinessRemediation> warningRemediationPlan() {
        return AgentReadinessRemediation.fromIssueHints(warningIssueHints());
    }

    public Map<String, List<AgentReadinessRemediation>> remediationPlanByArea() {
        return AgentReadinessRemediation.byArea(remediationPlan());
    }

    public Map<String, List<AgentReadinessRemediation>> remediationPlanByCode() {
        return AgentReadinessRemediation.byCode(remediationPlan());
    }

    public List<AgentReadinessRemediation> remediationPlanForCode(String code) {
        return AgentReadinessRemediation.forCode(remediationPlan(), code);
    }

    public Map<String, List<String>> issuesByArea() {
        Map<String, List<String>> grouped = new LinkedHashMap<>();
        for (Issue issue : issues) {
            grouped.computeIfAbsent(issue.area(), ignored -> new ArrayList<>())
                    .add(issue.message());
        }
        Map<String, List<String>> out = new LinkedHashMap<>();
        grouped.forEach((area, messages) -> out.put(area, List.copyOf(messages)));
        return Collections.unmodifiableMap(out);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("object", OBJECT);
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.putAll(route.toMetadata());
        metadata.putAll(AgentReadinessMetadata.fromChecks(checks()));
        List<Map<String, Object>> hints = issueHints().stream()
                .map(IssueHint::toMap)
                .toList();
        metadata.put("issue_hints", hints);
        metadata.put("issue_hints_by_area", AgentReadinessMetadata.issueHintsByArea(hints));
        metadata.put("issue_hints_by_code", AgentReadinessMetadata.issueHintsByCode(hints));
        List<AgentReadinessRemediation> remediations = remediationPlan();
        metadata.put("remediation_plan", AgentReadinessRemediation.toMaps(remediations));
        metadata.put("blocking_remediation_plan",
                AgentReadinessRemediation.toMaps(AgentReadinessRemediation.blocking(remediations)));
        metadata.put("warning_remediation_plan",
                AgentReadinessRemediation.toMaps(AgentReadinessRemediation.warning(remediations)));
        metadata.put("remediation_plan_by_area", AgentReadinessRemediation.toMapsByArea(remediations));
        metadata.put("remediation_plan_by_code", AgentReadinessRemediation.toMapsByCode(remediations));
        return Collections.unmodifiableMap(metadata);
    }

    public Map<String, Object> toReport() {
        Map<String, Object> report = new LinkedHashMap<>(toMetadata());
        report.put("boundary", Map.of(
                "validation_only", true,
                "model_invoked", false,
                "tool_execution", false,
                "retrieval_execution", false,
                "tool_authorization", false));
        report.put("checks", checks());
        return Collections.unmodifiableMap(report);
    }

    public Map<String, Object> checks() {
        Map<String, Object> checks = new LinkedHashMap<>();
        checkResults().forEach((area, check) -> checks.put(area, check.toMap()));
        return Collections.unmodifiableMap(checks);
    }

    public Map<String, Check> checkResults() {
        Map<String, Check> typed = new LinkedHashMap<>(sourceChecks);
        for (String area : checkedAreas) {
            typed.computeIfAbsent(area, ignored -> checkFromIssues(area));
        }
        return Collections.unmodifiableMap(typed);
    }

    public Check check(String area) {
        if (area == null || area.isBlank()) {
            return null;
        }
        return checkResults().get(area);
    }

    public boolean hasCheck(String area) {
        return check(area) != null;
    }

    public Check capabilitiesCheck() {
        return check(AREA_CAPABILITIES);
    }

    public Check contractCheck() {
        return check(AREA_CONTRACT);
    }

    public Check featureNegotiationCheck() {
        return check(AREA_FEATURE_NEGOTIATION);
    }

    public Check modelRouteCheck() {
        return check(AREA_MODEL_ROUTE);
    }

    public Check mcpDiscoveryCheck() {
        return check(AREA_MCP_DISCOVERY);
    }

    public Check toolValidationCheck() {
        return check(AREA_TOOL_VALIDATION);
    }

    public Check requestValidationCheck() {
        return check(AREA_REQUEST_VALIDATION);
    }

    public String summary() {
        if (ready()) {
            return hasWarnings()
                    ? "Gollek serving preflight ready with " + warnings().size() + " warning(s)"
                    : "Gollek serving preflight ready";
        }
        return "Gollek serving preflight blocked by " + blockingIssues().size() + " issue(s): " + issuesByArea();
    }

    public AgentServingReadinessReport requireReady() throws AgentIntegrationException {
        return requireReady("Gollek serving preflight failed");
    }

    public AgentServingReadinessReport requireReady(String message) throws AgentIntegrationException {
        if (!ready()) {
            String prefix = message == null || message.isBlank() ? "Gollek serving preflight failed" : message;
            throw new AgentIntegrationException(
                    "SDK_ERR_AGENT_PREFLIGHT",
                    prefix + ": " + issuesByArea());
        }
        return this;
    }

    public boolean hasIssue(String area, String message) {
        return issues.stream().anyMatch(issue ->
                issue.area().equals(area) && issue.message().equals(message));
    }

    public boolean hasIssue(String area, String code, String message) {
        String normalizedCode = AgentReadinessIssueCodes.normalize(code);
        return issues.stream().anyMatch(issue ->
                issue.area().equals(area)
                        && issue.code().equals(normalizedCode)
                        && issue.message().equals(message));
    }

    public boolean hasIssueCode(String code) {
        String normalizedCode = AgentReadinessIssueCodes.normalize(code);
        return normalizedCode != null && issues.stream().anyMatch(issue -> issue.code().equals(normalizedCode));
    }

    public boolean hasBlockingCode(String code) {
        return containsCode(blockingCodes(), code);
    }

    public boolean hasWarningCode(String code) {
        return containsCode(warningCodes(), code);
    }

    public enum Severity {
        BLOCKING,
        WARNING
    }

    public record Issue(String area, Severity severity, String message, String code) {
        public Issue(String area, Severity severity, String message) {
            this(area, severity, message, null);
        }

        public Issue {
            area = area == null || area.isBlank() ? "general" : area;
            severity = severity == null ? Severity.BLOCKING : severity;
            message = message == null || message.isBlank() ? "unspecified issue" : message;
            code = AgentReadinessIssueCodes.resolve(code, area, severity, message);
        }

        public AgentReadinessIssueCodes.CatalogEntry catalogEntry() {
            return AgentReadinessIssueCodes.describe(code);
        }

        public IssueHint toHint() {
            return IssueHint.from(this);
        }
    }

    public record IssueHint(
            String area,
            Severity severity,
            String message,
            String code,
            String defaultSeverity,
            String summary,
            String remediation) {
        public IssueHint {
            area = area == null || area.isBlank() ? "general" : area;
            severity = severity == null ? Severity.BLOCKING : severity;
            message = message == null || message.isBlank() ? "unspecified issue" : message;
            code = AgentReadinessIssueCodes.resolve(code, area, severity, message);
            AgentReadinessIssueCodes.CatalogEntry catalog = AgentReadinessIssueCodes.describe(code);
            defaultSeverity = defaultSeverity == null || defaultSeverity.isBlank()
                    ? catalog.defaultSeverity()
                    : defaultSeverity;
            summary = summary == null || summary.isBlank() ? catalog.summary() : summary;
            remediation = remediation == null || remediation.isBlank() ? catalog.remediation() : remediation;
        }

        public static IssueHint from(Issue issue) {
            if (issue == null) {
                return null;
            }
            AgentReadinessIssueCodes.CatalogEntry catalog = issue.catalogEntry();
            return new IssueHint(
                    issue.area(),
                    issue.severity(),
                    issue.message(),
                    issue.code(),
                    catalog.defaultSeverity(),
                    catalog.summary(),
                    catalog.remediation());
        }

        public AgentReadinessIssueCodes.CatalogEntry catalogEntry() {
            return AgentReadinessIssueCodes.describe(code);
        }

        public boolean blockingIssue() {
            return severity == Severity.BLOCKING;
        }

        public boolean warningIssue() {
            return severity == Severity.WARNING;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("area", area);
            out.put("severity", severity.name().toLowerCase(Locale.ROOT));
            out.put("message", message);
            out.put("code", code);
            out.put("default_severity", defaultSeverity);
            out.put("summary", summary);
            out.put("remediation", remediation);
            return Collections.unmodifiableMap(out);
        }
    }

    public record Check(
            String area,
            String status,
            boolean ready,
            boolean requested,
            List<String> blockingMessages,
            List<String> warningMessages,
            List<String> blockingCodes,
            List<String> warningCodes,
            Map<String, Object> details,
            List<IssueHint> suppliedIssueHints) {
        public Check {
            area = area == null || area.isBlank() ? "general" : area;
            blockingMessages = blockingMessages == null ? List.of() : List.copyOf(blockingMessages);
            warningMessages = warningMessages == null ? List.of() : List.copyOf(warningMessages);
            blockingCodes = issueCodes(blockingCodes, area, Severity.BLOCKING, blockingMessages);
            warningCodes = issueCodes(warningCodes, area, Severity.WARNING, warningMessages);
            details = details == null || details.isEmpty()
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(details));
            suppliedIssueHints = suppliedIssueHints == null ? List.of() : List.copyOf(suppliedIssueHints);
            status = status == null || status.isBlank()
                    ? (blockingMessages.isEmpty() ? "ready" : "blocked")
                    : status.trim().toLowerCase(Locale.ROOT);
            if ("blocked".equals(status)) {
                ready = false;
            } else if ("ready".equals(status) || "skipped".equals(status)) {
                ready = true;
            }
            if ("skipped".equals(status)) {
                requested = false;
            }
        }

        public Check(
                String area,
                String status,
                boolean ready,
                boolean requested,
                List<String> blockingMessages,
                List<String> warningMessages) {
            this(area, status, ready, requested, blockingMessages, warningMessages, List.of(), List.of(), Map.of());
        }

        public Check(
                String area,
                String status,
                boolean ready,
                boolean requested,
                List<String> blockingMessages,
                List<String> warningMessages,
                Map<String, Object> details) {
            this(area, status, ready, requested, blockingMessages, warningMessages, List.of(), List.of(), details);
        }

        public Check(
                String area,
                String status,
                boolean ready,
                boolean requested,
                List<String> blockingMessages,
                List<String> warningMessages,
                List<String> blockingCodes,
                List<String> warningCodes,
                Map<String, Object> details) {
            this(area, status, ready, requested, blockingMessages, warningMessages,
                    blockingCodes, warningCodes, details, List.of());
        }

        public boolean blocked() {
            return "blocked".equals(status) || !ready || !blockingMessages.isEmpty();
        }

        public boolean skipped() {
            return "skipped".equals(status);
        }

        public boolean hasWarnings() {
            return !warningMessages.isEmpty();
        }

        public boolean hasIssueCode(String code) {
            return hasBlockingCode(code) || hasWarningCode(code);
        }

        public boolean hasBlockingCode(String code) {
            return containsCode(blockingCodes, code);
        }

        public boolean hasWarningCode(String code) {
            return containsCode(warningCodes, code);
        }

        public List<IssueHint> issueHints() {
            if (!suppliedIssueHints.isEmpty()) {
                return suppliedIssueHints;
            }
            List<IssueHint> out = new ArrayList<>();
            out.addAll(blockingIssueHints());
            out.addAll(warningIssueHints());
            return List.copyOf(out);
        }

        public List<IssueHint> blockingIssueHints() {
            if (!suppliedIssueHints.isEmpty()) {
                return suppliedIssueHints.stream()
                        .filter(IssueHint::blockingIssue)
                        .toList();
            }
            return hints(area, Severity.BLOCKING, blockingMessages, blockingCodes);
        }

        public List<IssueHint> warningIssueHints() {
            if (!suppliedIssueHints.isEmpty()) {
                return suppliedIssueHints.stream()
                        .filter(IssueHint::warningIssue)
                        .toList();
            }
            return hints(area, Severity.WARNING, warningMessages, warningCodes);
        }

        public Map<String, List<IssueHint>> issueHintsByCode() {
            return hintsByCode(issueHints());
        }

        public List<IssueHint> issueHintsForCode(String code) {
            return hintsForCode(issueHints(), code);
        }

        public Optional<IssueHint> issueHint(String code) {
            return firstHintForCode(issueHints(), code);
        }

        public Optional<String> remediationForCode(String code) {
            return issueHint(code).map(IssueHint::remediation);
        }

        public List<AgentReadinessRemediation> remediationPlan() {
            return AgentReadinessRemediation.fromIssueHints(issueHints());
        }

        public Map<String, List<AgentReadinessRemediation>> remediationPlanByCode() {
            return AgentReadinessRemediation.byCode(remediationPlan());
        }

        public List<AgentReadinessRemediation> remediationPlanForCode(String code) {
            return AgentReadinessRemediation.forCode(remediationPlan(), code);
        }

        public String surface() {
            return text(details.get("surface"));
        }

        public String model() {
            return text(details.get("model"));
        }

        public BoundaryDetails boundaryDetails() {
            Map<String, Object> boundary = objectMap(details.get("boundary"));
            if (boundary.isEmpty()) {
                return BoundaryDetails.empty();
            }
            return new BoundaryDetails(
                    boolOrDefault(boundary.get("validation_only"), false),
                    boolOrDefault(boundary.get("model_invoked"), false),
                    boolOrDefault(boundary.get("tool_execution"), false),
                    boolOrDefault(boundary.get("retrieval_execution"), false));
        }

        public RequestDetails requestDetails() {
            Map<String, Object> request = objectMap(details.get("request"));
            if (request.isEmpty()) {
                return RequestDetails.empty();
            }
            return new RequestDetails(
                    boolOrDefault(request.get("streaming"), false),
                    intOrDefault(request.get("message_count"), 0),
                    intOrDefault(request.get("input_count"), 0),
                    intOrDefault(request.get("tool_count"), 0),
                    stringList(request.get("parameter_keys")),
                    requestRagDetails(request.get("rag")),
                    toolContractDetails(request.get("tool_contract")));
        }

        public EmbeddingDetails embeddingDetails() {
            Map<String, Object> embedding = objectMap(details.get("embedding"));
            if (embedding.isEmpty()) {
                return EmbeddingDetails.empty();
            }
            return new EmbeddingDetails(
                    intOrDefault(embedding.get("input_count"), 0),
                    intList(embedding.get("input_lengths")),
                    intValue(embedding.get("requested_dimensions")),
                    text(embedding.get("encoding_format")),
                    stringList(embedding.get("parameter_keys")),
                    stringList(embedding.get("metadata_keys")),
                    embeddingRagDetails(embedding.get("rag")));
        }

        public boolean hasRequestDetails() {
            return !objectMap(details.get("request")).isEmpty();
        }

        public boolean hasEmbeddingDetails() {
            return !objectMap(details.get("embedding")).isEmpty();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("status", status);
            value.put("ready", ready);
            value.put("requested", requested);
            value.put("blocking_messages", blockingMessages);
            value.put("warning_messages", warningMessages);
            value.put("blocking_codes", blockingCodes);
            value.put("warning_codes", warningCodes);
            value.put("issue_hints", issueHints().stream()
                    .map(IssueHint::toMap)
                    .toList());
            List<AgentReadinessRemediation> remediations = remediationPlan();
            value.put("remediation_plan", AgentReadinessRemediation.toMaps(remediations));
            value.put("blocking_remediation_plan",
                    AgentReadinessRemediation.toMaps(AgentReadinessRemediation.blocking(remediations)));
            value.put("warning_remediation_plan",
                    AgentReadinessRemediation.toMaps(AgentReadinessRemediation.warning(remediations)));
            value.put("remediation_plan_by_code", AgentReadinessRemediation.toMapsByCode(remediations));
            if (!details.isEmpty()) {
                value.put("details", details);
            }
            return Collections.unmodifiableMap(value);
        }
    }

    public record BoundaryDetails(
            boolean validationOnly,
            boolean modelInvoked,
            boolean toolExecution,
            boolean retrievalExecution) {
        private static BoundaryDetails empty() {
            return new BoundaryDetails(false, false, false, false);
        }
    }

    public record RequestDetails(
            boolean streaming,
            int messageCount,
            int inputCount,
            int toolCount,
            List<String> parameterKeys,
            RequestRagDetails rag,
            ToolContractDetails toolContract) {
        public RequestDetails {
            parameterKeys = parameterKeys == null ? List.of() : List.copyOf(parameterKeys);
            rag = rag == null ? RequestRagDetails.empty() : rag;
            toolContract = toolContract == null ? ToolContractDetails.empty() : toolContract;
        }

        private static RequestDetails empty() {
            return new RequestDetails(false, 0, 0, 0, List.of(), RequestRagDetails.empty(), ToolContractDetails.empty());
        }
    }

    public record RequestRagDetails(boolean injected, int items, String alias) {
        private static RequestRagDetails empty() {
            return new RequestRagDetails(false, 0, null);
        }
    }

    public record ToolContractDetails(boolean valid, int warningCount) {
        private static ToolContractDetails empty() {
            return new ToolContractDetails(false, 0);
        }
    }

    public record EmbeddingDetails(
            int inputCount,
            List<Integer> inputLengths,
            Integer requestedDimensions,
            String encodingFormat,
            List<String> parameterKeys,
            List<String> metadataKeys,
            EmbeddingRagDetails rag) {
        public EmbeddingDetails {
            inputLengths = inputLengths == null ? List.of() : List.copyOf(inputLengths);
            parameterKeys = parameterKeys == null ? List.of() : List.copyOf(parameterKeys);
            metadataKeys = metadataKeys == null ? List.of() : List.copyOf(metadataKeys);
            rag = rag == null ? EmbeddingRagDetails.empty() : rag;
        }

        public boolean storageOwnedByOrchestrator() {
            return rag.storageOwnedByOrchestrator();
        }

        private static EmbeddingDetails empty() {
            return new EmbeddingDetails(0, List.of(), null, null, List.of(), List.of(), EmbeddingRagDetails.empty());
        }
    }

    public record EmbeddingRagDetails(
            boolean embeddingGeneration,
            boolean retrievalExecution,
            String retrievalPolicyOwnedBy,
            String vectorStoreOwnedBy,
            boolean storageOwnedByOrchestrator) {
        private static EmbeddingRagDetails empty() {
            return new EmbeddingRagDetails(false, false, null, null, false);
        }
    }

    public static final class Builder {
        private final List<Issue> issues = new ArrayList<>();
        private final List<String> checkedAreas = new ArrayList<>();
        private final Map<String, Map<String, Object>> checkDetails = new LinkedHashMap<>();
        private String featureProfile;
        private String surface;
        private String model;

        private Builder() {
        }

        public Builder capabilities(AgentCapabilitiesView capabilities) {
            checked(AREA_CAPABILITIES);
            if (capabilities == null) {
                return blocking(AREA_CAPABILITIES, "capabilities response is missing");
            }
            for (String issue : capabilities.agentServingCapabilityIssues()) {
                blocking(AREA_CAPABILITIES, issue);
            }
            return this;
        }

        public Builder contract(AgentServingContract contract) {
            checked(AREA_CONTRACT);
            if (contract == null) {
                return blocking(AREA_CONTRACT, "serving contract response is missing");
            }
            for (String issue : contract.servingBoundaryIssues()) {
                blocking(AREA_CONTRACT, issue);
            }
            return this;
        }

        public Builder featureNegotiation(
                AgentFeatureNegotiation negotiation,
                String requiredContractVersion,
                List<String> requiredFeatures,
                List<String> optionalFeatures) {
            return featureNegotiation(
                    negotiation,
                    AgentServingFeatureProfile.DEFAULT_PROFILE,
                    requiredContractVersion,
                    requiredFeatures,
                    optionalFeatures);
        }

        public Builder featureNegotiation(
                AgentFeatureNegotiation negotiation,
                String featureProfile,
                String requiredContractVersion,
                List<String> requiredFeatures,
                List<String> optionalFeatures) {
            checked(AREA_FEATURE_NEGOTIATION);
            if (negotiation == null) {
                return blocking(AREA_FEATURE_NEGOTIATION, "feature negotiation metadata is missing");
            }
            String profileName = AgentServingFeatureProfile.normalizeName(featureProfile);
            Optional<AgentServingFeatureProfile> profile = AgentServingFeatureProfile.find(profileName);
            this.featureProfile = profileName;
            String contractVersion = requiredContractVersion == null || requiredContractVersion.isBlank()
                    ? AgentServingFeatureCatalog.CONTRACT_VERSION
                    : requiredContractVersion.trim();
            List<String> required = requiredFeatures == null || requiredFeatures.isEmpty()
                    ? profile.map(AgentServingFeatureProfile::requiredFeatures)
                            .orElse(AgentServingFeatureCatalog.REQUIRED_FEATURES)
                    : List.copyOf(requiredFeatures);
            List<String> optional = optionalFeatures == null || optionalFeatures.isEmpty()
                    ? profile.map(AgentServingFeatureProfile::optionalFeatures).orElse(List.of())
                    : List.copyOf(optionalFeatures);
            List<String> unsupportedRequired = negotiation.unsupportedFeatures(required);
            List<String> unsupportedOptional = negotiation.unsupportedFeatures(optional);
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("feature_profile", profileName);
            details.put("supported_feature_profiles", AgentServingFeatureProfile.supportedProfileNames());
            details.put("feature_profile_supported", profile.isPresent());
            details.put("required_contract_version", contractVersion);
            details.put("supported_contract_versions", negotiation.supportedContractVersions());
            details.put("required_features", required);
            details.put("optional_features", optional);
            details.put("available_features", negotiation.allFeatures());
            details.put("unsupported_required_features", unsupportedRequired);
            details.put("unsupported_optional_features", unsupportedOptional);
            details(AREA_FEATURE_NEGOTIATION, details);
            if (profile.isEmpty()) {
                blocking(AREA_FEATURE_NEGOTIATION, "feature profile is not supported: " + profileName);
            }
            if (!negotiation.supportsContractVersion(contractVersion)) {
                blocking(AREA_FEATURE_NEGOTIATION,
                        "required contract version is not supported: " + contractVersion);
            }
            for (String feature : unsupportedRequired) {
                blocking(AREA_FEATURE_NEGOTIATION, "required agent feature is not supported: " + feature);
            }
            for (String feature : unsupportedOptional) {
                warning(AREA_FEATURE_NEGOTIATION, "optional agent feature is not supported: " + feature);
            }
            return this;
        }

        public Builder featureProfile(String featureProfile) {
            this.featureProfile = normalizeFeatureProfile(featureProfile);
            return this;
        }

        public Builder modelRoute(AgentModelCapabilitiesView modelRoute) {
            return modelRoute(null, modelRoute);
        }

        public Builder modelRoute(String surface, AgentModelCapabilitiesView modelRoute) {
            checked(AREA_MODEL_ROUTE);
            if (modelRoute == null) {
                return blocking(AREA_MODEL_ROUTE, "model capability response is missing");
            }
            putIfPresentRoute(surface, modelRoute.modelId());
            for (String issue : modelRoute.agentServingRouteIssues(surface)) {
                blocking(AREA_MODEL_ROUTE, issue);
            }
            return this;
        }

        public Builder mcpDiscovery(AgentMcpDiscoveryView discovery) {
            return mcpDiscovery(discovery, false);
        }

        public Builder mcpDiscovery(AgentMcpDiscoveryView discovery, boolean required) {
            if (discovery == null) {
                return required
                        ? checked(AREA_MCP_DISCOVERY)
                                .blocking(AREA_MCP_DISCOVERY, "MCP discovery response is missing")
                        : this;
            }
            checked(AREA_MCP_DISCOVERY);
            if (!discovery.available()) {
                return required
                        ? blocking(AREA_MCP_DISCOVERY, "MCP discovery is not available")
                        : warning(AREA_MCP_DISCOVERY, "MCP discovery is not available");
            }
            if (!discovery.discoveryOnly()) {
                blocking(AREA_MCP_DISCOVERY, "MCP discovery endpoint crossed the serving boundary");
            }
            if (discovery.toolExecutionEnabled()) {
                blocking(AREA_MCP_DISCOVERY, "MCP discovery must not enable tool execution");
            }
            for (AgentMcpDiscoveryView.Tool tool : discovery.tools()) {
                if (tool.executionEnabled()) {
                    blocking(AREA_MCP_DISCOVERY,
                            "MCP tool must not be executable from Gollek: " + toolName(tool));
                }
            }
            return this;
        }

        public Builder toolValidation(AgentToolValidationView validation) {
            return toolValidation(validation, false);
        }

        public Builder toolValidation(AgentToolValidationView validation, boolean required) {
            if (validation == null) {
                return required
                        ? checked(AREA_TOOL_VALIDATION)
                                .blocking(AREA_TOOL_VALIDATION, "tool validation response is missing")
                        : this;
            }
            checked(AREA_TOOL_VALIDATION);
            if (!validation.valid()) {
                blocking(AREA_TOOL_VALIDATION, "tool definitions are not valid");
            }
            if (validation.modelInvoked()) {
                blocking(AREA_TOOL_VALIDATION, "tool validation must not invoke a model");
            }
            if (!validation.validationOnly()) {
                blocking(AREA_TOOL_VALIDATION, "tool validation must be validation-only");
            }
            if (validation.toolExecutionEnabled()) {
                blocking(AREA_TOOL_VALIDATION, "tool validation must not enable tool execution");
            }
            if (validation.toolAuthorizationEnabled()) {
                blocking(AREA_TOOL_VALIDATION, "tool validation must not authorize tools");
            }
            for (AgentToolValidationView.Warning warning : validation.warnings()) {
                warning(
                        AREA_TOOL_VALIDATION,
                        warningMessage(warning),
                        AgentReadinessIssueCodes.toolSchemaWarningCode(warning.code()));
            }
            return this;
        }

        public Builder requestValidation(AgentValidationView validation) {
            return requestValidation(validation, false);
        }

        public Builder requestValidation(AgentValidationView validation, boolean required) {
            if (validation == null) {
                return required
                        ? checked(AREA_REQUEST_VALIDATION)
                                .blocking(AREA_REQUEST_VALIDATION, "request validation response is missing")
                        : this;
            }
            checked(AREA_REQUEST_VALIDATION);
            Map<String, Object> details = requestValidationDetails(validation);
            details(AREA_REQUEST_VALIDATION, details);
            putIfPresentRoute(text(details.get("surface")), text(details.get("model")));
            if (!validation.valid()) {
                blocking(AREA_REQUEST_VALIDATION, "agent request is not valid");
            }
            if (validation.modelInvoked()) {
                blocking(AREA_REQUEST_VALIDATION, "request validation must not invoke a model");
            }
            if (!validation.validationOnly()) {
                blocking(AREA_REQUEST_VALIDATION, "request validation must be validation-only");
            }
            if (validation.toolExecutionEnabled()) {
                blocking(AREA_REQUEST_VALIDATION, "request validation must not enable tool execution");
            }
            if (validation.retrievalExecutionEnabled()) {
                blocking(AREA_REQUEST_VALIDATION, "request validation must not enable retrieval execution");
            }
            if (!validation.toolContractValid()) {
                blocking(AREA_REQUEST_VALIDATION, "embedded tool contract is not valid");
            }
            return this;
        }

        public Builder blocking(String area, String message) {
            return issue(area, Severity.BLOCKING, message);
        }

        public Builder warning(String area, String message) {
            return issue(area, Severity.WARNING, message);
        }

        public Builder warning(String area, String message, String code) {
            return issue(area, Severity.WARNING, message, code);
        }

        public Builder issue(String area, Severity severity, String message) {
            return issue(area, severity, message, null);
        }

        public Builder route(String surface, String model) {
            putIfPresentRoute(surface, model);
            return this;
        }

        public Builder route(AgentServingRoute route) {
            if (route == null) {
                return this;
            }
            putIfPresentRoute(route.surface(), route.model());
            if (route.featureProfile() != null) {
                this.featureProfile = route.featureProfile();
            }
            return this;
        }

        public Builder issue(String area, Severity severity, String message, String code) {
            Issue issue = new Issue(area, severity, message, code);
            if (!issues.contains(issue)) {
                issues.add(issue);
            }
            return this;
        }

        public AgentServingReadinessReport build() {
            Map<String, Check> checks = new LinkedHashMap<>();
            for (String area : checkedAreas) {
                List<String> blocking = messagesFor(issues, area, Severity.BLOCKING);
                List<String> warnings = messagesFor(issues, area, Severity.WARNING);
                checks.put(area, new Check(
                        area,
                        blocking.isEmpty() ? "ready" : "blocked",
                        blocking.isEmpty(),
                        true,
                        blocking,
                        warnings,
                        checkDetails.getOrDefault(area, Map.of())));
            }
            return new AgentServingReadinessReport(issues, checkedAreas, checks, null, featureProfile, surface, model);
        }

        private Builder checked(String area) {
            if (area != null && !area.isBlank() && !checkedAreas.contains(area)) {
                checkedAreas.add(area);
            }
            return this;
        }

        private Builder details(String area, Map<String, Object> details) {
            if (area == null || area.isBlank() || details == null || details.isEmpty()) {
                return this;
            }
            Map<String, Object> merged = new LinkedHashMap<>(checkDetails.getOrDefault(area, Map.of()));
            merged.putAll(details);
            checkDetails.put(area, Collections.unmodifiableMap(merged));
            return this;
        }

        private void putIfPresentRoute(String surface, String model) {
            String normalizedSurface = AgentServingRoute.normalizeSurface(surface);
            if (normalizedSurface != null) {
                this.surface = normalizedSurface;
            }
            String normalizedModel = text(model);
            if (normalizedModel != null) {
                this.model = normalizedModel;
            }
        }
    }

    private static List<String> messages(List<Issue> issues) {
        return issues.stream().map(Issue::message).toList();
    }

    private static List<String> codes(List<Issue> issues) {
        return issues.stream().map(Issue::code).toList();
    }

    private static Map<String, List<String>> codesByArea(List<Issue> issues) {
        if (issues == null || issues.isEmpty()) {
            return Map.of();
        }
        Map<String, List<String>> grouped = new LinkedHashMap<>();
        for (Issue issue : issues) {
            grouped.computeIfAbsent(issue.area(), ignored -> new ArrayList<>())
                    .add(issue.code());
        }
        Map<String, List<String>> out = new LinkedHashMap<>();
        grouped.forEach((area, codes) -> out.put(area, List.copyOf(codes)));
        return Collections.unmodifiableMap(out);
    }

    private static List<IssueHint> hints(List<Issue> issues) {
        if (issues == null || issues.isEmpty()) {
            return List.of();
        }
        List<IssueHint> out = new ArrayList<>();
        for (Issue issue : issues) {
            IssueHint hint = IssueHint.from(issue);
            if (hint != null) {
                out.add(hint);
            }
        }
        return List.copyOf(out);
    }

    private static List<IssueHint> hints(
            String area,
            Severity severity,
            List<String> messages,
            List<String> codes) {
        List<String> safeMessages = messages == null ? List.of() : messages;
        List<String> safeCodes = codes == null ? List.of() : codes;
        List<IssueHint> out = new ArrayList<>();
        for (int i = 0; i < safeMessages.size(); i++) {
            out.add(new IssueHint(area, severity, safeMessages.get(i), itemAt(safeCodes, i), null, null, null));
        }
        return List.copyOf(out);
    }

    private static List<IssueHint> issueHints(Object value, String fallbackArea) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        String defaultArea = fallbackArea == null || fallbackArea.isBlank() ? "preflight" : fallbackArea;
        List<IssueHint> out = new ArrayList<>();
        for (Object item : list) {
            Map<String, Object> hint = objectMap(item);
            if (hint.isEmpty()) {
                continue;
            }
            String code = text(hint.get("code"));
            String message = text(hint.get("message"));
            if (message == null && code == null) {
                continue;
            }
            if (message == null) {
                message = AgentReadinessIssueCodes.describe(code).summary();
            }
            out.add(new IssueHint(
                    textOrDefault(hint.get("area"), defaultArea),
                    severity(hint.getOrDefault("severity", hint.get("default_severity"))),
                    message,
                    code,
                    text(hint.get("default_severity")),
                    text(hint.get("summary")),
                    text(hint.get("remediation"))));
        }
        return List.copyOf(out);
    }

    private static List<IssueHint> responseIssueHints(Map<String, Object> response) {
        if (response == null || response.isEmpty()) {
            return List.of();
        }
        List<IssueHint> out = new ArrayList<>();
        for (IssueHint hint : issueHints(response.get("issue_hints"), "preflight")) {
            addIssueHint(out, hint);
        }
        Map<String, Object> readinessReport = objectMap(response.get("readiness_report"));
        for (IssueHint hint : issueHints(readinessReport.get("issue_hints"), "preflight")) {
            addIssueHint(out, hint);
        }
        return List.copyOf(out);
    }

    private static List<IssueHint> hintsForArea(List<IssueHint> hints, String area) {
        if (hints == null || hints.isEmpty()) {
            return List.of();
        }
        String normalizedArea = area == null || area.isBlank() ? "general" : area;
        List<IssueHint> out = new ArrayList<>();
        for (IssueHint hint : hints) {
            if (normalizedArea.equals(hint.area())) {
                addIssueHint(out, hint);
            }
        }
        return List.copyOf(out);
    }

    private static void addIssueHint(List<IssueHint> out, IssueHint hint) {
        if (out == null || hint == null) {
            return;
        }
        for (IssueHint existing : out) {
            if (sameIssueHintIdentity(existing, hint)) {
                return;
            }
        }
        out.add(hint);
    }

    private static boolean sameIssueHintIdentity(IssueHint left, IssueHint right) {
        return left.area().equals(right.area())
                && left.severity() == right.severity()
                && left.message().equals(right.message())
                && left.code().equals(right.code());
    }

    private static Map<String, List<IssueHint>> hintsByArea(List<Issue> issues) {
        if (issues == null || issues.isEmpty()) {
            return Map.of();
        }
        Map<String, List<IssueHint>> grouped = new LinkedHashMap<>();
        for (Issue issue : issues) {
            grouped.computeIfAbsent(issue.area(), ignored -> new ArrayList<>())
                    .add(issue.toHint());
        }
        Map<String, List<IssueHint>> out = new LinkedHashMap<>();
        grouped.forEach((area, areaHints) -> out.put(area, List.copyOf(areaHints)));
        return Collections.unmodifiableMap(out);
    }

    private static Map<String, List<IssueHint>> hintsByCode(List<IssueHint> hints) {
        if (hints == null || hints.isEmpty()) {
            return Map.of();
        }
        Map<String, List<IssueHint>> grouped = new LinkedHashMap<>();
        for (IssueHint hint : hints) {
            grouped.computeIfAbsent(hint.code(), ignored -> new ArrayList<>())
                    .add(hint);
        }
        Map<String, List<IssueHint>> out = new LinkedHashMap<>();
        grouped.forEach((code, codeHints) -> out.put(code, List.copyOf(codeHints)));
        return Collections.unmodifiableMap(out);
    }

    private static List<IssueHint> hintsForCode(List<IssueHint> hints, String code) {
        String normalized = AgentReadinessIssueCodes.normalize(code);
        if (normalized == null || hints == null || hints.isEmpty()) {
            return List.of();
        }
        List<IssueHint> out = new ArrayList<>();
        for (IssueHint hint : hints) {
            if (normalized.equals(hint.code())) {
                out.add(hint);
            }
        }
        return List.copyOf(out);
    }

    private static Optional<IssueHint> firstHintForCode(List<IssueHint> hints, String code) {
        List<IssueHint> matching = hintsForCode(hints, code);
        return matching.isEmpty() ? Optional.empty() : Optional.of(matching.get(0));
    }

    private static boolean containsCode(List<String> codes, String code) {
        String normalized = AgentReadinessIssueCodes.normalize(code);
        return normalized != null && codes != null && codes.stream().anyMatch(normalized::equals);
    }

    private static List<Issue> parseIssues(Map<String, Object> response, List<IssueHint> responseHints) {
        List<Issue> issues = new ArrayList<>();
        Object rawIssues = response.get("issues");
        if (rawIssues instanceof List<?> list) {
            for (Object item : list) {
                if (item instanceof Map<?, ?> map) {
                    String message = text(map.get("message"));
                    if (message != null) {
                        issues.add(new Issue(
                                textOrDefault(map.get("area"), "preflight"),
                                severity(map.get("severity")),
                                message,
                                text(map.get("code"))));
                    }
                }
            }
        }
        if (!issues.isEmpty()) {
            return issues;
        }

        Object rawChecks = rawChecks(response);
        if (rawChecks instanceof Map<?, ?> checks) {
            for (Map.Entry<?, ?> entry : checks.entrySet()) {
                if (!(entry.getValue() instanceof Map<?, ?> check)) {
                    continue;
                }
                String area = textOrDefault(entry.getKey(), "preflight");
                List<String> blockingMessages = stringList(check.get("blocking_messages"));
                List<String> blockingCodes = stringList(check.get("blocking_codes"));
                for (int i = 0; i < blockingMessages.size(); i++) {
                    issues.add(new Issue(
                            area,
                            Severity.BLOCKING,
                            blockingMessages.get(i),
                            itemAt(blockingCodes, i)));
                }
                List<String> warningMessages = stringList(check.get("warning_messages"));
                List<String> warningCodes = stringList(check.get("warning_codes"));
                for (int i = 0; i < warningMessages.size(); i++) {
                    issues.add(new Issue(
                            area,
                            Severity.WARNING,
                            warningMessages.get(i),
                            itemAt(warningCodes, i)));
                }
            }
        }
        if (!issues.isEmpty()) {
            return issues;
        }

        if (responseHints != null && !responseHints.isEmpty()) {
            for (IssueHint hint : responseHints) {
                issues.add(new Issue(hint.area(), hint.severity(), hint.message(), hint.code()));
            }
            return List.copyOf(issues);
        }

        List<String> warningMessages = stringList(response.get("warning_messages"));
        Object grouped = response.get("issues_by_area");
        if (grouped instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                String area = textOrDefault(entry.getKey(), "preflight");
                for (String message : stringList(entry.getValue())) {
                    Severity severity = warningMessages.contains(message) ? Severity.WARNING : Severity.BLOCKING;
                    issues.add(new Issue(area, severity, message));
                }
            }
        }
        if (!issues.isEmpty()) {
            return issues;
        }

        List<String> blockingMessages = stringList(response.get("blocking_messages"));
        List<String> blockingCodes = stringList(response.get("blocking_codes"));
        for (int i = 0; i < blockingMessages.size(); i++) {
            issues.add(new Issue(
                    "preflight",
                    Severity.BLOCKING,
                    blockingMessages.get(i),
                    itemAt(blockingCodes, i)));
        }
        List<String> warningCodes = stringList(response.get("warning_codes"));
        for (int i = 0; i < warningMessages.size(); i++) {
            issues.add(new Issue(
                    "preflight",
                    Severity.WARNING,
                    warningMessages.get(i),
                    itemAt(warningCodes, i)));
        }
        return issues;
    }

    private List<String> messagesFor(String area, Severity severity) {
        return messagesFor(issues, area, severity);
    }

    private Check checkFromIssues(String area) {
        List<String> blocking = messagesFor(area, Severity.BLOCKING);
        List<String> warnings = messagesFor(area, Severity.WARNING);
        return new Check(
                area,
                blocking.isEmpty() ? "ready" : "blocked",
                blocking.isEmpty(),
                true,
                blocking,
                warnings,
                List.of(),
                List.of(),
                Map.of(),
                hintsForArea(sourceIssueHints, area));
    }

    private List<IssueHint> checkIssueHints(Severity severity) {
        Map<String, Check> checks = checkResults();
        List<IssueHint> out = new ArrayList<>();
        for (IssueHint hint : sourceIssueHints) {
            if (severity == null || hint.severity() == severity) {
                addIssueHint(out, hint);
            }
        }
        if (checks.isEmpty()) {
            if (!out.isEmpty()) {
                return List.copyOf(out);
            }
            if (severity == Severity.BLOCKING) {
                return hints(blockingIssues());
            }
            if (severity == Severity.WARNING) {
                return hints(warnings());
            }
            return hints(issues);
        }
        for (Check check : checks.values()) {
            List<IssueHint> hints;
            if (severity == Severity.BLOCKING) {
                hints = check.blockingIssueHints();
            } else if (severity == Severity.WARNING) {
                hints = check.warningIssueHints();
            } else {
                hints = check.issueHints();
            }
            for (IssueHint hint : hints) {
                addIssueHint(out, hint);
            }
        }
        return List.copyOf(out);
    }

    private Map<String, List<IssueHint>> checkIssueHintsByArea() {
        List<IssueHint> hints = checkIssueHints(null);
        if (hints.isEmpty()) {
            return Map.of();
        }
        Map<String, List<IssueHint>> grouped = new LinkedHashMap<>();
        for (IssueHint hint : hints) {
            grouped.computeIfAbsent(hint.area(), ignored -> new ArrayList<>())
                    .add(hint);
        }
        Map<String, List<IssueHint>> out = new LinkedHashMap<>();
        grouped.forEach((area, areaHints) -> out.put(area, List.copyOf(areaHints)));
        return Collections.unmodifiableMap(out);
    }

    private static List<String> messagesFor(List<Issue> issues, String area, Severity severity) {
        return issues.stream()
                .filter(issue -> issue.area().equals(area))
                .filter(issue -> issue.severity() == severity)
                .map(Issue::message)
                .toList();
    }

    private static List<String> issueCodes(
            List<String> suppliedCodes,
            String area,
            Severity severity,
            List<String> messages) {
        List<String> safeMessages = messages == null ? List.of() : messages;
        List<String> safeCodes = suppliedCodes == null ? List.of() : suppliedCodes;
        List<String> out = new ArrayList<>();
        for (int i = 0; i < safeMessages.size(); i++) {
            out.add(AgentReadinessIssueCodes.resolve(itemAt(safeCodes, i), area, severity, safeMessages.get(i)));
        }
        return List.copyOf(out);
    }

    private static String itemAt(List<String> items, int index) {
        return items == null || index < 0 || index >= items.size() ? null : items.get(index);
    }

    private static Map<String, Check> parseChecks(Map<String, Object> response, List<IssueHint> responseHints) {
        Object rawChecks = rawChecks(response);
        if (!(rawChecks instanceof Map<?, ?> checks)) {
            return Map.of();
        }
        Map<String, Check> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : checks.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> check)) {
                continue;
            }
            String area = textOrDefault(entry.getKey(), "preflight");
            List<String> blocking = stringList(check.get("blocking_messages"));
            List<String> warnings = stringList(check.get("warning_messages"));
            List<String> blockingCodes = stringList(check.get("blocking_codes"));
            List<String> warningCodes = stringList(check.get("warning_codes"));
            String status = text(check.get("status"));
            Boolean ready = bool(check.get("ready"));
            if (status == null) {
                if (ready != null && !ready) {
                    status = "blocked";
                } else {
                    status = blocking.isEmpty() ? "ready" : "blocked";
                }
            }
            Boolean requested = bool(check.get("requested"));
            List<IssueHint> suppliedHints = issueHints(check.get("issue_hints"), area);
            if (suppliedHints.isEmpty()) {
                suppliedHints = hintsForArea(responseHints, area);
            }
            out.put(area, new Check(
                    area,
                    status,
                    ready == null ? blocking.isEmpty() : ready,
                    requested == null ? !"skipped".equals(status) : requested,
                    blocking,
                    warnings,
                    blockingCodes,
                    warningCodes,
                    checkDetails(check),
                    suppliedHints));
        }
        return Collections.unmodifiableMap(out);
    }

    private static Map<String, Object> requestValidationDetails(AgentValidationView validation) {
        Map<String, Object> details = new LinkedHashMap<>();
        putIfPresent(details, "surface", validation.surface());
        putIfPresent(details, "model", validation.model());
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", validation.validationOnly());
        boundary.put("model_invoked", validation.modelInvoked());
        boundary.put("tool_execution", validation.toolExecutionEnabled());
        boundary.put("retrieval_execution", validation.retrievalExecutionEnabled());
        details.put("boundary", boundary);
        if (validation.embeddingSurface()) {
            details.put("embedding", embeddingValidationDetails(validation.embeddingValidation()));
        } else {
            Map<String, Object> request = new LinkedHashMap<>();
            request.put("streaming", validation.streaming());
            request.put("message_count", validation.messageCount());
            request.put("input_count", validation.inputCount());
            request.put("tool_count", validation.toolCount());
            request.put("parameter_keys", validation.parameterKeys());
            Map<String, Object> rag = new LinkedHashMap<>();
            rag.put("injected", validation.ragContextInjected());
            rag.put("items", validation.ragContextItems());
            putIfPresent(rag, "alias", validation.ragContextAlias());
            request.put("rag", rag);
            request.put("tool_contract", Map.of(
                    "valid", validation.toolContractValid(),
                    "warning_count", validation.toolContractWarningCount()));
            details.put("request", request);
        }
        return Collections.unmodifiableMap(details);
    }

    private static Map<String, Object> embeddingValidationDetails(
            AgentValidationView.EmbeddingValidation validation) {
        Map<String, Object> embedding = new LinkedHashMap<>();
        embedding.put("input_count", validation.inputCount());
        embedding.put("input_lengths", validation.inputLengths());
        putIfPresent(embedding, "requested_dimensions", validation.requestedDimensions());
        putIfPresent(embedding, "encoding_format", validation.encodingFormat());
        embedding.put("parameter_keys", validation.parameterKeys());
        embedding.put("metadata_keys", validation.metadata().keySet().stream()
                .map(Object::toString)
                .sorted()
                .toList());
        Map<String, Object> rag = new LinkedHashMap<>();
        rag.put("embedding_generation", validation.rag().embeddingGeneration());
        rag.put("retrieval_execution", validation.rag().retrievalExecution());
        putIfPresent(rag, "retrieval_policy_owned_by", validation.rag().retrievalPolicyOwnedBy());
        putIfPresent(rag, "vector_store_owned_by", validation.rag().vectorStoreOwnedBy());
        rag.put("storage_owned_by_orchestrator", validation.storageOwnedByOrchestrator());
        embedding.put("rag", rag);
        return Collections.unmodifiableMap(embedding);
    }

    private static Map<String, Object> checkDetails(Map<?, ?> check) {
        Map<String, Object> details = objectMap(check.get("details"));
        return details.isEmpty() ? objectMap(check.get("metadata")) : details;
    }

    private static List<String> checkedAreas(Map<String, Object> response, List<Issue> parsed) {
        List<String> explicit = explicitCheckedAreas(response);
        if (!explicit.isEmpty()) {
            return explicit;
        }
        Object rawChecks = rawChecks(response);
        if (rawChecks instanceof Map<?, ?> checks) {
            List<String> areas = new ArrayList<>();
            for (Map.Entry<?, ?> entry : checks.entrySet()) {
                if (entry.getValue() instanceof Map<?, ?>) {
                    String area = text(entry.getKey());
                    if (area != null && !areas.contains(area)) {
                        areas.add(area);
                    }
                }
            }
            if (!areas.isEmpty()) {
                return List.copyOf(areas);
            }
        }
        return areasFromIssues(parsed);
    }

    private static Object rawChecks(Map<String, Object> response) {
        Object checkResults = response.get("check_results");
        if (checkResults != null) {
            return checkResults;
        }
        Object topLevelChecks = response.get("checks");
        if (hasCheckMaps(topLevelChecks)) {
            return topLevelChecks;
        }
        Object readinessChecks = objectMap(response.get("readiness_report")).get("checks");
        return readinessChecks == null ? topLevelChecks : readinessChecks;
    }

    private static List<String> explicitCheckedAreas(Map<String, Object> response) {
        List<String> topLevel = uniqueStringList(response.get("checked_areas"));
        if (!topLevel.isEmpty()) {
            return topLevel;
        }
        return uniqueStringList(objectMap(response.get("readiness_report")).get("checked_areas"));
    }

    private static boolean hasCheckMaps(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return false;
        }
        for (Object item : map.values()) {
            if (item instanceof Map<?, ?>) {
                return true;
            }
        }
        return false;
    }

    private static List<String> areasFromIssues(List<Issue> issues) {
        List<String> areas = new ArrayList<>();
        if (issues != null) {
            for (Issue issue : issues) {
                if (!areas.contains(issue.area())) {
                    areas.add(issue.area());
                }
            }
        }
        return List.copyOf(areas);
    }

    private static List<String> uniqueStringList(Object value) {
        List<String> out = new ArrayList<>();
        for (String item : stringList(value)) {
            if (!out.contains(item)) {
                out.add(item);
            }
        }
        return List.copyOf(out);
    }

    private static boolean blocked(Map<String, Object> response) {
        Object ready = response.get("ready");
        if (ready instanceof Boolean bool) {
            return !bool;
        }
        Object status = response.get("status");
        return status != null && "blocked".equalsIgnoreCase(String.valueOf(status));
    }

    private static String featureProfile(Map<String, Object> response, Map<String, Check> checks) {
        String topLevel = normalizeFeatureProfile(text(response.get("feature_profile")));
        if (topLevel != null) {
            return topLevel;
        }
        String nested = normalizeFeatureProfile(text(objectMap(response.get("readiness_report")).get("feature_profile")));
        if (nested != null) {
            return nested;
        }
        Check featureNegotiation = checks == null ? null : checks.get(AREA_FEATURE_NEGOTIATION);
        if (featureNegotiation == null) {
            return null;
        }
        return normalizeFeatureProfile(text(featureNegotiation.details().get("feature_profile")));
    }

    private static String routeSurface(Map<String, Object> response, Map<String, Check> checks) {
        String topLevel = AgentServingRoute.normalizeSurface(text(response.get("surface")));
        if (topLevel != null) {
            return topLevel;
        }
        String nested = AgentServingRoute.normalizeSurface(text(objectMap(response.get("readiness_report")).get("surface")));
        if (nested != null) {
            return nested;
        }
        Check requestValidation = checks == null ? null : checks.get(AREA_REQUEST_VALIDATION);
        if (requestValidation != null) {
            return AgentServingRoute.normalizeSurface(requestValidation.surface());
        }
        return null;
    }

    private static String routeModel(Map<String, Object> response, Map<String, Check> checks) {
        String topLevel = text(response.get("model"));
        if (topLevel != null) {
            return topLevel;
        }
        String nested = text(objectMap(response.get("readiness_report")).get("model"));
        if (nested != null) {
            return nested;
        }
        Check requestValidation = checks == null ? null : checks.get(AREA_REQUEST_VALIDATION);
        if (requestValidation != null && requestValidation.model() != null) {
            return requestValidation.model();
        }
        Check modelRoute = checks == null ? null : checks.get(AREA_MODEL_ROUTE);
        return modelRoute == null ? null : modelRoute.model();
    }

    private static String normalizeFeatureProfile(String featureProfile) {
        if (featureProfile == null || featureProfile.isBlank()) {
            return null;
        }
        return AgentServingFeatureProfile.normalizeName(featureProfile);
    }

    private static Severity severity(Object value) {
        String normalized = text(value);
        return "warning".equalsIgnoreCase(normalized) ? Severity.WARNING : Severity.BLOCKING;
    }

    private static void putIfPresent(Map<String, Object> map, String key, Object value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    private static RequestRagDetails requestRagDetails(Object value) {
        Map<String, Object> rag = objectMap(value);
        if (rag.isEmpty()) {
            return RequestRagDetails.empty();
        }
        return new RequestRagDetails(
                boolOrDefault(rag.get("injected"), false),
                intOrDefault(rag.get("items"), 0),
                text(rag.get("alias")));
    }

    private static ToolContractDetails toolContractDetails(Object value) {
        Map<String, Object> toolContract = objectMap(value);
        if (toolContract.isEmpty()) {
            return ToolContractDetails.empty();
        }
        return new ToolContractDetails(
                boolOrDefault(toolContract.get("valid"), false),
                intOrDefault(toolContract.get("warning_count"), 0));
    }

    private static EmbeddingRagDetails embeddingRagDetails(Object value) {
        Map<String, Object> rag = objectMap(value);
        if (rag.isEmpty()) {
            return EmbeddingRagDetails.empty();
        }
        return new EmbeddingRagDetails(
                boolOrDefault(rag.get("embedding_generation"), false),
                boolOrDefault(rag.get("retrieval_execution"), false),
                text(rag.get("retrieval_policy_owned_by")),
                text(rag.get("vector_store_owned_by")),
                boolOrDefault(rag.get("storage_owned_by_orchestrator"), false));
    }

    private static boolean boolOrDefault(Object value, boolean fallback) {
        Boolean bool = bool(value);
        return bool == null ? fallback : bool;
    }

    private static Boolean bool(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        String text = text(value);
        if (text == null) {
            return null;
        }
        if ("true".equalsIgnoreCase(text)) {
            return true;
        }
        if ("false".equalsIgnoreCase(text)) {
            return false;
        }
        return null;
    }

    private static int intOrDefault(Object value, int fallback) {
        Integer parsed = intValue(value);
        return parsed == null ? fallback : parsed;
    }

    private static Integer intValue(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        String text = text(value);
        if (text == null) {
            return null;
        }
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static List<Integer> intList(Object value) {
        if (!(value instanceof List<?> list)) {
            Integer parsed = intValue(value);
            return parsed == null ? List.of() : List.of(parsed);
        }
        List<Integer> out = new ArrayList<>();
        for (Object item : list) {
            Integer parsed = intValue(item);
            if (parsed != null) {
                out.add(parsed);
            }
        }
        return List.copyOf(out);
    }

    private static List<String> stringList(Object value) {
        if (value instanceof List<?> list) {
            List<String> out = new ArrayList<>();
            for (Object item : list) {
                String text = text(item);
                if (text != null) {
                    out.add(text);
                }
            }
            return List.copyOf(out);
        }
        String text = text(value);
        return text == null ? List.of() : List.of(text);
    }

    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            String key = text(entry.getKey());
            if (key != null) {
                out.put(key, entry.getValue());
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static String textOrDefault(Object value, String fallback) {
        String text = text(value);
        return text == null ? fallback : text;
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    private static String toolName(AgentMcpDiscoveryView.Tool tool) {
        if (tool == null) {
            return "unknown";
        }
        if (tool.name() != null) {
            return tool.name();
        }
        if (tool.mcpToolName() != null) {
            return tool.mcpToolName();
        }
        return "unknown";
    }

    private static String warningMessage(AgentToolValidationView.Warning warning) {
        StringBuilder message = new StringBuilder("tool schema warning");
        if (warning.code() != null) {
            message.append(" (").append(warning.code()).append(")");
        }
        if (warning.path() != null) {
            message.append(" at ").append(warning.path());
        }
        if (warning.message() != null) {
            message.append(": ").append(warning.message());
        }
        return message.toString();
    }
}
