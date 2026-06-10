package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.client.agent.AgentReadinessIssueCodes;
import tech.kayys.gollek.client.agent.AgentReadinessMetadata;
import tech.kayys.gollek.client.agent.AgentCapabilitiesView;
import tech.kayys.gollek.client.agent.AgentIntegrationClient;
import tech.kayys.gollek.client.agent.AgentIntegrationException;
import tech.kayys.gollek.client.agent.AgentMcpDiscoveryView;
import tech.kayys.gollek.client.agent.AgentModelCapabilitiesView;
import tech.kayys.gollek.client.agent.AgentReadinessIssueCatalogView;
import tech.kayys.gollek.client.agent.AgentRequestOptions;
import tech.kayys.gollek.client.agent.AgentServingContract;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;
import tech.kayys.gollek.client.agent.AgentServingPreflightGate;
import tech.kayys.gollek.client.agent.AgentServingPreflightPolicy;
import tech.kayys.gollek.client.agent.AgentServingPreflightRequest;
import tech.kayys.gollek.client.agent.AgentServingPreflightResult;
import tech.kayys.gollek.client.agent.AgentServingReadinessReport;
import tech.kayys.gollek.client.agent.AgentServingRoute;
import tech.kayys.gollek.client.agent.AgentToolValidationView;
import tech.kayys.gollek.client.agent.AgentValidationView;

import java.io.IOException;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Dependent
@Unremovable
@Command(
        name = "agent",
        description = "Agent integration helpers for Gollek serving APIs",
        subcommands = {
                AgentCommand.Capabilities.class,
                AgentCommand.Contract.class,
                AgentCommand.IssueCatalog.class,
                AgentCommand.ModelCapabilities.class,
                AgentCommand.ToolsValidate.class,
                AgentTemplateCheckCommand.class,
                AgentTemplateSchemaCommand.class,
                AgentTemplateExampleCommand.class,
                AgentCommand.RequestValidate.class,
                AgentCommand.McpTools.class,
                AgentCommand.Readiness.class,
                AgentCommand.Preflight.class
        })
public class AgentCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Override
    public void run() {
        System.out.println("Use one of: gollek agent capabilities | contract | issue-catalog | model-capabilities | tools-validate | template-check | template-schema | template-example | request-validate | mcp-tools | readiness | preflight");
    }

    GollekCommand root() {
        return parentCommand;
    }

    private static void printCheckDetails(Map<String, Object> check) {
        Map<String, Object> details = detailMap(check.get("details"));
        if (details.isEmpty()) {
            return;
        }
        List<String> summary = new ArrayList<>();
        addPart(summary, "surface", details.get("surface"));
        addPart(summary, "model", details.get("model"));
        if (!summary.isEmpty()) {
            System.out.println("  details: " + String.join(", ", summary));
        }
        Map<String, Object> boundary = detailMap(details.get("boundary"));
        if (!boundary.isEmpty()) {
            System.out.println("  boundary: validation_only=" + detailText(boundary.get("validation_only"))
                    + ", model_invoked=" + detailText(boundary.get("model_invoked"))
                    + ", tool_execution=" + detailText(boundary.get("tool_execution"))
                    + ", retrieval_execution=" + detailText(boundary.get("retrieval_execution")));
        }
        Map<String, Object> request = detailMap(details.get("request"));
        if (!request.isEmpty()) {
            System.out.println("  request: streaming=" + detailText(request.get("streaming"))
                    + ", messages=" + detailText(request.get("message_count"))
                    + ", inputs=" + detailText(request.get("input_count"))
                    + ", tools=" + detailText(request.get("tool_count"))
                    + ", parameters=" + detailValues(request.get("parameter_keys")));
            Map<String, Object> rag = detailMap(request.get("rag"));
            if (!rag.isEmpty()) {
                System.out.println("  rag: injected=" + detailText(rag.get("injected"))
                        + ", items=" + detailText(rag.get("items"))
                        + ", alias=" + detailText(rag.get("alias")));
            }
        }
        Map<String, Object> embedding = detailMap(details.get("embedding"));
        if (!embedding.isEmpty()) {
            System.out.println("  embeddings: inputs=" + detailText(embedding.get("input_count"))
                    + ", input_lengths=" + detailValues(embedding.get("input_lengths"))
                    + ", requested_dimensions=" + detailText(embedding.get("requested_dimensions"))
                    + ", encoding=" + detailText(embedding.get("encoding_format"))
                    + ", metadata_keys=" + detailValues(embedding.get("metadata_keys")));
            Map<String, Object> rag = detailMap(embedding.get("rag"));
            if (!rag.isEmpty()) {
                System.out.println("  embedding boundary: generation=" + detailText(rag.get("embedding_generation"))
                        + ", retrieval_execution=" + detailText(rag.get("retrieval_execution"))
                        + ", retrieval_policy_owner=" + detailText(rag.get("retrieval_policy_owned_by"))
                        + ", vector_store_owner=" + detailText(rag.get("vector_store_owned_by")));
            }
        }
    }

    private static void printCheckMessages(String area, String label, Map<String, Object> check) {
        String severity = "warning".equals(label) ? "warning" : "blocking";
        printMessageLines(
                "  " + label + ": ",
                area,
                severity,
                check.get(label + "_messages"),
                check.get(label + "_codes"),
                check.get("issue_hints"));
    }

    private static void printMessagesWithCodes(String label, Object rawMessages, Object rawCodes, String severity) {
        printMessagesWithCodes(label, rawMessages, rawCodes, severity, null);
    }

    private static void printMessagesWithCodes(
            String label,
            Object rawMessages,
            Object rawCodes,
            String severity,
            Object rawHints) {
        List<String> messages = detailStringList(rawMessages);
        if (messages.isEmpty()) {
            return;
        }
        System.out.println(label + ":");
        printMessageLines("- ", null, severity, messages, rawCodes, rawHints);
    }

    private static void printMessageLines(
            String prefix,
            String area,
            String severity,
            Object rawMessages,
            Object rawCodes) {
        printMessageLines(prefix, area, severity, rawMessages, rawCodes, null);
    }

    private static void printMessageLines(
            String prefix,
            String area,
            String severity,
            Object rawMessages,
            Object rawCodes,
            Object rawHints) {
        List<String> messages = detailStringList(rawMessages);
        List<String> codes = detailStringList(rawCodes);
        for (int i = 0; i < messages.size(); i++) {
            String message = messages.get(i);
            String messageArea = area == null || area.isBlank() ? prefixedArea(message) : area;
            String messageBody = stripAreaPrefix(message);
            String code = AgentReadinessIssueCodes.resolve(itemAt(codes, i), messageArea, severity, messageBody);
            System.out.println(prefix + message + " [code: " + code + "]");
            String remediation = issueRemediation(code, rawHints);
            if (DiscoveryCommand.hasText(remediation)) {
                System.out.println(continuationPrefix(prefix) + "remediation: " + remediation);
            }
        }
    }

    private static String issueRemediation(String code, Object rawHints) {
        String normalizedCode = AgentReadinessIssueCodes.normalize(code);
        if (normalizedCode == null) {
            return null;
        }
        Map<String, Object> hint = issueHintsByCode(rawHints).get(normalizedCode);
        String remediation = hint == null ? null : textValue(hint.get("remediation"));
        if (DiscoveryCommand.hasText(remediation)) {
            return remediation;
        }
        if (AgentReadinessIssueCodes.find(normalizedCode).isPresent()
                || normalizedCode.startsWith("TOOL_SCHEMA_")) {
            return AgentReadinessIssueCodes.describe(normalizedCode).remediation();
        }
        return null;
    }

    private static List<Map<String, Object>> issueHintMaps(
            String area,
            List<String> blockingMessages,
            List<String> blockingCodes,
            List<String> warningMessages,
            List<String> warningCodes) {
        List<Map<String, Object>> out = new ArrayList<>();
        out.addAll(issueHintMaps(
                area,
                AgentServingReadinessReport.Severity.BLOCKING,
                blockingMessages,
                blockingCodes));
        out.addAll(issueHintMaps(
                area,
                AgentServingReadinessReport.Severity.WARNING,
                warningMessages,
                warningCodes));
        return List.copyOf(out);
    }

    private static List<Map<String, Object>> issueHintMaps(
            String area,
            AgentServingReadinessReport.Severity severity,
            List<String> messages,
            List<String> codes) {
        List<String> safeMessages = messages == null ? List.of() : messages;
        List<String> safeCodes = codes == null ? List.of() : codes;
        List<Map<String, Object>> out = new ArrayList<>();
        for (int i = 0; i < safeMessages.size(); i++) {
            out.add(new AgentServingReadinessReport.IssueHint(
                    area,
                    severity,
                    safeMessages.get(i),
                    itemAt(safeCodes, i),
                    null,
                    null,
                    null).toMap());
        }
        return List.copyOf(out);
    }

    private static Map<String, Map<String, Object>> issueHintsByCode(Object rawHints) {
        List<?> hints = rawHints instanceof List<?> list ? list : List.of();
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        for (Object item : hints) {
            Map<String, Object> hint = detailMap(item);
            String code = AgentReadinessIssueCodes.normalize(textValue(hint.get("code")));
            if (code != null) {
                out.putIfAbsent(code, hint);
            }
        }
        return out;
    }

    private static String textValue(Object value) {
        if (value == null) {
            return null;
        }
        String text = value.toString();
        return text == null || text.isBlank() ? null : text;
    }

    private static String continuationPrefix(String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return "  ";
        }
        int spaces = 0;
        while (spaces < prefix.length() && Character.isWhitespace(prefix.charAt(spaces))) {
            spaces++;
        }
        if (prefix.trim().startsWith("-")) {
            return "  ";
        }
        return " ".repeat(spaces + 2);
    }

    private static Map<String, Object> checkDetails(String id, Map<String, Object> response) {
        Map<String, Object> direct = detailMap(response == null ? null : response.get("details"));
        if (!direct.isEmpty()) {
            return direct;
        }
        if ("request_validation".equals(id)) {
            return requestValidationDetails(response);
        }
        if ("preflight".equals(id)) {
            Map<String, Object> checks = detailMap(response == null ? null : response.get("check_results"));
            if (checks.isEmpty()) {
                checks = detailMap(detailMap(response == null ? null : response.get("readiness_report")).get("checks"));
            }
            Map<String, Object> requestCheck = detailMap(checks.get("request_validation"));
            Map<String, Object> details = detailMap(requestCheck.get("details"));
            if (!details.isEmpty()) {
                return details;
            }
            Map<String, Object> requestValidation = detailMap(response == null ? null : response.get("request_validation"));
            return requestValidationDetails(requestValidation);
        }
        return Map.of();
    }

    private static Map<String, Object> requestValidationDetails(Map<String, Object> validation) {
        if (validation == null || validation.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> normalized = detailMap(validation.get("normalized"));
        Map<String, Object> details = new LinkedHashMap<>();
        putIfPresent(details, "surface", validation.get("surface"));
        putIfPresent(details, "model", normalized.get("model"));
        details.put("boundary", boundaryDetails(validation));
        if ("embeddings".equals(validation.get("surface"))) {
            details.put("embedding", embeddingDetails(normalized));
        } else if (!normalized.isEmpty()) {
            details.put("request", requestDetails(normalized));
        }
        return details;
    }

    private static Map<String, Object> boundaryDetails(Map<String, Object> validation) {
        Map<String, Object> raw = detailMap(validation.get("boundary"));
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("validation_only", raw.getOrDefault("validation_only", true));
        boundary.put("model_invoked", validation.getOrDefault("model_invoked", false));
        boundary.put("tool_execution", raw.getOrDefault("tool_execution", false));
        boundary.put("retrieval_execution", raw.getOrDefault("retrieval_execution", false));
        return boundary;
    }

    private static Map<String, Object> requestDetails(Map<String, Object> normalized) {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("streaming", normalized.getOrDefault("streaming", false));
        request.put("message_count", normalized.getOrDefault("message_count", 0));
        request.put("input_count", normalized.getOrDefault("input_count", 0));
        request.put("tool_count", normalized.getOrDefault("tool_count", 0));
        request.put("parameter_keys", detailList(normalized.get("parameter_keys")));
        Map<String, Object> rag = detailMap(normalized.get("rag"));
        Map<String, Object> ragDetails = new LinkedHashMap<>();
        ragDetails.put("injected", rag.getOrDefault("injected", false));
        ragDetails.put("items", rag.getOrDefault("items", 0));
        putIfPresent(ragDetails, "alias", rag.get("alias"));
        request.put("rag", ragDetails);
        return request;
    }

    private static Map<String, Object> embeddingDetails(Map<String, Object> normalized) {
        Map<String, Object> embedding = new LinkedHashMap<>();
        embedding.put("input_count", normalized.getOrDefault("input_count", 0));
        embedding.put("input_lengths", detailList(normalized.get("input_lengths")));
        putIfPresent(embedding, "requested_dimensions", normalized.get("requested_dimensions"));
        putIfPresent(embedding, "encoding_format", normalized.get("encoding_format"));
        embedding.put("metadata_keys", sortedKeys(detailMap(normalized.get("metadata"))));
        Map<String, Object> rag = detailMap(normalized.get("rag"));
        Map<String, Object> ragDetails = new LinkedHashMap<>();
        ragDetails.put("embedding_generation", rag.getOrDefault("embedding_generation", false));
        ragDetails.put("retrieval_execution", rag.getOrDefault("retrieval_execution", false));
        putIfPresent(ragDetails, "retrieval_policy_owned_by", rag.get("retrieval_policy_owned_by"));
        putIfPresent(ragDetails, "vector_store_owned_by", rag.get("vector_store_owned_by"));
        embedding.put("rag", ragDetails);
        return embedding;
    }

    private static void addPart(List<String> parts, String label, Object value) {
        if (value != null) {
            parts.add(label + "=" + detailText(value));
        }
    }

    private static void putIfPresent(Map<String, Object> map, String key, Object value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    private static String detailText(Object value) {
        return value == null ? "n/a" : value.toString();
    }

    private static String detailValues(Object value) {
        List<?> values = detailList(value);
        if (values.isEmpty()) {
            return "n/a";
        }
        return String.join(",", values.stream()
                .map(AgentCommand::detailText)
                .toList());
    }

    private static List<?> detailList(Object value) {
        return value instanceof List<?> list ? List.copyOf(list) : List.of();
    }

    private static List<String> detailStringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (Object item : list) {
            String text = item == null ? null : item.toString();
            if (text != null && !text.isBlank()) {
                out.add(text);
            }
        }
        return List.copyOf(out);
    }

    private static String itemAt(List<String> values, int index) {
        return values == null || index < 0 || index >= values.size() ? null : values.get(index);
    }

    private static List<String> issueCodes(String area, String severity, List<String> messages, Object rawCodes) {
        List<String> supplied = detailStringList(rawCodes);
        List<String> out = new ArrayList<>();
        for (int i = 0; i < messages.size(); i++) {
            out.add(AgentReadinessIssueCodes.resolve(itemAt(supplied, i), area, severity, messages.get(i)));
        }
        return List.copyOf(out);
    }

    private static String prefixedArea(String message) {
        if (message == null) {
            return null;
        }
        int separator = message.indexOf(": ");
        return separator <= 0 ? null : message.substring(0, separator).trim();
    }

    private static String stripAreaPrefix(String message) {
        if (message == null) {
            return null;
        }
        int separator = message.indexOf(": ");
        return separator <= 0 ? message : message.substring(separator + 2);
    }

    private static List<String> sortedKeys(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return List.of();
        }
        return map.keySet().stream().sorted().toList();
    }

    private static Map<String, Object> detailMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getKey() != null) {
                out.put(entry.getKey().toString(), entry.getValue());
            }
        }
        return out;
    }

    abstract static class DiscoveryCommand implements Callable<Integer> {
        static final ObjectMapper DISCOVERY_MAPPER = new ObjectMapper();
        static final String DEFAULT_BASE_URL = "http://localhost:8080";
        static final String DEFAULT_API_KEY = "community";

        @ParentCommand
        AgentCommand parentCommand;

        @Option(names = { "--base-url" }, description = "Gollek serving base URL")
        String baseUrl;

        @Option(names = { "--api-key" }, description = "Gollek API key")
        String apiKey;

        @Option(names = { "--request-id" }, description = "Trace request id sent as X-Gollek-Request-Id")
        String requestId;

        @Option(names = { "--trace-id" }, description = "Trace id sent as X-Gollek-Trace-Id")
        String traceId;

        @Option(names = { "--session-id" }, description = "Session id sent as X-Gollek-Session-Id")
        String sessionId;

        @Option(names = { "--user-id" }, description = "User id sent as X-Gollek-User-Id")
        String userId;

        @Option(names = { "--header" }, split = ",", description = "Extra header as Name:Value, repeatable or comma-separated")
        List<String> headers = new ArrayList<>();

        @Option(names = { "--json" }, description = "Print the raw discovery response as JSON")
        boolean json;

        @Option(names = { "--require-serving-boundary" }, description = "Exit 1 when the response does not satisfy Gollek's serving-boundary contract")
        boolean requireServingBoundary;

        @Override
        public Integer call() {
            try {
                if (parentCommand != null && parentCommand.root() != null) {
                    parentCommand.root().bootstrapInheritedEnvironment();
                }
                Map<String, Object> response = fetch(client(), requestOptions());
                List<String> issues = servingBoundaryIssues(response);
                if (json) {
                    System.out.println(DISCOVERY_MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(response));
                } else {
                    printHumanReport(response, issues);
                }
                return shouldFail(response, issues) ? 1 : 0;
            } catch (AgentRouteConsistency.ConflictException e) {
                if (json) {
                    printJsonConfigError(configErrorObject(), e.result());
                } else {
                    printError(commandLabel() + " failed: " + e.getMessage());
                }
                return 2;
            } catch (AgentIntegrationException e) {
                printError(commandLabel() + " request failed: " + e.getMessage());
                return integrationFailureExitCode(e);
            } catch (Exception e) {
                printError(commandLabel() + " failed: " + e.getMessage());
                return 2;
            }
        }

        abstract String commandLabel();

        abstract Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException;

        abstract void printHumanReport(Map<String, Object> response, List<String> issues);

        abstract List<String> servingBoundaryIssues(Map<String, Object> response);

        String configErrorObject() {
            return "gollek.agent_cli_error";
        }

        boolean shouldFail(Map<String, Object> response, List<String> issues) {
            return requireServingBoundary && !issues.isEmpty();
        }

        int integrationFailureExitCode(AgentIntegrationException e) {
            return 2;
        }

        AgentIntegrationClient client() {
            return new AgentIntegrationClient(
                    HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(),
                    DISCOVERY_MAPPER,
                    effectiveBaseUrl(),
                    effectiveApiKey());
        }

        AgentRequestOptions requestOptions() {
            AgentRequestOptions.Builder builder = AgentRequestOptions.builder()
                    .requestId(requestId)
                    .traceId(traceId)
                    .sessionId(sessionId)
                    .userId(userId);
            if (headers != null) {
                for (String header : headers) {
                    if (!hasText(header)) {
                        continue;
                    }
                    int separator = header.indexOf(':');
                    if (separator <= 0 || separator == header.length() - 1) {
                        throw new IllegalArgumentException("--header must use Name:Value format.");
                    }
                    builder.header(header.substring(0, separator).trim(), header.substring(separator + 1).trim());
                }
            }
            return builder.build();
        }

        String effectiveBaseUrl() {
            return firstNonBlank(
                    baseUrl,
                    System.getProperty("gollek.api.base-url"),
                    System.getProperty("gollek.server.base-url"),
                    System.getenv("GOLLEK_BASE_URL"),
                    System.getenv("GOLLEK_API_URL"),
                    System.getenv("GOLLEK_SERVER_URL"),
                    DEFAULT_BASE_URL);
        }

        String effectiveApiKey() {
            return firstNonBlank(
                    apiKey,
                    System.getProperty("gollek.api.key"),
                    System.getenv("GOLLEK_API_KEY"),
                    DEFAULT_API_KEY);
        }

        String endpoint(String path) {
            return normalizedBaseUrl(effectiveBaseUrl()) + path;
        }

        static void printIssues(String label, List<String> issues) {
            if (issues == null || issues.isEmpty()) {
                return;
            }
            System.out.println(label + ":");
            for (String issue : issues) {
                System.out.println("- " + issue);
            }
        }

        static String yesNo(boolean value) {
            return value ? "yes" : "no";
        }

        static String text(String value) {
            return value == null || value.isBlank() ? "n/a" : value;
        }

        static String text(Object value) {
            return value == null ? "n/a" : text(value.toString());
        }

        static String normalizedBaseUrl(String value) {
            String out = firstNonBlank(value, DEFAULT_BASE_URL);
            while (out.endsWith("/")) {
                out = out.substring(0, out.length() - 1);
            }
            return out;
        }

        static String firstNonBlank(String... values) {
            if (values == null) {
                return null;
            }
            for (String value : values) {
                if (hasText(value)) {
                    return value.trim();
                }
            }
            return null;
        }

        static boolean hasText(String value) {
            return value != null && !value.isBlank();
        }

        static void printError(String message) {
            System.err.println(message);
        }

        static void printJsonConfigError(String object, AgentRouteConsistency.Result result) {
            try {
                System.out.println(DISCOVERY_MAPPER.writerWithDefaultPrettyPrinter()
                        .writeValueAsString(result.toMetadata(object)));
            } catch (IOException ignored) {
                printError(result.errorMessage());
            }
        }

        @SuppressWarnings("unchecked")
        static Map<String, Object> toMap(JsonNode node) {
            return DISCOVERY_MAPPER.convertValue(node, new TypeReference<LinkedHashMap<String, Object>>() {
            });
        }
    }

    @Command(
            name = "capabilities",
            description = "Show Gollek's agent-facing serving capabilities")
    public static class Capabilities extends DiscoveryCommand {
        @Override
        String commandLabel() {
            return "Agent capabilities";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            return client.capabilities(options);
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentCapabilitiesView capabilities = AgentCapabilitiesView.from(response, DISCOVERY_MAPPER);
            System.out.println("Gollek agent capabilities");
            System.out.println("Endpoint: " + endpoint("/v1/agent/capabilities"));
            System.out.println("Service role: " + text(capabilities.serviceRole()));
            System.out.println("OpenAI surfaces: chat=" + yesNo(capabilities.supportsOpenAiChat())
                    + ", responses=" + yesNo(capabilities.supportsOpenAiResponses())
                    + ", embeddings=" + yesNo(capabilities.supportsEmbeddings()));
            System.out.println("Agent surfaces: model_capabilities=" + yesNo(capabilities.supportsModelCapabilities())
                    + ", issue_catalog=" + yesNo(capabilities.supportsReadinessIssueCatalog())
                    + ", mcp_tool_discovery=" + yesNo(capabilities.supportsMcpToolDiscovery())
                    + ", rag_context=" + yesNo(capabilities.supportsRagContext()));
            System.out.println("Auth: api_key_header=" + text(capabilities.apiKeyHeader())
                    + ", authorization=" + text(capabilities.authorizationHeaderScheme()));
            System.out.println("Trace: request_id=" + text(capabilities.requestIdHeader())
                    + ", trace_id=" + text(capabilities.traceIdHeader())
                    + ", metadata=" + text(capabilities.traceMetadataKey()));
            printIssues("Serving capability issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            return AgentCapabilitiesView.from(response, DISCOVERY_MAPPER).agentServingCapabilityIssues();
        }
    }

    @Command(
            name = "contract",
            description = "Show Gollek's machine-readable agent serving contract")
    public static class Contract extends DiscoveryCommand {
        @Override
        String commandLabel() {
            return "Agent contract";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            return client.contract(options);
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentServingContract contract = AgentServingContract.from(response, DISCOVERY_MAPPER);
            System.out.println("Gollek agent contract");
            System.out.println("Endpoint: " + endpoint("/v1/agent/contract"));
            System.out.println("Version: " + text(contract.version()));
            System.out.println("Service role: " + text(contract.serviceRole()));
            System.out.println("Boundary: tool_execution=" + yesNo(contract.toolExecutionEnabled())
                    + ", retrieval_execution=" + yesNo(contract.retrievalExecutionEnabled())
                    + ", streaming=" + yesNo(contract.supportsStreaming()));
            System.out.println("Endpoints: chat=" + text(contract.endpointPath("chat_completions"))
                    + ", responses=" + text(contract.endpointPath("responses"))
                    + ", embeddings=" + text(contract.endpointPath("embeddings")));
            System.out.println("Agent endpoints: capabilities=" + text(contract.endpointPath("agent_capabilities"))
                    + ", issue_catalog=" + text(contract.endpointPath("agent_readiness_issues"))
                    + ", preflight=" + text(contract.endpointPath("agent_preflight"))
                    + ", validation=" + text(contract.endpointPath("agent_validation"))
                    + ", tool_validation=" + text(contract.endpointPath("agent_tool_validation")));
            printIssues("Serving contract issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            return AgentServingContract.from(response, DISCOVERY_MAPPER).servingBoundaryIssues();
        }
    }

    @Command(
            name = "issue-catalog",
            aliases = { "issues", "readiness-issues" },
            description = "Show stable readiness issue codes and remediation metadata")
    public static class IssueCatalog extends DiscoveryCommand {
        @Option(names = { "--area" }, description = "Show only one readiness area, such as tool_validation")
        String area;

        @Option(names = { "--code" }, description = "Show only one readiness issue code")
        String code;

        @Override
        String commandLabel() {
            return "Agent readiness issue catalog";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            return client.readinessIssueCatalog(options);
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentReadinessIssueCatalogView catalog = AgentReadinessIssueCatalogView.from(response, DISCOVERY_MAPPER);
            List<AgentReadinessIssueCodes.CatalogEntry> entries = filteredEntries(catalog);
            System.out.println("Gollek agent readiness issue catalog");
            System.out.println("Endpoint: " + endpoint("/v1/agent/readiness/issues"));
            System.out.println("Version: " + text(catalog.version()));
            System.out.println("Service role: " + text(catalog.serviceRole()));
            System.out.println("Entries: total=" + catalog.count() + ", shown=" + entries.size());
            System.out.println("Boundary: validation_only=" + yesNo(catalog.validationOnly())
                    + ", model_invoked=" + yesNo(catalog.modelInvoked())
                    + ", tool_execution=" + yesNo(catalog.toolExecutionEnabled())
                    + ", retrieval_execution=" + yesNo(catalog.retrievalExecutionEnabled()));
            if (!catalog.byArea().isEmpty()) {
                System.out.println("Areas: " + areaSummary(catalog));
            }
            for (AgentReadinessIssueCodes.CatalogEntry entry : entries) {
                System.out.println("- " + entry.code()
                        + " [" + entry.area() + ", " + entry.defaultSeverity() + "]: "
                        + text(entry.summary()));
                System.out.println("  remediation: " + text(entry.remediation()));
            }
            printIssues("Readiness issue catalog boundary issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            return AgentReadinessIssueCatalogView.from(response, DISCOVERY_MAPPER).servingBoundaryIssues();
        }

        private List<AgentReadinessIssueCodes.CatalogEntry> filteredEntries(
                AgentReadinessIssueCatalogView catalog) {
            List<AgentReadinessIssueCodes.CatalogEntry> entries = hasText(area)
                    ? catalog.entriesForArea(area)
                    : catalog.entries();
            if (!hasText(code)) {
                return entries;
            }
            String normalizedCode = AgentReadinessIssueCodes.normalize(code);
            if (normalizedCode == null) {
                return List.of();
            }
            return entries.stream()
                    .filter(entry -> normalizedCode.equals(entry.code()))
                    .toList();
        }

        private static String areaSummary(AgentReadinessIssueCatalogView catalog) {
            List<String> parts = new ArrayList<>();
            for (Map.Entry<String, List<AgentReadinessIssueCodes.CatalogEntry>> entry : catalog.byArea().entrySet()) {
                parts.add(entry.getKey() + "=" + entry.getValue().size());
            }
            return parts.isEmpty() ? "n/a" : String.join(", ", parts);
        }
    }

    @Command(
            name = "model-capabilities",
            aliases = { "model", "model-route" },
            description = "Show one model's agent-serving route capabilities")
    public static class ModelCapabilities extends DiscoveryCommand {
        @Option(names = { "-m", "--model" }, required = true, description = "Model id to inspect")
        String modelId;

        @Override
        String commandLabel() {
            return "Agent model capabilities";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            return client.modelCapabilities(modelId, options);
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentModelCapabilitiesView model = AgentModelCapabilitiesView.from(response, DISCOVERY_MAPPER);
            AgentModelCapabilitiesView.Limits limits = model.limits();
            AgentModelCapabilitiesView.EmbeddingCapabilities embeddings = model.embeddings();
            System.out.println("Gollek agent model capabilities");
            System.out.println("Endpoint: " + endpoint("/v1/models/" + modelId + "/capabilities"));
            System.out.println("Model: " + text(model.modelId()));
            System.out.println("Known: " + yesNo(model.known()) + ", available: " + yesNo(model.available()));
            System.out.println("Provider: " + text(model.preferredProvider())
                    + ", format=" + text(model.format())
                    + ", architecture=" + text(model.architecture()));
            System.out.println("OpenAI surfaces: chat=" + yesNo(model.supportsChatCompletions())
                    + ", chat_streaming=" + yesNo(model.supportsChatStreaming())
                    + ", responses=" + yesNo(model.supportsResponses())
                    + ", responses_streaming=" + yesNo(model.supportsResponsesStreaming())
                    + ", embeddings=" + yesNo(model.supportsEmbeddings()));
            System.out.println("Inference: completion=" + yesNo(model.supportsCompletion())
                    + ", streaming=" + yesNo(model.supportsStreaming())
                    + ", system_prompt=" + yesNo(model.supportsSystemPrompt())
                    + ", json_mode=" + yesNo(model.supportsJsonMode())
                    + ", structured_outputs=" + yesNo(model.supportsStructuredOutputs()));
            System.out.println("Tools: definitions=" + yesNo(model.supportsToolDefinitions())
                    + ", choice=" + yesNo(model.supportsToolChoice())
                    + ", model_tool_calling=" + yesNo(model.supportsModelToolCalling())
                    + ", mcp_definitions=" + yesNo(model.supportsMcpToolDefinitions())
                    + ", execution=" + yesNo(model.toolExecutionEnabled()));
            System.out.println("RAG: context_injection=" + yesNo(model.supportsRagContextInjection())
                    + ", retrieval_policy_in_gollek=" + yesNo(model.retrievalPolicyOwnedByGollek())
                    + ", vector_store_in_gollek=" + yesNo(model.vectorStoreOwnedByGollek()));
            System.out.println("Embeddings: generation=" + yesNo(embeddings.generation())
                    + ", endpoint=" + text(embeddings.endpoint())
                    + ", dimensions=" + text(numberText(embeddings.dimensions()))
                    + ", encodings=" + String.join(",", embeddings.encodingFormats())
                    + ", vector_store_in_gollek=" + yesNo(embeddings.vectorStoreOwnedByGollek()));
            System.out.println("Limits: context=" + text(numberText(limits.contextTokens()))
                    + ", input=" + text(numberText(limits.inputTokens()))
                    + ", output=" + text(numberText(limits.outputTokens()))
                    + ", embedding_dimensions=" + text(numberText(limits.embeddingDimensions())));
            System.out.println("Provider candidates: " + model.providerCandidates().size());
            printIssues("Agent-serving route issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            return AgentModelCapabilitiesView.from(response, DISCOVERY_MAPPER).agentServingRouteIssues();
        }

        private static String numberText(Number value) {
            return value == null ? null : value.toString();
        }
    }

    @Command(
            name = "tools-validate",
            aliases = { "validate-tools", "tool-contract" },
            description = "Validate OpenAI/MCP tool definitions without executing tools")
    public static class ToolsValidate extends DiscoveryCommand {
        @Option(names = { "--tool-json", "--tools-json" }, description = "Inline JSON tool object, array, or object with tools array")
        String toolsJson;

        @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s)")
        List<String> toolFiles = new ArrayList<>();

        @Option(names = { "--request-json" }, description = "Inline OpenAI-compatible request JSON containing tools")
        String requestJson;

        @Option(names = { "--request-file" }, description = "Path to OpenAI-compatible request JSON containing tools")
        String requestFile;

        @Option(names = { "--require-clean" }, description = "Exit 1 when validation returns portability warnings")
        boolean requireClean;

        @Override
        String commandLabel() {
            return "Agent tool validation";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            try {
                return client.validateTools(loadPayload(), options);
            } catch (IOException e) {
                throw new IllegalArgumentException(e.getMessage(), e);
            }
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentToolValidationView validation = AgentToolValidationView.from(response, DISCOVERY_MAPPER);
            System.out.println("Gollek agent tool validation: " + (validation.valid() ? "valid" : "invalid"));
            System.out.println("Endpoint: " + endpoint("/v1/agent/tools/validate"));
            System.out.println("Tools: count=" + validation.toolCount()
                    + ", names=" + toolNames(validation.toolNames()));
            System.out.println("Boundary: validation_only=" + yesNo(validation.validationOnly())
                    + ", model_invoked=" + yesNo(validation.modelInvoked())
                    + ", tool_execution=" + yesNo(validation.toolExecutionEnabled())
                    + ", tool_authorization=" + yesNo(validation.toolAuthorizationEnabled()));
            System.out.println("Warnings: " + validation.warnings().size());
            printWarnings(validation.warnings());
            printIssues("Tool validation issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            AgentToolValidationView validation = AgentToolValidationView.from(response, DISCOVERY_MAPPER);
            List<String> issues = new ArrayList<>();
            if (!validation.valid()) {
                issues.add("tool contract validation response is not valid");
            }
            if (!validation.validationOnly()) {
                issues.add("tool validation must be validation-only");
            }
            if (validation.modelInvoked()) {
                issues.add("tool validation must not invoke a model");
            }
            if (validation.toolExecutionEnabled()) {
                issues.add("tool validation must not execute tools");
            }
            if (validation.toolAuthorizationEnabled()) {
                issues.add("tool validation must not authorize tools");
            }
            return List.copyOf(issues);
        }

        @Override
        boolean shouldFail(Map<String, Object> response, List<String> issues) {
            AgentToolValidationView validation = AgentToolValidationView.from(response, DISCOVERY_MAPPER);
            return !validation.valid()
                    || (requireClean && validation.hasWarnings())
                    || (requireServingBoundary && !issues.isEmpty());
        }

        @Override
        int integrationFailureExitCode(AgentIntegrationException e) {
            String message = e.getMessage();
            return message != null && message.contains("status 400") ? 1 : 2;
        }

        private Map<String, Object> loadPayload() throws IOException {
            if (hasText(requestJson) && hasText(requestFile)) {
                throw new IllegalArgumentException("Use either --request-json or --request-file, not both.");
            }

            List<Object> tools = new ArrayList<>();
            if (hasText(requestJson)) {
                tools.addAll(parseTools(requestJson.trim(), "request JSON"));
            }
            if (hasText(requestFile)) {
                Path path = Path.of(requestFile.trim());
                tools.addAll(parseTools(Files.readString(path), path.toString()));
            }
            if (hasText(toolsJson)) {
                tools.addAll(parseTools(toolsJson.trim(), "tool JSON"));
            }
            if (toolFiles != null) {
                for (String file : toolFiles) {
                    if (!hasText(file)) {
                        continue;
                    }
                    Path path = Path.of(file.trim());
                    tools.addAll(parseTools(Files.readString(path), path.toString()));
                }
            }
            if (tools.isEmpty()) {
                throw new IllegalArgumentException(
                        "Provide --tool-json, --tool-file, --request-json, or --request-file with at least one tool.");
            }
            return Map.of("tools", tools);
        }

        private List<Object> parseTools(String jsonText, String source) throws IOException {
            JsonNode root = DISCOVERY_MAPPER.readTree(jsonText);
            JsonNode tools = root.path("tools").isArray() ? root.path("tools") : root;
            List<Object> out = new ArrayList<>();
            if (tools.isArray()) {
                for (JsonNode tool : tools) {
                    out.add(DISCOVERY_MAPPER.convertValue(tool, Object.class));
                }
            } else if (tools.isObject()) {
                out.add(toMap(tools));
            } else {
                throw new IllegalArgumentException(
                        "Tool input must be a JSON object, array, or object with tools array: " + source);
            }
            return out;
        }

        private static String toolNames(List<String> names) {
            return names == null || names.isEmpty() ? "n/a" : String.join(", ", names);
        }

        private static void printWarnings(List<AgentToolValidationView.Warning> warnings) {
            if (warnings == null || warnings.isEmpty()) {
                return;
            }
            System.out.println("Tool validation warnings:");
            for (AgentToolValidationView.Warning warning : warnings) {
                String location = warning.index() >= 0 ? "tools[" + warning.index() + "]" : "tools";
                String path = warning.path() == null ? "" : "." + warning.path();
                String code = warning.code() == null ? "warning" : warning.code();
                System.out.println("- " + location + path + " " + code + ": " + text(warning.message()));
            }
        }
    }

    @Command(
            name = "request-validate",
            aliases = { "validate-request", "dry-run" },
            description = "Validate an agent request without invoking a model")
    public static class RequestValidate extends DiscoveryCommand {
        private static final String DEFAULT_SURFACE = "chat";
        private static final String DEFAULT_PROMPT = "Validate this agent request.";
        private static final String DEFAULT_SYSTEM_PROMPT = "Use supplied context and tools when relevant.";

        @Option(names = { "--surface" }, description = "Compatibility surface: chat, responses, or embeddings")
        String surface;

        @Option(names = { "-m", "--model" }, description = "Model id to validate")
        String modelId;

        @Option(names = { "--feature-profile" }, description = "Serving feature profile used for route-template consistency")
        String featureProfile;

        @Option(names = { "-p", "--prompt", "--input" }, description = "Prompt/input used when no request JSON is provided")
        String prompt;

        @Option(names = { "--system" }, description = "System prompt used when building a chat request")
        String systemPrompt;

        @Option(names = { "--request-json" }, description = "Inline OpenAI-compatible request JSON")
        String requestJson;

        @Option(names = { "--request-file" }, description = "Path to OpenAI-compatible request JSON")
        String requestFile;

        @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s)")
        List<String> toolFiles = new ArrayList<>();

        @Option(names = { "--rag-context" }, split = "\\|", description = "Inline RAG context block(s), separated with |")
        List<String> ragContexts = new ArrayList<>();

        @Option(names = { "--rag-file", "--context-file" }, split = ",", description = "Text file(s) to include as caller-owned RAG context")
        List<String> ragFiles = new ArrayList<>();

        @Option(names = { "--require-clean" }, description = "Exit 1 when nested tool-contract validation returns warnings")
        boolean requireClean;

        private String effectiveSurface;

        @Override
        String commandLabel() {
            return "Agent request validation";
        }

        @Override
        String configErrorObject() {
            return "gollek.agent_request_validation_error";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            try {
                Map<String, Object> request = loadRequest();
                return client.validateRequest(effectiveSurface, request, options);
            } catch (IOException e) {
                throw new IllegalArgumentException(e.getMessage(), e);
            }
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentValidationView validation = AgentValidationView.from(response, DISCOVERY_MAPPER);
            System.out.println("Gollek agent request validation: " + (validation.valid() ? "valid" : "invalid"));
            System.out.println("Endpoint: " + endpoint("/v1/agent/validate?surface=" + text(validation.surface())));
            System.out.println("Surface: " + text(validation.surface()));
            System.out.println("Model: " + text(validation.model()));
            if (validation.embeddingSurface()) {
                AgentValidationView.EmbeddingValidation embedding = validation.embeddingValidation();
                System.out.println("Embeddings: inputs=" + embedding.inputCount()
                        + ", input_lengths=" + values(embedding.inputLengths())
                        + ", requested_dimensions=" + text(embedding.requestedDimensions())
                        + ", encoding=" + text(embedding.encodingFormat())
                        + ", metadata_keys=" + values(embedding.metadata().keySet().stream()
                                .map(Object::toString)
                                .sorted()
                                .toList()));
                System.out.println("Embedding boundary: generation=" + yesNo(embedding.rag().embeddingGeneration())
                        + ", retrieval_execution=" + yesNo(embedding.rag().retrievalExecution())
                        + ", retrieval_policy_owner=" + text(embedding.rag().retrievalPolicyOwnedBy())
                        + ", vector_store_owner=" + text(embedding.rag().vectorStoreOwnedBy()));
            } else {
                System.out.println("Request: streaming=" + yesNo(validation.streaming())
                        + ", messages=" + validation.messageCount()
                        + ", inputs=" + validation.inputCount()
                        + ", tools=" + validation.toolCount());
                System.out.println("RAG: injected=" + yesNo(validation.ragContextInjected())
                        + ", items=" + validation.ragContextItems()
                        + ", alias=" + text(validation.ragContextAlias()));
                System.out.println("Tool contract: valid=" + yesNo(validation.toolContractValid())
                        + ", warnings=" + validation.toolContractWarningCount());
            }
            System.out.println("Boundary: validation_only=" + yesNo(validation.validationOnly())
                    + ", model_invoked=" + yesNo(validation.modelInvoked())
                    + ", tool_execution=" + yesNo(validation.toolExecutionEnabled())
                    + ", retrieval_execution=" + yesNo(validation.retrievalExecutionEnabled()));
            printIssues("Request validation issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            AgentValidationView validation = AgentValidationView.from(response, DISCOVERY_MAPPER);
            List<String> issues = new ArrayList<>();
            if (!validation.valid()) {
                issues.add("request validation response is not valid");
            }
            if (!validation.validationOnly()) {
                issues.add("request validation must be validation-only");
            }
            if (validation.modelInvoked()) {
                issues.add("request validation must not invoke a model");
            }
            if (validation.toolExecutionEnabled()) {
                issues.add("request validation must not execute tools");
            }
            if (validation.retrievalExecutionEnabled()) {
                issues.add("request validation must not execute retrieval");
            }
            if (!validation.toolContractValid()) {
                issues.add("nested tool contract validation is not valid");
            }
            return List.copyOf(issues);
        }

        @Override
        boolean shouldFail(Map<String, Object> response, List<String> issues) {
            AgentValidationView validation = AgentValidationView.from(response, DISCOVERY_MAPPER);
            return !validation.valid()
                    || !validation.toolContractValid()
                    || (requireClean && validation.toolContractWarningCount() > 0)
                    || (requireServingBoundary && !issues.isEmpty());
        }

        @Override
        int integrationFailureExitCode(AgentIntegrationException e) {
            String message = e.getMessage();
            return message != null && message.contains("status 400") ? 1 : 2;
        }

        private Map<String, Object> loadRequest() throws IOException {
            AgentRouteInput input = loadInput();
            Map<String, Object> request = new LinkedHashMap<>(input.request());
            String effectiveModel = firstNonBlank(modelId, input.modelId(), stringValue(request.get("model")));
            String effectiveFeatureProfile = AgentServingFeatureProfile.normalizeName(
                    firstNonBlank(featureProfile, input.featureProfile()));
            effectiveSurface = AgentRequestShape.resolveSurface(
                    DEFAULT_SURFACE,
                    surface,
                    input.surface(),
                    stringValue(request.get("surface")),
                    request,
                    effectiveFeatureProfile);
            AgentRouteConsistency.Result routeConsistency = AgentRouteConsistency.validateAndNormalize(
                    input,
                    request,
                    new AgentRouteConsistency.Overrides(
                            hasText(modelId),
                            hasText(surface),
                            hasText(featureProfile)),
                    new AgentRouteConsistency.Route(
                            effectiveModel,
                            effectiveSurface,
                            effectiveFeatureProfile));
            if (!routeConsistency.ok()) {
                throw routeConsistency.toException();
            }
            if (hasText(effectiveModel)) {
                request.put("model", effectiveModel);
            }
            AgentRequestPayloads.appendToolFiles(request, toolFiles, DISCOVERY_MAPPER);
            AgentRequestPayloads.appendRagContext(request, ragContexts, ragFiles);
            return request;
        }

        private AgentRouteInput loadInput() throws IOException {
            if (hasText(requestJson) && hasText(requestFile)) {
                throw new IllegalArgumentException("Use either --request-json or --request-file, not both.");
            }
            if (hasText(requestJson)) {
                return AgentRouteInput.parse(requestJson.trim(), DISCOVERY_MAPPER, "Request validation input");
            }
            if (hasText(requestFile)) {
                return AgentRouteInput.parse(
                        Files.readString(Path.of(requestFile.trim())),
                        DISCOVERY_MAPPER,
                        "Request validation input");
            }
            return AgentRouteInput.generated(defaultRequest());
        }

        private Map<String, Object> defaultRequest() {
            Map<String, Object> request = AgentRequestShape.generatedRequest(
                    surface,
                    featureProfile,
                    prompt,
                    DEFAULT_PROMPT,
                    systemPrompt,
                    DEFAULT_SYSTEM_PROMPT);
            if (hasText(modelId)) {
                request.put("model", modelId.trim());
            }
            return request;
        }

        private static String stringValue(Object value) {
            return value == null ? null : value.toString();
        }

        private static String values(Iterable<?> values) {
            if (values == null) {
                return "n/a";
            }
            List<String> out = new ArrayList<>();
            for (Object value : values) {
                if (value != null) {
                    out.add(value.toString());
                }
            }
            return out.isEmpty() ? "n/a" : String.join(",", out);
        }
    }

    @Command(
            name = "mcp-tools",
            aliases = { "mcp", "discover-tools" },
            description = "Discover MCP tool definitions without executing tools")
    public static class McpTools extends DiscoveryCommand {
        @Option(names = { "--server" }, description = "Limit discovery to one MCP server")
        String serverName;

        @Option(names = { "--raw-mcp-tools" }, description = "Return raw MCP tool definitions instead of OpenAI-compatible function tools")
        boolean rawMcpTools;

        @Option(names = { "--include-disabled-mcp" }, description = "Include disabled MCP server registrations")
        boolean includeDisabledMcp;

        @Option(names = { "--require-tools" }, description = "Exit 1 when discovery returns no tools")
        boolean requireTools;

        @Override
        String commandLabel() {
            return "Agent MCP tool discovery";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            boolean openAiCompat = !rawMcpTools;
            if (hasText(serverName)) {
                return client.mcpServerTools(serverName, openAiCompat, options);
            }
            return client.mcpTools(openAiCompat, !includeDisabledMcp, options);
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            AgentMcpDiscoveryView discovery = AgentMcpDiscoveryView.from(response, DISCOVERY_MAPPER);
            System.out.println("Gollek agent MCP tool discovery");
            System.out.println("Endpoint: " + endpoint(endpointPath()));
            System.out.println("Available: " + yesNo(discovery.available())
                    + ", compat=" + text(discovery.compatibility())
                    + ", enabled_only=" + yesNo(discovery.enabledOnly()));
            if (hasText(discovery.serverName())) {
                System.out.println("Server: " + discovery.serverName());
            }
            System.out.println("Boundary: discovery_only=" + yesNo(discovery.discoveryOnly())
                    + ", tool_execution=" + yesNo(discovery.toolExecutionEnabled()));
            System.out.println("Tools: count=" + discovery.tools().size()
                    + ", openai_compatible=" + discovery.openAiToolDefinitions().size()
                    + ", names=" + names(discovery.toolNames()));
            if (!discovery.serverNames().isEmpty()) {
                System.out.println("Servers: " + names(discovery.serverNames()));
            }
            if (hasText(discovery.message())) {
                System.out.println("Message: " + discovery.message());
            }
            printIssues("MCP discovery issues", issues);
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            AgentMcpDiscoveryView discovery = AgentMcpDiscoveryView.from(response, DISCOVERY_MAPPER);
            List<String> issues = new ArrayList<>();
            if (!discovery.available()) {
                issues.add("MCP registry is unavailable");
            }
            if (!discovery.discoveryOnly()) {
                issues.add("MCP discovery must be discovery-only");
            }
            if (discovery.toolExecutionEnabled()) {
                issues.add("MCP discovery must not execute tools");
            }
            if (requireTools && discovery.tools().isEmpty()) {
                issues.add("MCP discovery returned no tools");
            }
            return List.copyOf(issues);
        }

        @Override
        boolean shouldFail(Map<String, Object> response, List<String> issues) {
            AgentMcpDiscoveryView discovery = AgentMcpDiscoveryView.from(response, DISCOVERY_MAPPER);
            return (requireTools && discovery.tools().isEmpty())
                    || (requireServingBoundary && !issues.isEmpty());
        }

        private String endpointPath() {
            String compat = rawMcpTools ? "mcp" : "openai";
            if (hasText(serverName)) {
                return "/v1/mcp/servers/" + serverName + "/tools?compat=" + compat;
            }
            return "/v1/mcp/tools?compat=" + compat + "&enabledOnly=" + !includeDisabledMcp;
        }

        private static String names(List<String> values) {
            return values == null || values.isEmpty() ? "n/a" : String.join(", ", values);
        }
    }

    @Command(
            name = "readiness",
            aliases = { "check", "ci" },
            description = "Run the agent-serving readiness checklist for CI")
    public static class Readiness extends DiscoveryCommand {
        private static final String DEFAULT_SURFACE = "chat";
        private static final String DEFAULT_PROMPT = "Can this route support my agent request?";
        private static final String DEFAULT_SYSTEM_PROMPT = "Use supplied context and tools when relevant.";

        @Option(names = { "-m", "--model" }, description = "Model id to validate")
        String modelId;

        @Option(names = { "--surface" }, description = "Compatibility surface: chat, responses, or embeddings")
        String surface;

        @Option(names = { "--feature-profile" }, description = "Serving feature profile: agent_serving, chat_agent, responses_agent, embedding_rag, or mcp_tools")
        String featureProfile;

        @Option(names = { "-p", "--prompt", "--input" }, description = "Prompt/input used when no request JSON is provided")
        String prompt;

        @Option(names = { "--system" }, description = "System prompt used when building a chat request")
        String systemPrompt;

        @Option(names = { "--request-json" }, description = "Inline OpenAI-compatible request JSON, or a full preflight JSON body")
        String requestJson;

        @Option(names = { "--request-file" }, description = "Path to OpenAI-compatible request JSON, or a full preflight JSON body")
        String requestFile;

        @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s)")
        List<String> toolFiles = new ArrayList<>();

        @Option(names = { "--rag-context" }, split = "\\|", description = "Inline RAG context block(s), separated with |")
        List<String> ragContexts = new ArrayList<>();

        @Option(names = { "--rag-file", "--context-file" }, split = ",", description = "Text file(s) to include as caller-owned RAG context")
        List<String> ragFiles = new ArrayList<>();

        @Option(names = { "--no-mcp" }, description = "Skip MCP tool discovery")
        boolean noMcp;

        @Option(names = { "--mcp-optional" }, description = "Warn instead of blocking when MCP discovery is unavailable")
        boolean mcpOptional;

        @Option(names = { "--require-tools" }, description = "Exit 1 when MCP/request tool discovery returns no tools")
        boolean requireTools;

        @Option(names = { "--require-clean" }, description = "Exit 1 when tool or request validation returns portability warnings")
        boolean requireClean;

        @Option(names = { "--raw-mcp-tools" }, description = "Ask MCP discovery for raw MCP tools instead of OpenAI-compatible tools")
        boolean rawMcpTools;

        @Option(names = { "--include-disabled-mcp" }, description = "Include disabled MCP server registrations in discovery")
        boolean includeDisabledMcp;

        @Option(names = { "--no-tool-validation" }, description = "Skip tool schema validation")
        boolean noToolValidation;

        @Option(names = { "--no-request-validation" }, description = "Skip request dry-run validation")
        boolean noRequestValidation;

        @Override
        String commandLabel() {
            return "Agent readiness";
        }

        @Override
        String configErrorObject() {
            return "gollek.agent_readiness_error";
        }

        @Override
        Map<String, Object> fetch(AgentIntegrationClient client, AgentRequestOptions options)
                throws AgentIntegrationException {
            try {
                return readinessReport(client, options);
            } catch (IOException e) {
                throw new IllegalArgumentException(e.getMessage(), e);
            }
        }

        @Override
        void printHumanReport(Map<String, Object> response, List<String> issues) {
            System.out.println("Gollek agent readiness: " + text(response.get("status")));
            System.out.println("Model: " + text(response.get("model")));
            System.out.println("Surface: " + text(response.get("surface")));
            System.out.println("Feature profile: " + text(response.get("feature_profile")));
            System.out.println("Checks: ready=" + text(response.get("ready_check_count"))
                    + ", blocked=" + text(response.get("blocked_check_count"))
                    + ", skipped=" + text(response.get("skipped_check_count"))
                    + ", warnings=" + text(response.get("warning_count")));
            for (Map.Entry<String, Object> entry : checks(response).entrySet()) {
                Map<String, Object> check = mapValue(entry.getValue());
                System.out.println("- " + entry.getKey() + ": " + text(check.get("status"))
                        + " (" + text(check.get("endpoint")) + ")");
                AgentCommand.printCheckMessages(entry.getKey(), "blocking", check);
                AgentCommand.printCheckMessages(entry.getKey(), "warning", check);
                AgentCommand.printCheckDetails(check);
            }
            AgentCommand.printMessagesWithCodes(
                    "Blocking readiness issues",
                    issues,
                    response.get("blocking_codes"),
                    "blocking",
                    response.get("issue_hints"));
            AgentCommand.printMessagesWithCodes(
                    "Readiness warnings",
                    response.get("warning_messages"),
                    response.get("warning_codes"),
                    "warning",
                    response.get("issue_hints"));
        }

        @Override
        List<String> servingBoundaryIssues(Map<String, Object> response) {
            return stringList(response.get("blocking_messages"));
        }

        @Override
        boolean shouldFail(Map<String, Object> response, List<String> issues) {
            return !Boolean.TRUE.equals(response.get("ready"));
        }

        private Map<String, Object> readinessReport(AgentIntegrationClient client, AgentRequestOptions options)
                throws IOException {
            AgentRouteInput input = loadInput();
            Map<String, Object> request = new LinkedHashMap<>(input.request());
            AgentRequestPayloads.appendToolFiles(request, toolFiles, DISCOVERY_MAPPER);
            AgentRequestPayloads.appendRagContext(request, ragContexts, ragFiles);

            String effectiveModel = firstNonBlank(modelId, input.modelId(), stringValue(request.get("model")));
            String effectiveFeatureProfile = AgentServingFeatureProfile.normalizeName(
                    firstNonBlank(featureProfile, input.featureProfile(), AgentServingFeatureProfile.DEFAULT_PROFILE));
            String effectiveSurface = AgentRequestShape.resolveSurface(
                    DEFAULT_SURFACE,
                    surface,
                    input.surface(),
                    stringValue(request.get("surface")),
                    request,
                    effectiveFeatureProfile);
            if (!hasText(effectiveModel)) {
                throw new IllegalArgumentException("Provide --model or include model/model_id in the request JSON.");
            }
            AgentRouteConsistency.Result routeConsistency = AgentRouteConsistency.validateAndNormalize(
                    input,
                    request,
                    new AgentRouteConsistency.Overrides(
                            hasText(modelId),
                            hasText(surface),
                            hasText(featureProfile)),
                    new AgentRouteConsistency.Route(
                            effectiveModel,
                            effectiveSurface,
                            effectiveFeatureProfile));
            if (!routeConsistency.ok()) {
                throw routeConsistency.toException();
            }
            request.put("model", effectiveModel);

            Map<String, Object> checks = new LinkedHashMap<>();
            List<String> blockingMessages = new ArrayList<>();
            List<String> warningMessages = new ArrayList<>();

            addCapabilitiesCheck(client, options, checks, blockingMessages);
            addContractCheck(client, options, checks, blockingMessages);
            addModelCapabilitiesCheck(client, options, effectiveModel, checks, blockingMessages);
            Map<String, Object> mcpTools = addMcpCheck(client, options, checks, blockingMessages, warningMessages);
            appendDiscoveredTools(request, mcpTools);
            addToolValidationCheck(client, options, request, checks, blockingMessages, warningMessages);
            addRequestValidationCheck(client, options, request, effectiveSurface, checks, blockingMessages, warningMessages);
            addPreflightCheck(
                    client,
                    request,
                    options,
                    effectiveModel,
                    effectiveSurface,
                    effectiveFeatureProfile,
                    checks,
                    blockingMessages,
                    warningMessages);

            Map<String, Object> report = new LinkedHashMap<>();
            boolean ready = blockingMessages.isEmpty();
            report.put("object", "gollek.agent_readiness_report");
            report.put("status", ready ? "ready" : "blocked");
            report.put("ready", ready);
            report.put("model", effectiveModel);
            report.put("surface", effectiveSurface);
            report.put("feature_profile", effectiveFeatureProfile);
            report.put("boundary", Map.of(
                    "validation_only", true,
                    "model_invoked", false,
                    "tool_execution", false,
                    "retrieval_execution", false,
                    "tool_authorization", false));
            report.put("checks", checks);
            report.putAll(AgentReadinessMetadata.fromChecks(checks, true));
            return report;
        }

        private void addCapabilitiesCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                Map<String, Object> checks,
                List<String> blockingMessages) {
            String id = "capabilities";
            try {
                Map<String, Object> response = client.capabilities(options);
                List<String> issues = AgentCapabilitiesView.from(response, DISCOVERY_MAPPER)
                        .agentServingCapabilityIssues();
                addCheck(checks, id, "/v1/agent/capabilities", response, issues, List.of());
                blockingMessages.addAll(prefix(id, issues));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, "/v1/agent/capabilities", e, blockingMessages);
            }
        }

        private void addContractCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                Map<String, Object> checks,
                List<String> blockingMessages) {
            String id = "contract";
            try {
                Map<String, Object> response = client.contract(options);
                List<String> issues = AgentServingContract.from(response, DISCOVERY_MAPPER).servingBoundaryIssues();
                addCheck(checks, id, "/v1/agent/contract", response, issues, List.of());
                blockingMessages.addAll(prefix(id, issues));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, "/v1/agent/contract", e, blockingMessages);
            }
        }

        private void addModelCapabilitiesCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                String effectiveModel,
                Map<String, Object> checks,
                List<String> blockingMessages) {
            String id = "model_capabilities";
            String endpoint = "/v1/models/" + effectiveModel + "/capabilities";
            try {
                Map<String, Object> response = client.modelCapabilities(effectiveModel, options);
                List<String> issues = AgentModelCapabilitiesView.from(response, DISCOVERY_MAPPER)
                        .agentServingRouteIssues();
                addCheck(checks, id, endpoint, response, issues, List.of());
                blockingMessages.addAll(prefix(id, issues));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, endpoint, e, blockingMessages);
            }
        }

        private Map<String, Object> addMcpCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                Map<String, Object> checks,
                List<String> blockingMessages,
                List<String> warningMessages) {
            String id = "mcp_tools";
            if (noMcp) {
                addSkipped(checks, id, "/v1/mcp/tools", "MCP discovery skipped by --no-mcp.");
                return null;
            }
            String endpoint = "/v1/mcp/tools?compat=" + (rawMcpTools ? "mcp" : "openai")
                    + "&enabledOnly=" + !includeDisabledMcp;
            try {
                Map<String, Object> response = client.mcpTools(!rawMcpTools, !includeDisabledMcp, options);
                AgentMcpDiscoveryView discovery = AgentMcpDiscoveryView.from(response, DISCOVERY_MAPPER);
                List<String> issues = mcpIssues(discovery);
                if (mcpOptional) {
                    addCheck(checks, id, endpoint, response, List.of(), issues);
                    warningMessages.addAll(prefix(id, issues));
                } else {
                    addCheck(checks, id, endpoint, response, issues, List.of());
                    blockingMessages.addAll(prefix(id, issues));
                }
                return response;
            } catch (AgentIntegrationException e) {
                if (mcpOptional) {
                    List<String> warnings = List.of(e.getMessage());
                    addCheck(checks, id, endpoint, Map.of(), List.of(), warnings);
                    warningMessages.addAll(prefix(id, warnings));
                    return null;
                }
                addFailure(checks, id, endpoint, e, blockingMessages);
                return null;
            }
        }

        private void addToolValidationCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                Map<String, Object> request,
                Map<String, Object> checks,
                List<String> blockingMessages,
                List<String> warningMessages) {
            String id = "tool_validation";
            if (noToolValidation) {
                addSkipped(checks, id, "/v1/agent/tools/validate", "Tool validation skipped by --no-tool-validation.");
                return;
            }
            try {
                List<Object> tools = AgentRequestPayloads.listValue(request.get("tools"), "tools");
                Map<String, Object> response = client.validateTools(Map.of("tools", tools), options);
                AgentToolValidationView validation = AgentToolValidationView.from(response, DISCOVERY_MAPPER);
                List<String> issues = toolValidationIssues(validation);
                List<String> warnings = validation.warnings().stream()
                        .map(AgentToolValidationView.Warning::message)
                        .filter(Readiness::hasText)
                        .toList();
                if (requireClean && validation.hasWarnings()) {
                    issues = appendIssue(issues, "tool validation returned portability warnings");
                }
                if (requireTools && tools.isEmpty()) {
                    issues = appendIssue(issues, "request has no tool definitions");
                }
                addCheck(checks, id, "/v1/agent/tools/validate", response, issues, warnings);
                blockingMessages.addAll(prefix(id, issues));
                warningMessages.addAll(prefix(id, warnings));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, "/v1/agent/tools/validate", e, blockingMessages);
            }
        }

        private void addRequestValidationCheck(
                AgentIntegrationClient client,
                AgentRequestOptions options,
                Map<String, Object> request,
                String effectiveSurface,
                Map<String, Object> checks,
                List<String> blockingMessages,
                List<String> warningMessages) {
            String id = "request_validation";
            if (noRequestValidation) {
                addSkipped(checks, id, "/v1/agent/validate", "Request validation skipped by --no-request-validation.");
                return;
            }
            String endpoint = "/v1/agent/validate?surface=" + effectiveSurface;
            try {
                Map<String, Object> response = client.validateRequest(effectiveSurface, request, options);
                AgentValidationView validation = AgentValidationView.from(response, DISCOVERY_MAPPER);
                List<String> issues = requestValidationIssues(validation);
                List<String> warnings = validation.toolContractWarningCount() > 0
                        ? List.of("request validation returned " + validation.toolContractWarningCount()
                                + " nested tool-contract warning(s)")
                        : List.of();
                if (requireClean && validation.toolContractWarningCount() > 0) {
                    issues = appendIssue(issues, "request validation returned nested tool-contract warnings");
                }
                addCheck(checks, id, endpoint, response, issues, warnings);
                blockingMessages.addAll(prefix(id, issues));
                warningMessages.addAll(prefix(id, warnings));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, endpoint, e, blockingMessages);
            }
        }

        private void addPreflightCheck(
                AgentIntegrationClient client,
                Map<String, Object> request,
                AgentRequestOptions options,
                String effectiveModel,
                String effectiveSurface,
                String effectiveFeatureProfile,
                Map<String, Object> checks,
                List<String> blockingMessages,
                List<String> warningMessages) {
            String id = "preflight";
            try {
                AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                        .modelId(effectiveModel)
                        .surface(effectiveSurface)
                        .featureProfile(effectiveFeatureProfile)
                        .request(request)
                        .requestOptions(options)
                        .discoverMcpTools(!noMcp)
                        .mcpDiscoveryRequired(!noMcp && !mcpOptional)
                        .validateTools(!noToolValidation)
                        .toolValidationRequired(!noToolValidation)
                        .validateRequest(!noRequestValidation)
                        .requestValidationRequired(!noRequestValidation)
                        .openAiToolCompatibility(!rawMcpTools)
                        .enabledOnly(!includeDisabledMcp)
                        .build();
                Map<String, Object> response = client.servingPreflight(preflight);
                List<String> issues = isReady(response)
                        ? List.of()
                        : fallbackMessages(response, "preflight is blocked");
                List<String> warnings = stringList(response.get("warning_messages"));
                addCheck(checks, id, "/v1/agent/preflight", response, issues, warnings);
                blockingMessages.addAll(prefix(id, issues));
                warningMessages.addAll(prefix(id, warnings));
            } catch (AgentIntegrationException e) {
                addFailure(checks, id, "/v1/agent/preflight", e, blockingMessages);
            }
        }

        private AgentRouteInput loadInput() throws IOException {
            if (hasText(requestJson) && hasText(requestFile)) {
                throw new IllegalArgumentException("Use either --request-json or --request-file, not both.");
            }
            if (hasText(requestJson)) {
                return AgentRouteInput.parse(requestJson.trim(), DISCOVERY_MAPPER, "Readiness input");
            }
            if (hasText(requestFile)) {
                return AgentRouteInput.parse(
                        Files.readString(Path.of(requestFile.trim())),
                        DISCOVERY_MAPPER,
                        "Readiness input");
            }
            return AgentRouteInput.generated(defaultRequest());
        }

        private Map<String, Object> defaultRequest() {
            return AgentRequestShape.generatedRequest(
                    surface,
                    featureProfile,
                    prompt,
                    DEFAULT_PROMPT,
                    systemPrompt,
                    DEFAULT_SYSTEM_PROMPT);
        }

        private void appendDiscoveredTools(Map<String, Object> request, Map<String, Object> mcpTools) {
            if (mcpTools == null || request.containsKey("tools")) {
                return;
            }
            List<Map<String, Object>> tools = AgentMcpDiscoveryView.from(mcpTools, DISCOVERY_MAPPER)
                    .openAiToolDefinitions();
            if (!tools.isEmpty()) {
                request.put("tools", new ArrayList<>(tools));
            }
        }

        private void addCheck(
                Map<String, Object> checks,
                String id,
                String endpoint,
                Map<String, Object> response,
                List<String> blocking,
                List<String> warnings) {
            Map<String, Object> check = new LinkedHashMap<>();
            List<String> blockingMessages = blocking == null ? List.of() : List.copyOf(blocking);
            List<String> warningMessages = warnings == null ? List.of() : List.copyOf(warnings);
            check.put("status", blockingMessages.isEmpty() ? "ready" : "blocked");
            check.put("ready", blockingMessages.isEmpty());
            check.put("endpoint", endpoint);
            check.put("blocking_messages", blockingMessages);
            check.put("warning_messages", warningMessages);
            List<String> blockingCodes = issueCodes(id, "blocking", blockingMessages,
                    response == null ? null : response.get("blocking_codes"));
            List<String> warningCodes = issueCodes(id, "warning", warningMessages,
                    response == null ? null : response.get("warning_codes"));
            check.put("blocking_codes", blockingCodes);
            check.put("warning_codes", warningCodes);
            check.put("issue_hints", issueHintMaps(id, blockingMessages, blockingCodes, warningMessages, warningCodes));
            check.put("response", response == null ? Map.of() : response);
            Map<String, Object> details = AgentCommand.checkDetails(id, response);
            if (!details.isEmpty()) {
                check.put("details", details);
            }
            checks.put(id, check);
        }

        private void addSkipped(Map<String, Object> checks, String id, String endpoint, String message) {
            Map<String, Object> check = new LinkedHashMap<>();
            check.put("status", "skipped");
            check.put("ready", true);
            check.put("endpoint", endpoint);
            check.put("blocking_messages", List.of());
            check.put("warning_messages", List.of(message));
            check.put("blocking_codes", List.of());
            List<String> warningCodes = issueCodes(id, "warning", List.of(message), null);
            check.put("warning_codes", warningCodes);
            check.put("issue_hints", issueHintMaps(id, List.of(), List.of(), List.of(message), warningCodes));
            check.put("response", Map.of());
            checks.put(id, check);
        }

        private void addFailure(
                Map<String, Object> checks,
                String id,
                String endpoint,
                AgentIntegrationException error,
                List<String> blockingMessages) {
            List<String> issues = List.of(error.getMessage());
            Map<String, Object> response = Map.of(
                    "error_code", error.getErrorCode(),
                    "message", error.getMessage());
            addCheck(checks, id, endpoint, response, issues, List.of());
            blockingMessages.addAll(prefix(id, issues));
        }

        private List<String> mcpIssues(AgentMcpDiscoveryView discovery) {
            List<String> issues = new ArrayList<>();
            if (!discovery.available()) {
                issues.add("MCP registry is unavailable");
            }
            if (!discovery.discoveryOnly()) {
                issues.add("MCP discovery must be discovery-only");
            }
            if (discovery.toolExecutionEnabled()) {
                issues.add("MCP discovery must not execute tools");
            }
            if (requireTools && discovery.tools().isEmpty()) {
                issues.add("MCP discovery returned no tools");
            }
            return List.copyOf(issues);
        }

        private List<String> toolValidationIssues(AgentToolValidationView validation) {
            List<String> issues = new ArrayList<>();
            if (!validation.valid()) {
                issues.add("tool contract validation response is not valid");
            }
            if (!validation.validationOnly()) {
                issues.add("tool validation must be validation-only");
            }
            if (validation.modelInvoked()) {
                issues.add("tool validation must not invoke a model");
            }
            if (validation.toolExecutionEnabled()) {
                issues.add("tool validation must not execute tools");
            }
            if (validation.toolAuthorizationEnabled()) {
                issues.add("tool validation must not authorize tools");
            }
            return List.copyOf(issues);
        }

        private List<String> requestValidationIssues(AgentValidationView validation) {
            List<String> issues = new ArrayList<>();
            if (!validation.valid()) {
                issues.add("request validation response is not valid");
            }
            if (!validation.validationOnly()) {
                issues.add("request validation must be validation-only");
            }
            if (validation.modelInvoked()) {
                issues.add("request validation must not invoke a model");
            }
            if (validation.toolExecutionEnabled()) {
                issues.add("request validation must not execute tools");
            }
            if (validation.retrievalExecutionEnabled()) {
                issues.add("request validation must not execute retrieval");
            }
            if (!validation.toolContractValid()) {
                issues.add("nested tool contract validation is not valid");
            }
            return List.copyOf(issues);
        }

        private static List<String> prefix(String id, List<String> messages) {
            if (messages == null || messages.isEmpty()) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (String message : messages) {
                if (hasText(message)) {
                    out.add(id + ": " + message);
                }
            }
            return List.copyOf(out);
        }

        private static List<String> appendIssue(List<String> issues, String issue) {
            List<String> out = new ArrayList<>(issues == null ? List.of() : issues);
            out.add(issue);
            return List.copyOf(out);
        }

        private static List<String> fallbackMessages(Map<String, Object> response, String fallback) {
            List<String> messages = stringList(response.get("blocking_messages"));
            return messages.isEmpty() ? List.of(fallback) : messages;
        }

        private static boolean isReady(Map<String, Object> response) {
            Object ready = response.get("ready");
            if (ready instanceof Boolean bool) {
                return bool;
            }
            return "ready".equalsIgnoreCase(stringValue(response.get("status")));
        }

        @SuppressWarnings("unchecked")
        private static Map<String, Object> mapValue(Object value) {
            return value instanceof Map<?, ?> map ? (Map<String, Object>) map : Map.of();
        }

        private static Map<String, Object> checks(Map<String, Object> response) {
            return mapValue(response.get("checks"));
        }

        private static long countChecks(Map<String, Object> checks, String status) {
            return checks.values().stream()
                    .map(Readiness::mapValue)
                    .filter(check -> status.equals(check.get("status")))
                    .count();
        }

        private static List<String> stringList(Object value) {
            if (!(value instanceof List<?> list)) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (Object item : list) {
                String text = stringValue(item);
                if (hasText(text)) {
                    out.add(text);
                }
            }
            return List.copyOf(out);
        }

        private static String stringValue(Object value) {
            return value == null ? null : value.toString();
        }
    }

    @Command(
            name = "preflight",
            description = "Validate an agent-serving route without invoking a model")
    public static class Preflight implements Callable<Integer> {
        private static final ObjectMapper MAPPER = new ObjectMapper();
        private static final String DEFAULT_BASE_URL = "http://localhost:8080";
        private static final String DEFAULT_API_KEY = "community";
        private static final String DEFAULT_SURFACE = "chat";
        private static final String DEFAULT_PROMPT = "Can this route support my agent request?";
        private static final String DEFAULT_SYSTEM_PROMPT = "Use supplied context and tools when relevant.";

        @ParentCommand
        AgentCommand parentCommand;

        @Option(names = { "--base-url" }, description = "Gollek serving base URL")
        String baseUrl;

        @Option(names = { "--api-key" }, description = "Gollek API key")
        String apiKey;

        @Option(names = { "-m", "--model" }, description = "Model id to preflight")
        String modelId;

        @Option(names = { "--surface" }, description = "Compatibility surface: chat, responses, or embeddings")
        String surface;

        @Option(names = { "--feature-profile" }, description = "Serving feature profile: agent_serving, chat_agent, responses_agent, embedding_rag, or mcp_tools")
        String featureProfile;

        @Option(names = { "-p", "--prompt", "--input" }, description = "Prompt/input used when no request JSON is provided")
        String prompt;

        @Option(names = { "--system" }, description = "System prompt used when building a chat request")
        String systemPrompt;

        @Option(names = { "--request-json" }, description = "Inline OpenAI-compatible request JSON, or a full preflight JSON body")
        String requestJson;

        @Option(names = { "--request-file" }, description = "Path to OpenAI-compatible request JSON, or a full preflight JSON body")
        String requestFile;

        @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s)")
        List<String> toolFiles = new ArrayList<>();

        @Option(names = { "--rag-context" }, split = "\\|", description = "Inline RAG context block(s), separated with |")
        List<String> ragContexts = new ArrayList<>();

        @Option(names = { "--rag-file", "--context-file" }, split = ",", description = "Text file(s) to include as caller-owned RAG context")
        List<String> ragFiles = new ArrayList<>();

        @Option(names = { "--no-mcp" }, description = "Skip MCP tool discovery for this preflight")
        boolean noMcp;

        @Option(names = { "--mcp-optional" }, description = "Warn instead of blocking when MCP discovery is unavailable")
        boolean mcpOptional;

        @Option(names = { "--no-tool-validation" }, description = "Skip tool schema validation")
        boolean noToolValidation;

        @Option(names = { "--tool-validation-optional" }, description = "Warn instead of blocking on tool-validation availability")
        boolean toolValidationOptional;

        @Option(names = { "--no-request-validation" }, description = "Skip request dry-run validation")
        boolean noRequestValidation;

        @Option(names = { "--request-validation-optional" }, description = "Warn instead of blocking on request-validation availability")
        boolean requestValidationOptional;

        @Option(names = { "--raw-mcp-tools" }, description = "Ask MCP discovery for raw MCP tools instead of OpenAI-compatible tools")
        boolean rawMcpTools;

        @Option(names = { "--include-disabled-mcp" }, description = "Include disabled MCP server registrations in discovery")
        boolean includeDisabledMcp;

        @Option(names = { "--request-id" }, description = "Trace request id sent as X-Gollek-Request-Id")
        String requestId;

        @Option(names = { "--trace-id" }, description = "Trace id sent as X-Gollek-Trace-Id")
        String traceId;

        @Option(names = { "--session-id" }, description = "Session id sent as X-Gollek-Session-Id")
        String sessionId;

        @Option(names = { "--user-id" }, description = "User id sent as X-Gollek-User-Id")
        String userId;

        @Option(names = { "--header" }, split = ",", description = "Extra header as Name:Value, repeatable or comma-separated")
        List<String> headers = new ArrayList<>();

        @Option(names = { "--json" }, description = "Print the typed preflight result as JSON")
        boolean json;

        @Option(names = { "--require-route-match" }, description = "Exit blocked when Gollek selects a different route than requested")
        boolean requireRouteMatch;

        @Option(names = { "--preflight-policy" }, description = "Gate profile: readiness_only, requested_route, or clean")
        String preflightPolicy;

        @Option(names = { "--fail-on-warning", "--fail-on-warnings" }, description = "Exit blocked when preflight returns warnings")
        boolean failOnWarnings;

        @Override
        public Integer call() {
            try {
                if (parentCommand != null && parentCommand.root() != null) {
                    parentCommand.root().bootstrapInheritedEnvironment();
                }

                AgentRouteInput input = loadInput();
                Map<String, Object> request = new LinkedHashMap<>(input.request());
                AgentRequestPayloads.appendToolFiles(request, toolFiles, MAPPER);
                AgentRequestPayloads.appendRagContext(request, ragContexts, ragFiles);

                String effectiveModel = firstNonBlank(modelId, input.modelId(), text(request.get("model")));
                String effectiveFeatureProfile = AgentServingFeatureProfile.normalizeName(
                        firstNonBlank(featureProfile, input.featureProfile(), AgentServingFeatureProfile.DEFAULT_PROFILE));
                String effectiveSurface = AgentRequestShape.resolveSurface(
                        DEFAULT_SURFACE,
                        surface,
                        input.surface(),
                        text(request.get("surface")),
                        request,
                        effectiveFeatureProfile);
                if (effectiveModel == null || effectiveModel.isBlank()) {
                    printConfigError("Provide --model or include model/model_id in the request JSON.");
                    return 2;
                }
                AgentRouteConsistency.Result routeConsistency = AgentRouteConsistency.validateAndNormalize(
                        input,
                        request,
                        new AgentRouteConsistency.Overrides(
                                hasText(modelId),
                                hasText(surface),
                                hasText(featureProfile)),
                        new AgentRouteConsistency.Route(
                                effectiveModel,
                                effectiveSurface,
                                effectiveFeatureProfile));
                if (!routeConsistency.ok()) {
                    throw routeConsistency.toException();
                }
                AgentServingPreflightPolicy gatePolicy;
                try {
                    gatePolicy = effectivePreflightPolicy(input);
                } catch (IllegalArgumentException e) {
                    printConfigError(e.getMessage());
                    return 2;
                }

                AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                        .modelId(effectiveModel)
                        .surface(effectiveSurface)
                        .featureProfile(effectiveFeatureProfile)
                        .request(request)
                        .requestOptions(requestOptions())
                        .discoverMcpTools(effectiveBoolean(input.discoverMcpTools(), true, noMcp, false))
                        .mcpDiscoveryRequired(effectiveBoolean(input.mcpDiscoveryRequired(), true, noMcp || mcpOptional, false))
                        .validateTools(effectiveBoolean(input.validateTools(), true, noToolValidation, false))
                        .toolValidationRequired(effectiveBoolean(input.toolValidationRequired(), true, noToolValidation || toolValidationOptional, false))
                        .validateRequest(effectiveBoolean(input.validateRequest(), true, noRequestValidation, false))
                        .requestValidationRequired(effectiveBoolean(input.requestValidationRequired(), true, noRequestValidation || requestValidationOptional, false))
                        .openAiToolCompatibility(effectiveBoolean(input.openAiToolCompatibility(), true, rawMcpTools, false))
                        .enabledOnly(effectiveBoolean(input.enabledOnly(), true, includeDisabledMcp, false))
                        .build();

                AgentIntegrationClient client = new AgentIntegrationClient(
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(),
                        MAPPER,
                        effectiveBaseUrl(),
                        effectiveApiKey());
                AgentServingPreflightResult result = client.servingPreflightResult(preflight);
                AgentServingPreflightGate gate = result.gate(gatePolicy);

                if (json) {
                    System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(outputMetadata(gate)));
                } else {
                    printHumanReport(gate, effectiveModel, effectiveSurface, effectiveFeatureProfile);
                }
                return gate.ready() ? 0 : 1;
            } catch (AgentRouteConsistency.ConflictException e) {
                printRouteConfigError(e.result());
                return 2;
            } catch (AgentIntegrationException e) {
                printError("Agent preflight request failed: " + e.getMessage());
                return 2;
            } catch (Exception e) {
                printError("Agent preflight failed: " + e.getMessage());
                return 2;
            }
        }

        private AgentRouteInput loadInput() throws IOException {
            if (hasText(requestJson) && hasText(requestFile)) {
                throw new IllegalArgumentException("Use either --request-json or --request-file, not both.");
            }
            if (hasText(requestJson)) {
                return AgentRouteInput.parse(requestJson.trim(), MAPPER, "Preflight input");
            }
            if (hasText(requestFile)) {
                return AgentRouteInput.parse(
                        Files.readString(Path.of(requestFile.trim())),
                        MAPPER,
                        "Preflight input");
            }
            return AgentRouteInput.generated(defaultRequest());
        }

        private Map<String, Object> defaultRequest() {
            return AgentRequestShape.generatedRequest(
                    surface,
                    featureProfile,
                    prompt,
                    DEFAULT_PROMPT,
                    systemPrompt,
                    DEFAULT_SYSTEM_PROMPT);
        }

        private AgentServingPreflightPolicy effectivePreflightPolicy(AgentRouteInput input) {
            AgentServingPreflightPolicy policy = AgentServingPreflightPolicy.requireProfile(
                    firstNonBlank(preflightPolicy, input.preflightPolicy()));
            boolean effectiveRequireRouteMatch = policy.requireRouteMatch()
                    || bool(input.requireRouteMatch())
                    || requireRouteMatch;
            boolean effectiveFailOnWarnings = policy.failOnWarnings()
                    || bool(input.failOnWarnings())
                    || failOnWarnings;
            if (effectiveRequireRouteMatch == policy.requireRouteMatch()
                    && effectiveFailOnWarnings == policy.failOnWarnings()) {
                return policy;
            }
            return AgentServingPreflightPolicy.of(effectiveRequireRouteMatch, effectiveFailOnWarnings);
        }

        private AgentRequestOptions requestOptions() {
            AgentRequestOptions.Builder builder = AgentRequestOptions.builder()
                    .requestId(requestId)
                    .traceId(traceId)
                    .sessionId(sessionId)
                    .userId(userId);
            if (headers != null) {
                for (String header : headers) {
                    if (!hasText(header)) {
                        continue;
                    }
                    int separator = header.indexOf(':');
                    if (separator <= 0 || separator == header.length() - 1) {
                        throw new IllegalArgumentException("--header must use Name:Value format.");
                    }
                    builder.header(header.substring(0, separator).trim(), header.substring(separator + 1).trim());
                }
            }
            return builder.build();
        }

        private void printHumanReport(
                AgentServingPreflightGate gate,
                String model,
                String surface,
                String featureProfile) {
            AgentServingPreflightResult result = gate.result();
            Map<String, Object> response = result.response();
            String status = text(response.get("status"));
            System.out.println("Gollek agent preflight: " + firstNonBlank(status, isReady(response) ? "ready" : "blocked"));
            System.out.println("Endpoint: " + normalizedBaseUrl(effectiveBaseUrl()) + "/v1/agent/preflight");
            System.out.println("Model: " + model);
            System.out.println("Surface: " + surface);
            System.out.println("Feature profile: " + featureProfile);
            printRouteComparison(result);
            printPreflightGate(gate);
            printBoundary(response);
            int blocking = intValue(response.get("blocking_issue_count"));
            int warnings = intValue(response.get("warning_count"));
            System.out.println("Issues: blocking=" + blocking + ", warnings=" + warnings);
            printCheckResults(response);
            AgentCommand.printMessagesWithCodes(
                    "Blocking issues",
                    response.get("blocking_messages"),
                    response.get("blocking_codes"),
                    "blocking",
                    response.get("issue_hints"));
            AgentCommand.printMessagesWithCodes(
                    "Warnings",
                    response.get("warning_messages"),
                    response.get("warning_codes"),
                    "warning",
                    response.get("issue_hints"));
        }

        private void printRouteComparison(AgentServingPreflightResult result) {
            AgentServingRoute selected = result.selectedRoute();
            if (!selected.known()) {
                System.out.println("Selected route: n/a");
                System.out.println("Route comparison: unavailable");
                return;
            }
            System.out.println("Selected route: model=" + firstNonBlank(selected.model(), "n/a")
                    + ", surface=" + firstNonBlank(selected.surface(), "n/a")
                    + ", feature_profile=" + firstNonBlank(selected.featureProfile(), "n/a"));
            if (result.routeMatches()) {
                System.out.println("Route comparison: match");
                return;
            }
            System.out.println("Route comparison: mismatch fields=" + String.join(",", result.routeMismatchFields()));
            for (String message : result.routeMismatchMessages()) {
                System.out.println("  mismatch: " + message);
            }
        }

        private Map<String, Object> outputMetadata(AgentServingPreflightGate gate) {
            Map<String, Object> out = new LinkedHashMap<>(gate.result().toMetadata());
            Map<String, Object> gateMetadata = gate.toMetadata();
            out.put("preflight_gate", gateMetadata);
            out.put("preflight_policy", gate.policy().toMetadata());
            out.put("route_match_required", gate.requireRouteMatch());
            out.put("warnings_blocking", gate.failOnWarnings());
            out.put("exit_ready", gate.ready());
            return out;
        }

        private void printPreflightGate(AgentServingPreflightGate gate) {
            if (!gate.requireRouteMatch() && !gate.failOnWarnings()) {
                return;
            }
            System.out.println("Preflight gate: " + gate.status());
            for (String message : gate.blockingMessages()) {
                System.out.println("  blocking: " + message);
            }
        }

        @SuppressWarnings("unchecked")
        private void printBoundary(Map<String, Object> response) {
            Object raw = response.get("boundary");
            if (!(raw instanceof Map<?, ?> boundary)) {
                return;
            }
            Map<String, Object> typed = (Map<String, Object>) boundary;
            System.out.println("Boundary: validation_only=" + boolText(typed.get("validation_only"))
                    + ", model_invoked=" + boolText(typed.get("model_invoked"))
                    + ", tool_execution=" + boolText(typed.get("tool_execution"))
                    + ", retrieval_execution=" + boolText(typed.get("retrieval_execution"))
                    + ", tool_authorization=" + boolText(typed.get("tool_authorization")));
        }

        private void printCheckResults(Map<String, Object> response) {
            Map<String, Object> checks = checkResults(response);
            if (checks.isEmpty()) {
                return;
            }
            int ready = hasCount(response, "ready_check_count")
                    ? intValue(response.get("ready_check_count"))
                    : countCheckStatus(checks, "ready");
            int blocked = hasCount(response, "blocked_check_count")
                    ? intValue(response.get("blocked_check_count"))
                    : countCheckStatus(checks, "blocked");
            int skipped = hasCount(response, "skipped_check_count")
                    ? intValue(response.get("skipped_check_count"))
                    : countCheckStatus(checks, "skipped");
            System.out.println("Checks: ready=" + ready + ", blocked=" + blocked + ", skipped=" + skipped);
            for (Map.Entry<String, Object> entry : checks.entrySet()) {
                Map<String, Object> check = mapValue(entry.getValue());
                String requested = firstNonBlank(boolText(check.get("requested")), "n/a");
                System.out.println("- " + entry.getKey() + ": " + text(check.get("status"))
                        + ", requested=" + requested);
                AgentCommand.printCheckMessages(entry.getKey(), "blocking", check);
                AgentCommand.printCheckMessages(entry.getKey(), "warning", check);
                AgentCommand.printCheckDetails(check);
            }
        }

        private Map<String, Object> checkResults(Map<String, Object> response) {
            Object raw = response.get("check_results");
            if (!(raw instanceof Map<?, ?>) && response.get("readiness_report") instanceof Map<?, ?> report) {
                raw = report.get("checks");
            }
            return mapValue(raw);
        }

        private static int countCheckStatus(Map<String, Object> checks, String status) {
            int count = 0;
            for (Object raw : checks.values()) {
                Map<String, Object> check = mapValue(raw);
                if (status.equals(check.get("status"))) {
                    count++;
                }
            }
            return count;
        }

        private static boolean hasCount(Map<String, Object> response, String field) {
            return response.containsKey(field) && response.get(field) != null;
        }

        @SuppressWarnings("unchecked")
        private static Map<String, Object> mapValue(Object value) {
            if (!(value instanceof Map<?, ?> map)) {
                return Map.of();
            }
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (entry.getKey() != null) {
                    out.put(entry.getKey().toString(), entry.getValue());
                }
            }
            return out;
        }

        private boolean isReady(Map<String, Object> response) {
            Object ready = response.get("ready");
            if (ready instanceof Boolean value) {
                return value;
            }
            return "ready".equalsIgnoreCase(text(response.get("status")));
        }

        private String effectiveBaseUrl() {
            return firstNonBlank(
                    baseUrl,
                    System.getProperty("gollek.api.base-url"),
                    System.getProperty("gollek.server.base-url"),
                    System.getenv("GOLLEK_BASE_URL"),
                    System.getenv("GOLLEK_API_URL"),
                    System.getenv("GOLLEK_SERVER_URL"),
                    DEFAULT_BASE_URL);
        }

        private String effectiveApiKey() {
            return firstNonBlank(
                    apiKey,
                    System.getProperty("gollek.api.key"),
                    System.getenv("GOLLEK_API_KEY"),
                    DEFAULT_API_KEY);
        }

        private static boolean effectiveBoolean(Boolean configured, boolean fallback, boolean override, boolean overrideValue) {
            if (override) {
                return overrideValue;
            }
            return configured == null ? fallback : configured;
        }

        private static String normalizedBaseUrl(String value) {
            String out = firstNonBlank(value, DEFAULT_BASE_URL);
            while (out.endsWith("/")) {
                out = out.substring(0, out.length() - 1);
            }
            return out;
        }

        private static int intValue(Object value) {
            if (value instanceof Number number) {
                return number.intValue();
            }
            if (value instanceof String text && !text.isBlank()) {
                try {
                    return Integer.parseInt(text);
                } catch (NumberFormatException ignored) {
                    return 0;
                }
            }
            return 0;
        }

        private static List<String> stringList(Object value) {
            if (!(value instanceof List<?> list)) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (Object item : list) {
                String text = text(item);
                if (hasText(text)) {
                    out.add(text);
                }
            }
            return out;
        }

        private static String boolText(Object value) {
            if (value instanceof Boolean bool) {
                return Boolean.toString(bool);
            }
            return text(value);
        }

        private static boolean bool(Boolean value) {
            return value != null && value;
        }

        private static String text(Object value) {
            return value == null ? null : value.toString();
        }

        private static String firstNonBlank(String... values) {
            if (values == null) {
                return null;
            }
            for (String value : values) {
                if (hasText(value)) {
                    return value.trim();
                }
            }
            return null;
        }

        private static boolean hasText(String value) {
            return value != null && !value.isBlank();
        }

        private void printConfigError(String message) {
            if (json) {
                try {
                    System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                            "object", "gollek.agent_preflight_error",
                            "error", message)));
                    return;
                } catch (IOException ignored) {
                }
            }
            printError(message);
        }

        private void printRouteConfigError(AgentRouteConsistency.Result result) {
            if (json) {
                try {
                    System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(
                            result.toMetadata("gollek.agent_preflight_error")));
                    return;
                } catch (IOException ignored) {
                }
            }
            printError(result.errorMessage());
        }

        private static void printError(String message) {
            System.err.println(message);
        }

    }
}
