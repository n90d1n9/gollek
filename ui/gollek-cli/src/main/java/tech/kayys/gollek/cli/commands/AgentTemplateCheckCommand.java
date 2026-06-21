package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;
import tech.kayys.gollek.client.agent.AgentServingPreflightPolicy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Command(
        name = "template-check",
        aliases = { "check-template", "route-check" },
        description = "Check a saved agent route template locally without HTTP, model invocation, tool execution, or retrieval")
public class AgentTemplateCheckCommand implements Callable<Integer> {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String OBJECT = "gollek.agent_template_check";
    private static final String ERROR_OBJECT = "gollek.agent_template_check_error";
    private static final String INVALID_CODE = "AGENT_ROUTE_TEMPLATE_INVALID";
    private static final String MODEL_MISSING_CODE = "AGENT_TEMPLATE_MODEL_MISSING";
    private static final String SURFACE_UNSUPPORTED_CODE = "AGENT_TEMPLATE_SURFACE_UNSUPPORTED";
    private static final String FEATURE_PROFILE_UNSUPPORTED_CODE = "AGENT_TEMPLATE_FEATURE_PROFILE_UNSUPPORTED";
    private static final String PREFLIGHT_POLICY_UNSUPPORTED_CODE = "AGENT_TEMPLATE_PREFLIGHT_POLICY_UNSUPPORTED";
    private static final String DEFAULT_SURFACE = "chat";
    private static final List<String> SUPPORTED_SURFACES = List.of("chat", "responses", "embeddings");

    @ParentCommand
    AgentCommand parentCommand;

    @Option(names = { "-m", "--model" }, description = "Model id used to override the saved route template")
    String modelId;

    @Option(names = { "--surface" }, description = "Compatibility surface override: chat, responses, or embeddings")
    String surface;

    @Option(names = { "--feature-profile" }, description = "Serving feature profile override")
    String featureProfile;

    @Option(names = { "--request-json" }, description = "Inline OpenAI-compatible request JSON, or a full route template")
    String requestJson;

    @Option(names = { "--request-file" }, description = "Path to OpenAI-compatible request JSON, or a full route template")
    String requestFile;

    @Option(names = { "--json" }, description = "Print the local template-check result as JSON")
    boolean json;

    @Override
    public Integer call() {
        try {
            if (parentCommand != null && parentCommand.root() != null) {
                parentCommand.root().bootstrapInheritedEnvironment();
            }

            TemplateCheckResult result = checkTemplate();
            if (json) {
                System.out.println(MAPPER.writerWithDefaultPrettyPrinter()
                        .writeValueAsString(result.toMetadata()));
            } else {
                printHumanReport(result);
            }
            return result.valid() ? 0 : 1;
        } catch (IOException e) {
            printConfigError(e.getMessage());
            return 2;
        } catch (IllegalArgumentException e) {
            printConfigError(e.getMessage());
            return 2;
        }
    }

    private TemplateCheckResult checkTemplate() throws IOException {
        AgentRouteInput input = loadInput();
        Map<String, Object> request = new LinkedHashMap<>(input.request());
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
            return TemplateCheckResult.routeConflict(routeConsistency);
        }

        List<TemplateCheckIssue> issues = new ArrayList<>();
        if (!hasText(effectiveModel)) {
            issues.add(TemplateCheckIssue.of(
                    MODEL_MISSING_CODE,
                    "model",
                    null,
                    "missing model: provide --model or include model/model_id in the route template",
                    null,
                    List.of()));
        } else {
            request.put("model", effectiveModel);
        }
        if (!SUPPORTED_SURFACES.contains(effectiveSurface)) {
            String suggestedValue = AgentRouteTemplateSuggestions.suggestValue(effectiveSurface, SUPPORTED_SURFACES);
            issues.add(TemplateCheckIssue.of(
                    SURFACE_UNSUPPORTED_CODE,
                    "surface",
                    effectiveSurface,
                    "unsupported surface: " + printable(effectiveSurface)
                            + ". Use one of: " + String.join(", ", SUPPORTED_SURFACES),
                    suggestedValue,
                    SUPPORTED_SURFACES));
        }
        if (AgentServingFeatureProfile.find(effectiveFeatureProfile).isEmpty()) {
            List<String> allowedValues = AgentServingFeatureProfile.supportedProfileNames();
            String suggestedValue = AgentRouteTemplateSuggestions.suggestValue(effectiveFeatureProfile, allowedValues);
            issues.add(TemplateCheckIssue.of(
                    FEATURE_PROFILE_UNSUPPORTED_CODE,
                    "feature_profile",
                    effectiveFeatureProfile,
                    "unsupported feature profile: " + printable(effectiveFeatureProfile)
                            + ". Use one of: " + String.join(", ", allowedValues),
                    suggestedValue,
                    allowedValues));
        }
        AgentServingPreflightPolicy preflightPolicy = null;
        try {
            preflightPolicy = AgentServingPreflightPolicy.requireProfile(input.preflightPolicy());
        } catch (IllegalArgumentException e) {
            List<String> allowedValues = AgentServingPreflightPolicy.supportedProfileNames();
            String suggestedValue = AgentRouteTemplateSuggestions.suggestValue(input.preflightPolicy(), allowedValues);
            issues.add(TemplateCheckIssue.of(
                    PREFLIGHT_POLICY_UNSUPPORTED_CODE,
                    "preflight_policy",
                    input.preflightPolicy(),
                    e.getMessage(),
                    suggestedValue,
                    allowedValues));
        }
        request.put("surface", effectiveSurface);
        request.put("feature_profile", effectiveFeatureProfile);
        request.remove("featureProfile");

        return new TemplateCheckResult(
                issues.isEmpty(),
                effectiveModel,
                effectiveSurface,
                effectiveFeatureProfile,
                preflightPolicy == null ? Map.of() : preflightPolicy.toMetadata(),
                request,
                issues,
                null);
    }

    private AgentRouteInput loadInput() throws IOException {
        if (hasText(requestJson) && hasText(requestFile)) {
            throw new IllegalArgumentException("Use either --request-json or --request-file, not both.");
        }
        if (hasText(requestJson)) {
            return AgentRouteInput.parse(requestJson.trim(), MAPPER, "Agent template");
        }
        if (hasText(requestFile)) {
            return AgentRouteInput.parse(
                    Files.readString(Path.of(requestFile.trim())),
                    MAPPER,
                    "Agent template");
        }
        throw new IllegalArgumentException(
                "Provide --request-json or --request-file with a saved agent route template.");
    }

    private void printHumanReport(TemplateCheckResult result) {
        System.out.println("Gollek agent template check: " + (result.valid() ? "valid" : "invalid"));
        if (result.routeConflict() != null) {
            System.out.println("Route conflict: " + result.routeConflict().errorMessage()
                    + " [code: " + AgentRouteConsistency.ERROR_CODE + "]");
        } else {
            System.out.println("Route: model=" + printable(result.model())
                    + ", surface=" + printable(result.surface())
                    + ", feature_profile=" + printable(result.featureProfile()));
            if (!result.preflightPolicy().isEmpty()) {
                System.out.println("Preflight policy: " + printable(text(result.preflightPolicy().get("profile"))));
            }
            Map<String, Object> request = requestSummary(result.request());
            System.out.println("Request: parameters=" + String.join(",", stringList(request.get("parameters")))
                    + ", messages=" + request.get("messages")
                    + ", inputs=" + request.get("inputs")
                    + ", tools=" + request.get("tools")
                    + ", rag_context=" + request.get("rag_context"));
            printIssues(result.issues());
        }
        printBoundary();
    }

    private void printIssues(List<TemplateCheckIssue> issues) {
        if (issues == null || issues.isEmpty()) {
            return;
        }
        System.out.println("Template issues:");
        for (TemplateCheckIssue issue : issues) {
            System.out.println("- " + issue.message() + " [code: " + issue.code() + "]"
                    + issueHint(issue));
        }
    }

    private static String issueHint(TemplateCheckIssue issue) {
        List<String> parts = new ArrayList<>();
        if (hasText(issue.field())) {
            parts.add("field: " + issue.field());
        }
        if (hasText(issue.value())) {
            parts.add("value: " + issue.value());
        }
        if (hasText(issue.suggestedValue())) {
            parts.add("suggested: " + issue.suggestedValue());
        }
        if (!issue.allowedValues().isEmpty()) {
            parts.add("allowed: " + String.join("|", issue.allowedValues()));
        }
        return parts.isEmpty() ? "" : " [" + String.join(", ", parts) + "]";
    }

    private void printBoundary() {
        System.out.println("Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no");
    }

    private void printConfigError(String message) {
        if (json) {
            try {
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("object", ERROR_OBJECT);
                out.put("valid", false);
                out.put("code", INVALID_CODE);
                out.put("error", message);
                out.put("boundary", boundaryMetadata());
                System.out.println(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
                return;
            } catch (IOException ignored) {
            }
        }
        System.err.println("Agent template check failed: " + message);
    }

    private static Map<String, Object> routeMetadata(
            String model,
            String surface,
            String featureProfile) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", model);
        out.put("surface", surface);
        out.put("feature_profile", featureProfile);
        return out;
    }

    private static Map<String, Object> requestSummary(Map<String, Object> request) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("parameters", request == null
                ? List.of()
                : request.keySet().stream().map(Object::toString).sorted().toList());
        out.put("messages", countItems(value(request, "messages")));
        out.put("inputs", countItems(value(request, "input")));
        out.put("tools", countItems(value(request, "tools")));
        out.put("rag_context", countItems(firstNonNull(
                value(request, "rag_context"),
                value(request, "ragContext"),
                value(request, "context"))));
        return out;
    }

    private static Object value(Map<String, Object> request, String key) {
        return request == null ? null : request.get(key);
    }

    private static Object firstNonNull(Object... values) {
        if (values == null) {
            return null;
        }
        for (Object value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static int countItems(Object value) {
        if (value instanceof List<?> list) {
            return list.size();
        }
        if (value instanceof Map<?, ?> map) {
            return map.isEmpty() ? 0 : 1;
        }
        if (value instanceof String text) {
            return text.isBlank() ? 0 : 1;
        }
        return value == null ? 0 : 1;
    }

    private static Map<String, Object> boundaryMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("local_only", true);
        out.put("http_request", false);
        out.put("model_invoked", false);
        out.put("tool_execution", false);
        out.put("retrieval_execution", false);
        out.put("workflow_state", false);
        return out;
    }

    private static String printable(String value) {
        return hasText(value) ? value : "n/a";
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

    private static List<String> stringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (Object item : list) {
            if (item != null) {
                out.add(item.toString());
            }
        }
        return List.copyOf(out);
    }

    private record TemplateCheckIssue(
            String code,
            String field,
            String value,
            String message,
            String suggestedValue,
            List<String> allowedValues) {

        private static TemplateCheckIssue of(
                String code,
                String field,
                String value,
                String message,
                String suggestedValue,
                List<String> allowedValues) {
            return new TemplateCheckIssue(code, field, value, message, suggestedValue, allowedValues);
        }

        private TemplateCheckIssue {
            allowedValues = allowedValues == null ? List.of() : List.copyOf(allowedValues);
        }

        private Map<String, Object> toMetadata() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("code", code);
            out.put("field", field);
            if (value != null) {
                out.put("value", value);
            }
            out.put("message", message);
            if (suggestedValue != null) {
                out.put("suggested_value", suggestedValue);
            }
            if (!allowedValues.isEmpty()) {
                out.put("allowed_values", allowedValues);
            }
            return out;
        }
    }

    private record TemplateCheckResult(
            boolean valid,
            String model,
            String surface,
            String featureProfile,
            Map<String, Object> preflightPolicy,
            Map<String, Object> request,
            List<TemplateCheckIssue> issues,
            AgentRouteConsistency.Result routeConflict) {

        private static TemplateCheckResult routeConflict(AgentRouteConsistency.Result result) {
            return new TemplateCheckResult(
                    false,
                    null,
                    null,
                    null,
                    Map.of(),
                    Map.of(),
                    List.of(),
                    result);
        }

        private TemplateCheckResult {
            preflightPolicy = preflightPolicy == null ? Map.of() : new LinkedHashMap<>(preflightPolicy);
            request = request == null ? Map.of() : new LinkedHashMap<>(request);
            issues = issues == null ? List.of() : List.copyOf(issues);
        }

        private List<String> issueMessages() {
            return issues.stream()
                    .map(TemplateCheckIssue::message)
                    .toList();
        }

        private List<Map<String, Object>> issueDetails() {
            return issues.stream()
                    .map(TemplateCheckIssue::toMetadata)
                    .toList();
        }

        private Map<String, Object> toMetadata() {
            if (routeConflict != null) {
                Map<String, Object> out = new LinkedHashMap<>(routeConflict.toMetadata(OBJECT));
                out.put("valid", false);
                out.put("issue_details", List.of(Map.of(
                        "code", AgentRouteConsistency.ERROR_CODE,
                        "field", "route",
                        "message", routeConflict.errorMessage())));
                out.put("boundary", boundaryMetadata());
                return out;
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("object", OBJECT);
            out.put("valid", valid);
            if (!valid) {
                List<String> messages = issueMessages();
                out.put("code", INVALID_CODE);
                out.put("error", messages.isEmpty() ? "agent route template is invalid" : messages.get(0));
            }
            out.put("route", routeMetadata(model, surface, featureProfile));
            out.put("preflight_policy", preflightPolicy);
            out.put("request", requestSummary(request));
            out.put("issues", issueMessages());
            out.put("issue_details", issueDetails());
            out.put("boundary", boundaryMetadata());
            return out;
        }
    }
}
