package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

final class AgentRouteConsistency {

    static final String ERROR_CODE = "AGENT_ROUTE_CONFLICT";

    private static final String ERROR_PREFIX = "Agent request route conflict: ";
    private static final String ERROR_SUFFIX =
            ". Align the top-level route and nested request, or pass --model, --surface, or --feature-profile to override.";
    private static final String OVERRIDE_HINT =
            "Pass --model, --surface, or --feature-profile to intentionally override a saved template.";

    private AgentRouteConsistency() {
    }

    static Result validateAndNormalize(
            AgentRouteInput input,
            Map<String, Object> request,
            Overrides overrides,
            Route route) {
        List<Conflict> conflicts = new ArrayList<>();
        validateOrOverrideRouteField(
                conflicts,
                request,
                "model",
                overrides.model(),
                route.model(),
                input.modelId(),
                text(request.get("model")),
                value -> value == null ? null : value.trim());
        validateOrOverrideRouteField(
                conflicts,
                request,
                "surface",
                overrides.surface(),
                route.surface(),
                input.surface(),
                text(request.get("surface")),
                AgentRequestShape::normalizeSurface);
        validateOrOverrideFeatureProfile(
                conflicts,
                request,
                input,
                overrides.featureProfile(),
                route.featureProfile());
        return new Result(List.copyOf(conflicts));
    }

    private static void validateOrOverrideRouteField(
            List<Conflict> conflicts,
            Map<String, Object> request,
            String field,
            boolean hasCliOverride,
            String effectiveValue,
            String inputValue,
            String requestValue,
            UnaryOperator<String> normalizer) {
        if (!hasText(requestValue)) {
            return;
        }
        if (hasCliOverride) {
            request.put(field, effectiveValue);
            return;
        }
        String normalizedInput = normalizeForComparison(inputValue, normalizer);
        String normalizedRequest = normalizeForComparison(requestValue, normalizer);
        if (hasText(normalizedInput) && hasText(normalizedRequest) && !normalizedInput.equals(normalizedRequest)) {
            conflicts.add(new Conflict(field, normalizedInput, normalizedRequest));
        }
    }

    private static void validateOrOverrideFeatureProfile(
            List<Conflict> conflicts,
            Map<String, Object> request,
            AgentRouteInput input,
            boolean hasFeatureProfileOverride,
            String effectiveFeatureProfile) {
        String requestProfile = firstNonBlank(
                text(request.get("feature_profile")),
                text(request.get("featureProfile")));
        if (!hasText(requestProfile)) {
            return;
        }
        if (hasFeatureProfileOverride) {
            request.put("feature_profile", effectiveFeatureProfile);
            request.remove("featureProfile");
            return;
        }
        String normalizedInput = normalizeForComparison(
                input.featureProfile(),
                AgentServingFeatureProfile::normalizeName);
        String normalizedRequest = normalizeForComparison(
                requestProfile,
                AgentServingFeatureProfile::normalizeName);
        if (hasText(normalizedInput) && hasText(normalizedRequest) && !normalizedInput.equals(normalizedRequest)) {
            conflicts.add(new Conflict("feature_profile", normalizedInput, normalizedRequest));
        }
    }

    private static String normalizeForComparison(String value, UnaryOperator<String> normalizer) {
        if (!hasText(value)) {
            return null;
        }
        String normalized = normalizer.apply(value.trim());
        return hasText(normalized) ? normalized.trim() : null;
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

    record Overrides(boolean model, boolean surface, boolean featureProfile) {
    }

    record Route(String model, String surface, String featureProfile) {
    }

    record Conflict(String field, String topLevel, String request) {

        String message() {
            return field + " top-level=" + topLevel + " request=" + request;
        }

        Map<String, Object> toMetadata() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("field", field);
            out.put("top_level", topLevel);
            out.put("request", request);
            return out;
        }
    }

    record Result(List<Conflict> conflicts) {

        Result {
            conflicts = conflicts == null ? List.of() : List.copyOf(conflicts);
        }

        boolean ok() {
            return conflicts.isEmpty();
        }

        String errorMessage() {
            if (ok()) {
                return null;
            }
            return ERROR_PREFIX + String.join("; ", conflictMessages()) + ERROR_SUFFIX;
        }

        ConflictException toException() {
            return new ConflictException(this);
        }

        List<String> conflictMessages() {
            return conflicts.stream()
                    .map(Conflict::message)
                    .toList();
        }

        List<Map<String, Object>> conflictMetadata() {
            return conflicts.stream()
                    .map(Conflict::toMetadata)
                    .toList();
        }

        Map<String, Object> toMetadata(String object) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("object", object);
            out.put("code", ERROR_CODE);
            out.put("error", errorMessage());
            out.put("conflicts", conflictMetadata());
            out.put("override_hint", OVERRIDE_HINT);
            return out;
        }
    }

    static final class ConflictException extends IllegalArgumentException {
        private final Result result;

        ConflictException(Result result) {
            super(result.errorMessage());
            this.result = result;
        }

        Result result() {
            return result;
        }
    }
}
