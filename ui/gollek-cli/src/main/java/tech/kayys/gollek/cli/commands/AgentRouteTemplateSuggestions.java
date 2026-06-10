package tech.kayys.gollek.cli.commands;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class AgentRouteTemplateSuggestions {
    private static final Map<String, String> ALIASES = Map.ofEntries(
            Map.entry("response", "responses"),
            Map.entry("response_api", "responses"),
            Map.entry("embedding", "embeddings"),
            Map.entry("embed", "embeddings"),
            Map.entry("embeds", "embeddings"),
            Map.entry("chatting", "chat"),
            Map.entry("chat_completion", "chat"),
            Map.entry("chat_completions", "chat"),
            Map.entry("default", "readiness_only"),
            Map.entry("route", "requested_route"),
            Map.entry("strict", "clean"),
            Map.entry("ci", "clean"));

    private AgentRouteTemplateSuggestions() {
    }

    static String suggestValue(String value, List<String> allowedValues) {
        if (!hasText(value) || allowedValues == null || allowedValues.isEmpty()) {
            return null;
        }
        String normalized = normalize(value);
        if (!hasText(normalized)) {
            return null;
        }

        Map<String, String> byNormalized = allowedByNormalizedName(allowedValues);
        String exact = byNormalized.get(normalized);
        if (exact != null) {
            return exact;
        }
        String alias = byNormalized.get(ALIASES.get(normalized));
        if (alias != null) {
            return alias;
        }

        String best = null;
        int bestDistance = Integer.MAX_VALUE;
        for (Map.Entry<String, String> entry : byNormalized.entrySet()) {
            int distance = distance(normalized, entry.getKey());
            if (distance < bestDistance) {
                bestDistance = distance;
                best = entry.getValue();
            }
        }
        return bestDistance <= suggestionThreshold(normalized) ? best : null;
    }

    private static Map<String, String> allowedByNormalizedName(List<String> allowedValues) {
        Map<String, String> out = new LinkedHashMap<>();
        for (String allowed : allowedValues) {
            if (hasText(allowed)) {
                out.put(normalize(allowed), allowed);
            }
        }
        return out;
    }

    private static int suggestionThreshold(String normalized) {
        if (normalized.length() <= 5) {
            return 1;
        }
        if (normalized.length() <= 12) {
            return 2;
        }
        return 3;
    }

    private static String normalize(String value) {
        if (!hasText(value)) {
            return null;
        }
        String normalized = value.trim()
                .replaceAll("([a-z0-9])([A-Z])", "$1_$2")
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_|_$", "");
        return normalized.isBlank() ? null : normalized;
    }

    private static int distance(String left, String right) {
        int[] previous = new int[right.length() + 1];
        int[] current = new int[right.length() + 1];
        for (int j = 0; j <= right.length(); j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= left.length(); i++) {
            current[0] = i;
            for (int j = 1; j <= right.length(); j++) {
                int substitution = left.charAt(i - 1) == right.charAt(j - 1) ? 0 : 1;
                current[j] = Math.min(
                        Math.min(previous[j] + 1, current[j - 1] + 1),
                        previous[j - 1] + substitution);
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[right.length()];
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
