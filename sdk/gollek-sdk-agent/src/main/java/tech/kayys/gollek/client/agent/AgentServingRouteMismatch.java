package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One field-level difference between a requested and selected serving route.
 *
 * <p>This is diagnostic metadata only. It does not choose routes, retry
 * requests, execute tools, run retrieval, update memory, or invoke a model.
 */
public record AgentServingRouteMismatch(String field, String requested, String selected) {

    public AgentServingRouteMismatch {
        field = text(field);
        requested = text(requested);
        selected = text(selected);
    }

    public String message() {
        String label = field == null ? "route" : field;
        return label + " requested=" + display(requested) + " selected=" + display(selected);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> out = new LinkedHashMap<>();
        putIfPresent(out, "field", field);
        putIfPresent(out, "requested", requested);
        putIfPresent(out, "selected", selected);
        out.put("message", message());
        return Collections.unmodifiableMap(out);
    }

    private static void putIfPresent(Map<String, Object> out, String key, String value) {
        if (value != null) {
            out.put(key, value);
        }
    }

    private static String display(String value) {
        return value == null ? "n/a" : value;
    }

    private static String text(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.trim();
    }
}
