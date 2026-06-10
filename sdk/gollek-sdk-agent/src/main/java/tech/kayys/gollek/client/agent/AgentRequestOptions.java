package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Per-request options for agent-facing Gollek calls.
 *
 * <p>Use this to propagate trace/session identity from an external agent runtime
 * into Gollek's serving boundary without moving planning or tool execution into
 * the SDK.
 */
public record AgentRequestOptions(
        String requestId,
        String traceId,
        String sessionId,
        String userId,
        Map<String, String> headers) {

    public static final String REQUEST_ID_HEADER = "X-Gollek-Request-Id";
    public static final String TRACE_ID_HEADER = "X-Gollek-Trace-Id";
    public static final String SESSION_ID_HEADER = "X-Gollek-Session-Id";
    public static final String USER_ID_HEADER = "X-Gollek-User-Id";

    public AgentRequestOptions {
        headers = headers == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(headers));
    }

    public static AgentRequestOptions empty() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Map<String, String> toHeaders() {
        Map<String, String> out = new LinkedHashMap<>();
        putIfPresent(out, REQUEST_ID_HEADER, requestId);
        putIfPresent(out, TRACE_ID_HEADER, traceId);
        putIfPresent(out, SESSION_ID_HEADER, sessionId);
        putIfPresent(out, USER_ID_HEADER, userId);
        headers.forEach((name, value) -> putIfPresent(out, name, value));
        return out;
    }

    private static void putIfPresent(Map<String, String> out, String name, String value) {
        if (name != null && !name.isBlank() && value != null && !value.isBlank()) {
            out.put(name, value);
        }
    }

    public static final class Builder {
        private String requestId;
        private String traceId;
        private String sessionId;
        private String userId;
        private final Map<String, String> headers = new LinkedHashMap<>();

        private Builder() {
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder traceId(String traceId) {
            this.traceId = traceId;
            return this;
        }

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder header(String name, String value) {
            putIfPresent(headers, name, value);
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            if (headers != null) {
                headers.forEach(this::header);
            }
            return this;
        }

        public AgentRequestOptions build() {
            return new AgentRequestOptions(requestId, traceId, sessionId, userId, headers);
        }
    }
}
