package tech.kayys.gollek.ml.train;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A typed checkpoint resume issue before it is flattened into published summary metadata.
 */
record TrainerCheckpointResumeIssue(
        TrainerCheckpointResumeIssueKind kind,
        String artifact,
        String message) {

    TrainerCheckpointResumeIssue {
        kind = Objects.requireNonNull(kind, "kind must not be null");
        artifact = normalize(artifact);
        message = normalize(message);
    }

    String kindValue() {
        return kind.kind();
    }

    String code() {
        return kind.code();
    }

    String severity() {
        return kind.severity();
    }

    boolean blocking() {
        return kind.blocking();
    }

    String action() {
        return kind.action();
    }

    Map<String, Object> toMetadataMap() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        putNonBlank(metadata, "kind", kindValue());
        putNonBlank(metadata, "code", code());
        putNonBlank(metadata, "severity", severity());
        metadata.put("blocking", blocking());
        putNonBlank(metadata, "artifact", artifact);
        putNonBlank(metadata, "message", message);
        putNonBlank(metadata, "action", action());
        return Collections.unmodifiableMap(metadata);
    }

    String value(String key) {
        return switch (key) {
            case "kind" -> kindValue();
            case "code" -> code();
            case "severity" -> severity();
            case "artifact" -> artifact;
            case "message" -> message;
            case "action" -> action();
            default -> null;
        };
    }

    private static void putNonBlank(Map<String, Object> values, String key, String value) {
        if (value != null && !value.isBlank()) {
            values.put(key, value);
        }
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
