package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

record TrainerHealthIssue(
        String kind,
        String code,
        String severity,
        boolean blocking,
        String artifact,
        String message,
        String action,
        Map<String, Object> evidence) {
    static final String ERROR = "error";
    static final String WARNING = "warning";

    TrainerHealthIssue {
        evidence = evidence == null ? Map.of() : Map.copyOf(evidence);
    }

    static TrainerHealthIssue warning(
            String kind,
            String code,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return issue(kind, code, WARNING, false, artifact, message, action, evidence);
    }

    static TrainerHealthIssue issue(
            String kind,
            String code,
            String severity,
            boolean blocking,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return new TrainerHealthIssue(kind, code, severity, blocking, artifact, message, action, evidence);
    }

    Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", kind);
        metadata.put("code", code);
        metadata.put("severity", severity);
        metadata.put("blocking", blocking);
        metadata.put("artifact", artifact);
        metadata.put("message", message);
        metadata.put("action", action);
        metadata.put("evidence", evidence);
        return Collections.unmodifiableMap(metadata);
    }

    static List<Map<String, Object>> toMetadata(List<TrainerHealthIssue> issues) {
        List<Map<String, Object>> metadata = new ArrayList<>();
        for (TrainerHealthIssue issue : issues) {
            metadata.add(issue.toMetadata());
        }
        return Collections.unmodifiableList(metadata);
    }

    static List<String> values(List<TrainerHealthIssue> issues, Function<TrainerHealthIssue, String> valueFn) {
        List<String> values = new ArrayList<>();
        for (TrainerHealthIssue issue : issues) {
            values.add(valueFn.apply(issue));
        }
        return Collections.unmodifiableList(values);
    }

    static List<String> distinctValues(List<TrainerHealthIssue> issues, Function<TrainerHealthIssue, String> valueFn) {
        List<String> values = new ArrayList<>();
        for (TrainerHealthIssue issue : issues) {
            String value = valueFn.apply(issue);
            if (!values.contains(value)) {
                values.add(value);
            }
        }
        return Collections.unmodifiableList(values);
    }
}
