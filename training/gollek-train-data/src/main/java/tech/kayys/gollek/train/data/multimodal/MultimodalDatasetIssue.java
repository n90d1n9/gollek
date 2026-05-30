package tech.kayys.gollek.train.data.multimodal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Machine-readable validation issue for multimodal training datasets.
 */
public record MultimodalDatasetIssue(
        Severity severity,
        String code,
        String message,
        List<Integer> sampleIndices,
        Map<String, Object> details) {
    public MultimodalDatasetIssue {
        severity = Objects.requireNonNull(severity, "severity must not be null");
        code = requireText(code, "code").toUpperCase(Locale.ROOT);
        message = requireText(message, "message");
        sampleIndices = immutableSampleIndices(sampleIndices);
        details = immutableDetails(details);
    }

    public static MultimodalDatasetIssue error(
            String code,
            String message,
            List<Integer> sampleIndices,
            Map<String, Object> details) {
        return new MultimodalDatasetIssue(Severity.ERROR, code, message, sampleIndices, details);
    }

    public static MultimodalDatasetIssue warning(
            String code,
            String message,
            List<Integer> sampleIndices,
            Map<String, Object> details) {
        return new MultimodalDatasetIssue(Severity.WARNING, code, message, sampleIndices, details);
    }

    public boolean isError() {
        return severity == Severity.ERROR;
    }

    public boolean isWarning() {
        return severity == Severity.WARNING;
    }

    public enum Severity {
        ERROR,
        WARNING
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name + " must not be null");
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return trimmed;
    }

    private static List<Integer> immutableSampleIndices(List<Integer> indices) {
        Objects.requireNonNull(indices, "sampleIndices must not be null");
        List<Integer> copy = new ArrayList<>(indices.size());
        for (Integer index : indices) {
            if (index == null || index < 0) {
                throw new IllegalArgumentException("sampleIndices must not contain null or negative values");
            }
            copy.add(index);
        }
        return Collections.unmodifiableList(copy);
    }

    private static Map<String, Object> immutableDetails(Map<String, Object> details) {
        Objects.requireNonNull(details, "details must not be null");
        Map<String, Object> copy = new LinkedHashMap<>();
        details.forEach((key, value) -> copy.put(requireText(key, "details key"), value));
        return Collections.unmodifiableMap(copy);
    }
}
