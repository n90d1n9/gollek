package tech.kayys.gollek.train.data.multimodal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Result of validating a multimodal dataset before a trainer consumes it.
 */
public record MultimodalValidationResult(
        MultimodalDatasetReport report,
        List<MultimodalDatasetIssue> issues) {
    public MultimodalValidationResult {
        report = Objects.requireNonNull(report, "report must not be null");
        Objects.requireNonNull(issues, "issues must not be null");
        List<MultimodalDatasetIssue> copy = new ArrayList<>(issues.size());
        for (MultimodalDatasetIssue issue : issues) {
            copy.add(Objects.requireNonNull(issue, "issues must not contain null"));
        }
        issues = Collections.unmodifiableList(copy);
    }

    public boolean isValid() {
        return !hasErrors();
    }

    public boolean hasErrors() {
        return issues.stream().anyMatch(MultimodalDatasetIssue::isError);
    }

    public boolean hasWarnings() {
        return issues.stream().anyMatch(MultimodalDatasetIssue::isWarning);
    }

    public List<MultimodalDatasetIssue> errors() {
        return filtered(MultimodalDatasetIssue.Severity.ERROR);
    }

    public List<MultimodalDatasetIssue> warnings() {
        return filtered(MultimodalDatasetIssue.Severity.WARNING);
    }

    public boolean hasIssue(String code) {
        return !issues(code).isEmpty();
    }

    public List<MultimodalDatasetIssue> issues(String code) {
        String normalized = normalizeCode(code);
        return issues.stream()
                .filter(issue -> issue.code().equals(normalized))
                .toList();
    }

    public void throwIfInvalid() {
        if (hasErrors()) {
            throw new IllegalStateException(summary());
        }
    }

    public String summary() {
        long errorCount = errors().size();
        long warningCount = warnings().size();
        StringBuilder builder = new StringBuilder()
                .append("Multimodal dataset validation found ")
                .append(errorCount)
                .append(errorCount == 1 ? " error" : " errors")
                .append(" and ")
                .append(warningCount)
                .append(warningCount == 1 ? " warning" : " warnings");
        if (!issues.isEmpty()) {
            builder.append(": ");
            builder.append(issues.stream()
                    .map(issue -> issue.code() + "=" + issue.sampleIndices())
                    .toList());
        }
        return builder.toString();
    }

    private List<MultimodalDatasetIssue> filtered(MultimodalDatasetIssue.Severity severity) {
        return issues.stream()
                .filter(issue -> issue.severity() == severity)
                .toList();
    }

    private static String normalizeCode(String code) {
        Objects.requireNonNull(code, "code must not be null");
        String normalized = code.trim().toUpperCase(Locale.ROOT);
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("code must not be blank");
        }
        return normalized;
    }
}
