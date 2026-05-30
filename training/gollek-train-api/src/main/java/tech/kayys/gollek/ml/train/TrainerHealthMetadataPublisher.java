package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

final class TrainerHealthMetadataPublisher {
    private TrainerHealthMetadataPublisher() {
    }

    static void putUnavailable(
            Map<String, Object> metadata,
            String prefix,
            String skipReason) {
        metadata.put(prefix + ".available", Boolean.FALSE);
        metadata.put(prefix + ".skipReason", skipReason);
        metadata.put(prefix + "Status", "unknown");
        metadata.put(prefix + "GatePassed", Boolean.TRUE);
    }

    static void put(
            Map<String, Object> metadata,
            String prefix,
            List<TrainerHealthIssue> issues) {
        boolean hasError = issues.stream().anyMatch(issue -> TrainerHealthIssue.ERROR.equals(issue.severity()));
        long warningCount = issues.stream().filter(issue -> TrainerHealthIssue.WARNING.equals(issue.severity())).count();
        long errorCount = issues.stream().filter(issue -> TrainerHealthIssue.ERROR.equals(issue.severity())).count();
        metadata.put(prefix + ".available", Boolean.TRUE);
        metadata.put(prefix + "Status", hasError ? "error" : issues.isEmpty() ? "healthy" : "warning");
        metadata.put(prefix + "Healthy", issues.isEmpty());
        metadata.put(prefix + "GatePassed", !hasError);
        metadata.put(prefix + "IssueDetected", !issues.isEmpty());
        metadata.put(prefix + "IssueCount", issues.size());
        metadata.put(prefix + "WarningCount", (int) warningCount);
        metadata.put(prefix + "ErrorCount", (int) errorCount);
        metadata.put(prefix + "IssueCodes", TrainerHealthIssue.values(issues, TrainerHealthIssue::code));
        metadata.put(prefix + "IssueSeverities", TrainerHealthIssue.distinctValues(issues, TrainerHealthIssue::severity));
        metadata.put(prefix + "RecommendedActions", TrainerHealthIssue.distinctValues(issues, TrainerHealthIssue::action));
        metadata.put(prefix + "Issues", TrainerHealthIssue.toMetadata(issues));
    }
}
