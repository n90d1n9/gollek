package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * Prioritized runtime-performance plan derived from trainer profiling recommendations.
 */
public record TrainingReportRuntimeProfileActionPlan(
        Status status,
        List<Target> targets,
        List<String> nextActions) {
    public enum Status {
        NO_PROFILE,
        READY,
        NEEDS_OPTIMIZATION,
        BLOCKED
    }

    public enum TargetKind {
        HOTSPOT,
        GROUP,
        STABILITY,
        OTHER
    }

    public TrainingReportRuntimeProfileActionPlan {
        status = Objects.requireNonNull(status, "status must not be null");
        targets = targets == null ? List.of() : List.copyOf(targets);
        nextActions = nextActions == null ? List.of() : List.copyOf(nextActions);
    }

    public static TrainingReportRuntimeProfileActionPlan from(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        if (!report.runtimeProfile().available()) {
            return new TrainingReportRuntimeProfileActionPlan(Status.NO_PROFILE, List.of(), List.of());
        }
        List<Target> targets = report.actionPlan().recommendations().stream()
                .filter(TrainingReportRuntimeProfileActionPlan::runtimeProfileRecommendation)
                .map(Target::from)
                .sorted(TARGET_ORDER)
                .toList();
        return new TrainingReportRuntimeProfileActionPlan(status(targets), targets, nextActions(targets));
    }

    public boolean available() {
        return status != Status.NO_PROFILE;
    }

    public boolean requiresOptimization() {
        return status == Status.NEEDS_OPTIMIZATION || status == Status.BLOCKED;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("status", status.name());
        map.put("available", available());
        map.put("requiresOptimization", requiresOptimization());
        map.put("targetCount", targets.size());
        map.put("targets", targets.stream().map(Target::toMap).toList());
        map.put("nextActions", nextActions);
        return Map.copyOf(map);
    }

    private static final Comparator<Target> TARGET_ORDER = Comparator
            .comparingInt((Target target) -> priorityRank(target.priority()))
            .thenComparing(
                    target -> target.totalMillis().orElse(Double.NEGATIVE_INFINITY),
                    Comparator.reverseOrder())
            .thenComparing(Target::name);

    private static boolean runtimeProfileRecommendation(TrainingReportRecommendation recommendation) {
        return recommendation.diagnosticCode().startsWith("runtime_profile.");
    }

    private static Status status(List<Target> targets) {
        if (targets.isEmpty()) {
            return Status.READY;
        }
        if (targets.stream().anyMatch(target -> target.priority() == TrainingReportRecommendation.Priority.BLOCKER)) {
            return Status.BLOCKED;
        }
        return Status.NEEDS_OPTIMIZATION;
    }

    private static List<String> nextActions(List<Target> targets) {
        List<String> actions = new ArrayList<>();
        for (Target target : targets) {
            for (String action : target.actions()) {
                if (!actions.contains(action)) {
                    actions.add(action);
                }
            }
        }
        return List.copyOf(actions);
    }

    private static int priorityRank(TrainingReportRecommendation.Priority priority) {
        return switch (priority) {
            case BLOCKER -> 0;
            case HIGH -> 1;
            case MEDIUM -> 2;
            case LOW -> 3;
        };
    }

    public record Target(
            TargetKind kind,
            String name,
            TrainingReportRecommendation.Priority priority,
            TrainingReportRecommendation.Category category,
            String diagnosticCode,
            String title,
            OptionalDouble totalMillis,
            OptionalDouble percentTotal,
            List<String> actions,
            Map<String, Object> evidence) {
        public Target {
            kind = kind == null ? TargetKind.OTHER : kind;
            name = name == null || name.isBlank() ? "unknown" : name.trim();
            priority = Objects.requireNonNull(priority, "priority must not be null");
            category = Objects.requireNonNull(category, "category must not be null");
            diagnosticCode = diagnosticCode == null || diagnosticCode.isBlank()
                    ? "runtime_profile.unknown"
                    : diagnosticCode.trim();
            title = title == null || title.isBlank() ? "Review runtime profile" : title.trim();
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            percentTotal = percentTotal == null ? OptionalDouble.empty() : percentTotal;
            actions = actions == null ? List.of() : List.copyOf(actions);
            evidence = evidence == null ? Map.of() : Map.copyOf(evidence);
        }

        static Target from(TrainingReportRecommendation recommendation) {
            Map<String, Object> evidence = recommendation.evidence();
            return new Target(
                    kind(recommendation.diagnosticCode()),
                    targetName(recommendation, evidence),
                    recommendation.priority(),
                    recommendation.category(),
                    recommendation.diagnosticCode(),
                    recommendation.title(),
                    optionalDouble(evidence.get("totalMillis")),
                    optionalDouble(evidence.get("percentTotal")),
                    recommendation.actions(),
                    evidence);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("kind", kind.name());
            map.put("name", name);
            map.put("priority", priority.name());
            map.put("category", category.name());
            map.put("diagnosticCode", diagnosticCode);
            map.put("title", title);
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            percentTotal.ifPresent(value -> map.put("percentTotal", value));
            map.put("actions", actions);
            map.put("evidence", evidence);
            return Map.copyOf(map);
        }

        private static TargetKind kind(String diagnosticCode) {
            if (diagnosticCode.contains("hotspot")) {
                return TargetKind.HOTSPOT;
            }
            if (diagnosticCode.contains("variability")) {
                return TargetKind.STABILITY;
            }
            if (diagnosticCode.contains("group")) {
                return TargetKind.GROUP;
            }
            return TargetKind.OTHER;
        }

        private static String targetName(
                TrainingReportRecommendation recommendation,
                Map<String, Object> evidence) {
            Object phase = evidence.get("phase");
            if (phase instanceof String value && !value.isBlank()) {
                return value;
            }
            Object group = evidence.get("group");
            if (group instanceof String value && !value.isBlank()) {
                return value;
            }
            return recommendation.title();
        }

        private static OptionalDouble optionalDouble(Object value) {
            return value instanceof Number number
                    ? OptionalDouble.of(number.doubleValue())
                    : OptionalDouble.empty();
        }
    }
}
