package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * High-level action plan derived from canonical trainer diagnostics.
 */
public record TrainingReportActionPlan(
        TrainingReportDiagnostics.Summary diagnosticSummary,
        List<TrainingReportRecommendation> recommendations) {
    public enum Status {
        READY,
        NEEDS_ATTENTION,
        BLOCKED
    }

    public TrainingReportActionPlan {
        diagnosticSummary = Objects.requireNonNull(diagnosticSummary, "diagnosticSummary must not be null");
        recommendations = recommendations == null ? List.of() : List.copyOf(recommendations);
    }

    public Status status() {
        if (hasBlockers() || diagnosticSummary.hasCritical()) {
            return Status.BLOCKED;
        }
        if (!recommendations.isEmpty() || diagnosticSummary.hasWarnings() || diagnosticSummary.hasInfo()) {
            return Status.NEEDS_ATTENTION;
        }
        return Status.READY;
    }

    public boolean ready() {
        return status() == Status.READY;
    }

    public static TrainingReportActionPlan fromMap(Map<String, ?> map) {
        if (map == null || map.isEmpty()) {
            return new TrainingReportActionPlan(
                    TrainingReportDiagnostics.Summary.fromMap(Map.of()),
                    List.of());
        }
        return new TrainingReportActionPlan(
                TrainingReportDiagnostics.Summary.fromMap(TrainingReportValues.mapValue(map, "diagnostics")),
                recommendationsFromObject(map.get("recommendations")));
    }

    public boolean requiresAttention() {
        return status() != Status.READY;
    }

    public boolean hasBlockers() {
        return recommendations.stream().anyMatch(TrainingReportRecommendation::blocksPromotion);
    }

    public List<TrainingReportRecommendation> blockers() {
        return recommendations.stream()
                .filter(TrainingReportRecommendation::blocksPromotion)
                .toList();
    }

    public List<TrainingReportRecommendation> byPriority(TrainingReportRecommendation.Priority priority) {
        Objects.requireNonNull(priority, "priority must not be null");
        return recommendations.stream()
                .filter(recommendation -> recommendation.priority() == priority)
                .toList();
    }

    public List<String> actionItems() {
        return recommendations.stream()
                .flatMap(recommendation -> recommendation.actions().stream())
                .distinct()
                .toList();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("status", status().name());
        map.put("ready", ready());
        map.put("requiresAttention", requiresAttention());
        map.put("hasBlockers", hasBlockers());
        map.put("diagnostics", diagnosticSummary.toMap());
        map.put("recommendations", recommendations.stream()
                .map(TrainingReportRecommendation::toMap)
                .toList());
        map.put("actionItems", actionItems());
        return Map.copyOf(map);
    }

    private static List<TrainingReportRecommendation> recommendationsFromObject(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<TrainingReportRecommendation> recommendations = new java.util.ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
                if (snapshot instanceof Map<?, ?> snapshotMap) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> typedMap = (Map<String, Object>) snapshotMap;
                    recommendations.add(TrainingReportRecommendation.fromMap(typedMap));
                }
            }
        }
        return List.copyOf(recommendations);
    }
}
