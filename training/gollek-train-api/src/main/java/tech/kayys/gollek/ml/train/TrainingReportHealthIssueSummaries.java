package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainingReportHealthIssueSummaries {
    private TrainingReportHealthIssueSummaries() {
    }

    static List<Map<String, Object>> from(List<Map<String, Object>> issues) {
        if (issues == null || issues.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> summaries = new ArrayList<>();
        for (Map<String, Object> issue : issues) {
            if (issue != null && !issue.isEmpty()) {
                summaries.add(summary(issue));
            }
        }
        return List.copyOf(summaries);
    }

    private static Map<String, Object> summary(Map<String, Object> issue) {
        Map<String, Object> summary = new LinkedHashMap<>();
        putString(summary, "code", issue.get("code"));
        putString(summary, "severity", issue.get("severity"));
        putString(summary, "artifact", issue.get("artifact"));
        summary.put("blocking", booleanValue(issue.get("blocking")));
        putString(summary, "message", issue.get("message"));
        putString(summary, "action", issue.get("action"));
        String evidenceSummary = TrainingReportEvidenceSummary.compact(issue.get("evidence"));
        if (!evidenceSummary.isBlank()) {
            summary.put("evidenceSummary", evidenceSummary);
        }
        return Map.copyOf(summary);
    }

    private static void putString(Map<String, Object> target, String key, Object value) {
        String string = stringValue(value);
        if (!string.isBlank()) {
            target.put(key, string);
        }
    }

    private static boolean booleanValue(Object value) {
        return value instanceof Boolean bool && bool;
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }
}
