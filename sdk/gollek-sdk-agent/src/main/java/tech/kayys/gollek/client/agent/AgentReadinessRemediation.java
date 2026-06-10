package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Deduplicated remediation step derived from active readiness issue hints.
 */
public record AgentReadinessRemediation(
        String area,
        String code,
        AgentServingReadinessReport.Severity severity,
        String summary,
        String remediation,
        List<String> messages) {
    public AgentReadinessRemediation {
        area = blank(area) ? "general" : area.trim();
        severity = severity == null ? AgentServingReadinessReport.Severity.BLOCKING : severity;
        messages = distinctMessages(messages);
        String message = messages.isEmpty() ? summary : messages.get(0);
        code = AgentReadinessIssueCodes.resolve(code, area, severity, message);
        AgentReadinessIssueCodes.CatalogEntry catalog = AgentReadinessIssueCodes.describe(code);
        summary = blank(summary) ? catalog.summary() : summary.trim();
        remediation = blank(remediation) ? catalog.remediation() : remediation.trim();
    }

    static List<AgentReadinessRemediation> fromIssueHints(
            List<AgentServingReadinessReport.IssueHint> hints) {
        if (hints == null || hints.isEmpty()) {
            return List.of();
        }
        List<MutableStep> grouped = new ArrayList<>();
        for (AgentServingReadinessReport.IssueHint hint : hints) {
            if (hint == null) {
                continue;
            }
            MutableStep step = matchingStep(grouped, hint);
            if (step == null) {
                step = new MutableStep(
                        hint.area(),
                        hint.code(),
                        hint.severity(),
                        hint.message(),
                        hint.summary(),
                        hint.remediation());
                grouped.add(step);
            }
            step.addMessage(hint.message());
        }
        return grouped.stream()
                .map(MutableStep::toRemediation)
                .toList();
    }

    static List<AgentReadinessRemediation> fromIssueHintMaps(List<Map<String, Object>> hints) {
        if (hints == null || hints.isEmpty()) {
            return List.of();
        }
        List<AgentServingReadinessReport.IssueHint> typed = new ArrayList<>();
        for (Map<String, Object> hint : hints) {
            if (hint == null || hint.isEmpty()) {
                continue;
            }
            String code = text(hint.get("code"));
            String message = text(hint.get("message"));
            if (message == null && code == null) {
                continue;
            }
            if (message == null) {
                message = AgentReadinessIssueCodes.describe(code).summary();
            }
            typed.add(new AgentServingReadinessReport.IssueHint(
                    text(hint.get("area")),
                    severity(hint.getOrDefault("severity", hint.get("default_severity"))),
                    message,
                    code,
                    text(hint.get("default_severity")),
                    text(hint.get("summary")),
                    text(hint.get("remediation"))));
        }
        return fromIssueHints(typed);
    }

    static Map<String, List<AgentReadinessRemediation>> byArea(
            List<AgentReadinessRemediation> remediations) {
        return groupedBy(remediations, Grouping.AREA);
    }

    static Map<String, List<AgentReadinessRemediation>> byCode(
            List<AgentReadinessRemediation> remediations) {
        return groupedBy(remediations, Grouping.CODE);
    }

    static List<AgentReadinessRemediation> forCode(
            List<AgentReadinessRemediation> remediations,
            String code) {
        String normalized = AgentReadinessIssueCodes.normalize(code);
        if (normalized == null || remediations == null || remediations.isEmpty()) {
            return List.of();
        }
        List<AgentReadinessRemediation> out = new ArrayList<>();
        for (AgentReadinessRemediation remediation : remediations) {
            if (remediation != null && normalized.equals(remediation.code())) {
                out.add(remediation);
            }
        }
        return List.copyOf(out);
    }

    static List<AgentReadinessRemediation> blocking(List<AgentReadinessRemediation> remediations) {
        return bySeverity(remediations, AgentServingReadinessReport.Severity.BLOCKING);
    }

    static List<AgentReadinessRemediation> warning(List<AgentReadinessRemediation> remediations) {
        return bySeverity(remediations, AgentServingReadinessReport.Severity.WARNING);
    }

    static List<Map<String, Object>> toMaps(List<AgentReadinessRemediation> remediations) {
        if (remediations == null || remediations.isEmpty()) {
            return List.of();
        }
        return remediations.stream()
                .map(AgentReadinessRemediation::toMap)
                .toList();
    }

    static Map<String, List<Map<String, Object>>> toMapsByArea(
            List<AgentReadinessRemediation> remediations) {
        Map<String, List<AgentReadinessRemediation>> grouped = byArea(remediations);
        if (grouped.isEmpty()) {
            return Map.of();
        }
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        grouped.forEach((area, steps) -> out.put(area, toMaps(steps)));
        return Collections.unmodifiableMap(out);
    }

    static Map<String, List<Map<String, Object>>> toMapsByCode(
            List<AgentReadinessRemediation> remediations) {
        Map<String, List<AgentReadinessRemediation>> grouped = byCode(remediations);
        if (grouped.isEmpty()) {
            return Map.of();
        }
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        grouped.forEach((code, steps) -> out.put(code, toMaps(steps)));
        return Collections.unmodifiableMap(out);
    }

    public boolean blocking() {
        return severity == AgentServingReadinessReport.Severity.BLOCKING;
    }

    public boolean warning() {
        return severity == AgentServingReadinessReport.Severity.WARNING;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("area", area);
        out.put("code", code);
        out.put("severity", severity.name().toLowerCase(Locale.ROOT));
        out.put("summary", summary);
        out.put("remediation", remediation);
        out.put("messages", messages);
        return Collections.unmodifiableMap(out);
    }

    private static MutableStep matchingStep(
            List<MutableStep> steps,
            AgentServingReadinessReport.IssueHint hint) {
        for (MutableStep step : steps) {
            if (step.matches(hint)) {
                return step;
            }
        }
        return null;
    }

    private static List<String> distinctMessages(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (String value : values) {
            if (blank(value)) {
                continue;
            }
            String text = value.trim();
            if (!out.contains(text)) {
                out.add(text);
            }
        }
        return List.copyOf(out);
    }

    private static boolean blank(String value) {
        return value == null || value.isBlank();
    }

    private static List<AgentReadinessRemediation> bySeverity(
            List<AgentReadinessRemediation> remediations,
            AgentServingReadinessReport.Severity severity) {
        if (remediations == null || remediations.isEmpty()) {
            return List.of();
        }
        List<AgentReadinessRemediation> out = new ArrayList<>();
        for (AgentReadinessRemediation remediation : remediations) {
            if (remediation != null && remediation.severity() == severity) {
                out.add(remediation);
            }
        }
        return List.copyOf(out);
    }

    private static Map<String, List<AgentReadinessRemediation>> groupedBy(
            List<AgentReadinessRemediation> remediations,
            Grouping grouping) {
        if (remediations == null || remediations.isEmpty()) {
            return Map.of();
        }
        Map<String, List<AgentReadinessRemediation>> grouped = new LinkedHashMap<>();
        for (AgentReadinessRemediation remediation : remediations) {
            if (remediation == null) {
                continue;
            }
            String key = grouping == Grouping.CODE ? remediation.code() : remediation.area();
            grouped.computeIfAbsent(key, ignored -> new ArrayList<>())
                    .add(remediation);
        }
        Map<String, List<AgentReadinessRemediation>> out = new LinkedHashMap<>();
        grouped.forEach((key, steps) -> out.put(key, List.copyOf(steps)));
        return Collections.unmodifiableMap(out);
    }

    private enum Grouping {
        AREA,
        CODE
    }

    private static final class MutableStep {
        private final String area;
        private final String code;
        private final AgentServingReadinessReport.Severity severity;
        private final String summary;
        private final String remediation;
        private final List<String> messages = new ArrayList<>();

        private MutableStep(
                String area,
                String code,
                AgentServingReadinessReport.Severity severity,
                String message,
                String summary,
                String remediation) {
            this.area = blank(area) ? "general" : area.trim();
            this.severity = severity == null ? AgentServingReadinessReport.Severity.BLOCKING : severity;
            this.code = AgentReadinessIssueCodes.resolve(code, this.area, this.severity, message);
            AgentReadinessIssueCodes.CatalogEntry catalog = AgentReadinessIssueCodes.describe(this.code);
            this.summary = blank(summary) ? catalog.summary() : summary.trim();
            this.remediation = blank(remediation) ? catalog.remediation() : remediation.trim();
        }

        private boolean matches(AgentServingReadinessReport.IssueHint hint) {
            if (hint == null) {
                return false;
            }
            String hintArea = blank(hint.area()) ? "general" : hint.area().trim();
            AgentServingReadinessReport.Severity hintSeverity = hint.severity() == null
                    ? AgentServingReadinessReport.Severity.BLOCKING
                    : hint.severity();
            String hintCode = AgentReadinessIssueCodes.resolve(
                    hint.code(),
                    hintArea,
                    hintSeverity,
                    hint.message());
            AgentReadinessIssueCodes.CatalogEntry catalog = AgentReadinessIssueCodes.describe(hintCode);
            String hintSummary = blank(hint.summary()) ? catalog.summary() : hint.summary().trim();
            String hintRemediation = blank(hint.remediation()) ? catalog.remediation() : hint.remediation().trim();
            return area.equals(hintArea)
                    && code.equals(hintCode)
                    && severity == hintSeverity
                    && summary.equals(hintSummary)
                    && remediation.equals(hintRemediation);
        }

        private void addMessage(String message) {
            if (!blank(message)) {
                String text = message.trim();
                if (!messages.contains(text)) {
                    messages.add(text);
                }
            }
        }

        private AgentReadinessRemediation toRemediation() {
            return new AgentReadinessRemediation(area, code, severity, summary, remediation, messages);
        }
    }

    private static AgentServingReadinessReport.Severity severity(Object value) {
        String normalized = text(value);
        return "warning".equalsIgnoreCase(normalized)
                ? AgentServingReadinessReport.Severity.WARNING
                : AgentServingReadinessReport.Severity.BLOCKING;
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }
}
