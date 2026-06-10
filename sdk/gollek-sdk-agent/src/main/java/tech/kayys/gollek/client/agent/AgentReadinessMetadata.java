package tech.kayys.gollek.client.agent;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shared metadata helpers for Gollek agent-serving readiness reports.
 *
 * <p>The helper only derives metadata from already-built check maps. It does
 * not run discovery, validate requests, execute tools, run retrieval, or invoke
 * a model.
 */
public final class AgentReadinessMetadata {
    public static final String OBJECT_READINESS_REPORT = "gollek.agent_readiness_report";
    public static final String OBJECT_PREFLIGHT = "gollek.agent_preflight";

    private AgentReadinessMetadata() {
    }

    public static Map<String, Object> fromChecks(Map<String, ?> checks) {
        return fromChecks(checks, false);
    }

    public static Map<String, Object> fromChecks(Map<String, ?> checks, boolean prefixFlatMessages) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("checked_areas", checkedAreas(checks));
        metadata.put("ready_check_count", countChecks(checks, "ready"));
        metadata.put("blocked_check_count", countChecks(checks, "blocked"));
        metadata.put("skipped_check_count", countChecks(checks, "skipped"));
        metadata.put("blocking_issue_count", messageCount(checks, "blocking_messages"));
        metadata.put("warning_count", messageCount(checks, "warning_messages"));
        metadata.put("issues_by_area", issuesByArea(checks));
        metadata.put("issue_codes_by_area", issueCodesByArea(checks));
        metadata.put("blocking_messages", flatMessages(checks, "blocking_messages", prefixFlatMessages));
        metadata.put("warning_messages", flatMessages(checks, "warning_messages", prefixFlatMessages));
        metadata.put("blocking_codes", flatCodes(checks, "blocking_messages", "blocking_codes"));
        metadata.put("warning_codes", flatCodes(checks, "warning_messages", "warning_codes"));
        List<Map<String, Object>> hints = issueHints(checks);
        metadata.put("issue_hints", hints);
        metadata.put("issue_hints_by_area", issueHintsByArea(hints));
        metadata.put("issue_hints_by_code", issueHintsByCode(hints));
        List<AgentReadinessRemediation> remediations = AgentReadinessRemediation.fromIssueHintMaps(hints);
        metadata.put("remediation_plan", AgentReadinessRemediation.toMaps(remediations));
        metadata.put("blocking_remediation_plan",
                AgentReadinessRemediation.toMaps(AgentReadinessRemediation.blocking(remediations)));
        metadata.put("warning_remediation_plan",
                AgentReadinessRemediation.toMaps(AgentReadinessRemediation.warning(remediations)));
        metadata.put("remediation_plan_by_area", AgentReadinessRemediation.toMapsByArea(remediations));
        metadata.put("remediation_plan_by_code", AgentReadinessRemediation.toMapsByCode(remediations));
        return Collections.unmodifiableMap(metadata);
    }

    public static List<String> checkedAreas(Map<String, ?> checks) {
        if (checks == null || checks.isEmpty()) {
            return List.of();
        }
        List<String> areas = new ArrayList<>();
        for (Object key : checks.keySet()) {
            String area = text(key);
            if (area != null && !areas.contains(area)) {
                areas.add(area);
            }
        }
        return List.copyOf(areas);
    }

    public static long countChecks(Map<String, ?> checks, String status) {
        if (checks == null || checks.isEmpty()) {
            return 0L;
        }
        long count = 0L;
        for (Object value : checks.values()) {
            Map<String, Object> check = objectMap(value);
            if (status.equals(check.get("status"))) {
                count++;
            }
        }
        return count;
    }

    public static Map<String, List<String>> issuesByArea(Map<String, ?> checks) {
        if (checks == null || checks.isEmpty()) {
            return Map.of();
        }
        Map<String, List<String>> grouped = new LinkedHashMap<>();
        for (Map.Entry<String, ?> entry : normalizedEntries(checks)) {
            List<String> messages = new ArrayList<>();
            Map<String, Object> check = objectMap(entry.getValue());
            messages.addAll(stringList(check.get("blocking_messages")));
            messages.addAll(stringList(check.get("warning_messages")));
            if (!messages.isEmpty()) {
                grouped.put(entry.getKey(), List.copyOf(messages));
            }
        }
        return Collections.unmodifiableMap(grouped);
    }

    public static Map<String, List<String>> issueCodesByArea(Map<String, ?> checks) {
        if (checks == null || checks.isEmpty()) {
            return Map.of();
        }
        Map<String, List<String>> grouped = new LinkedHashMap<>();
        for (Map.Entry<String, ?> entry : normalizedEntries(checks)) {
            Map<String, Object> check = objectMap(entry.getValue());
            List<String> codes = new ArrayList<>();
            codes.addAll(codesFor(entry.getKey(), "blocking", check, "blocking_messages", "blocking_codes"));
            codes.addAll(codesFor(entry.getKey(), "warning", check, "warning_messages", "warning_codes"));
            if (!codes.isEmpty()) {
                grouped.put(entry.getKey(), List.copyOf(codes));
            }
        }
        return Collections.unmodifiableMap(grouped);
    }

    public static List<String> blockingMessages(Map<String, ?> checks) {
        return flatMessages(checks, "blocking_messages", false);
    }

    public static List<String> warningMessages(Map<String, ?> checks) {
        return flatMessages(checks, "warning_messages", false);
    }

    public static List<String> blockingCodes(Map<String, ?> checks) {
        return flatCodes(checks, "blocking_messages", "blocking_codes");
    }

    public static List<String> warningCodes(Map<String, ?> checks) {
        return flatCodes(checks, "warning_messages", "warning_codes");
    }

    public static List<Map<String, Object>> issueHints(Map<String, ?> checks) {
        if (checks == null || checks.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> hints = new ArrayList<>();
        for (Map.Entry<String, ?> entry : normalizedEntries(checks)) {
            Map<String, Object> check = objectMap(entry.getValue());
            List<Map<String, Object>> suppliedHints = mapList(check.get("issue_hints"));
            if (!suppliedHints.isEmpty()) {
                hints.addAll(suppliedHints);
                continue;
            }
            hints.addAll(issueHintsFor(
                    entry.getKey(),
                    AgentServingReadinessReport.Severity.BLOCKING,
                    check,
                    "blocking_messages",
                    "blocking_codes"));
            hints.addAll(issueHintsFor(
                    entry.getKey(),
                    AgentServingReadinessReport.Severity.WARNING,
                    check,
                    "warning_messages",
                    "warning_codes"));
        }
        return List.copyOf(hints);
    }

    public static Map<String, List<Map<String, Object>>> issueHintsByArea(Map<String, ?> checks) {
        return issueHintsByArea(issueHints(checks));
    }

    public static Map<String, List<Map<String, Object>>> issueHintsByArea(List<Map<String, Object>> hints) {
        return groupedIssueHints(hints, "area", "general");
    }

    public static Map<String, List<Map<String, Object>>> issueHintsByCode(Map<String, ?> checks) {
        return issueHintsByCode(issueHints(checks));
    }

    public static Map<String, List<Map<String, Object>>> issueHintsByCode(List<Map<String, Object>> hints) {
        if (hints == null || hints.isEmpty()) {
            return Map.of();
        }
        Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
        for (Map<String, Object> hint : hints) {
            String code = AgentReadinessIssueCodes.normalize(text(hint.get("code")));
            if (code != null) {
                grouped.computeIfAbsent(code, ignored -> new ArrayList<>())
                        .add(hint);
            }
        }
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        grouped.forEach((code, codeHints) -> out.put(code, List.copyOf(codeHints)));
        return Collections.unmodifiableMap(out);
    }

    public static List<Map<String, Object>> remediationPlan(Map<String, ?> checks) {
        return remediationPlanFromHints(issueHints(checks));
    }

    public static Map<String, List<Map<String, Object>>> remediationPlanByArea(Map<String, ?> checks) {
        return remediationPlanByAreaFromHints(issueHints(checks));
    }

    public static Map<String, List<Map<String, Object>>> remediationPlanByCode(Map<String, ?> checks) {
        return remediationPlanByCodeFromHints(issueHints(checks));
    }

    public static List<Map<String, Object>> blockingRemediationPlan(Map<String, ?> checks) {
        return blockingRemediationPlanFromHints(issueHints(checks));
    }

    public static List<Map<String, Object>> warningRemediationPlan(Map<String, ?> checks) {
        return warningRemediationPlanFromHints(issueHints(checks));
    }

    static List<Map<String, Object>> remediationPlanFromHints(List<Map<String, Object>> hints) {
        return AgentReadinessRemediation.toMaps(AgentReadinessRemediation.fromIssueHintMaps(hints));
    }

    static List<Map<String, Object>> blockingRemediationPlanFromHints(List<Map<String, Object>> hints) {
        return AgentReadinessRemediation.toMaps(
                AgentReadinessRemediation.blocking(AgentReadinessRemediation.fromIssueHintMaps(hints)));
    }

    static List<Map<String, Object>> warningRemediationPlanFromHints(List<Map<String, Object>> hints) {
        return AgentReadinessRemediation.toMaps(
                AgentReadinessRemediation.warning(AgentReadinessRemediation.fromIssueHintMaps(hints)));
    }

    static Map<String, List<Map<String, Object>>> remediationPlanByAreaFromHints(
            List<Map<String, Object>> hints) {
        return AgentReadinessRemediation.toMapsByArea(AgentReadinessRemediation.fromIssueHintMaps(hints));
    }

    static Map<String, List<Map<String, Object>>> remediationPlanByCodeFromHints(
            List<Map<String, Object>> hints) {
        return AgentReadinessRemediation.toMapsByCode(AgentReadinessRemediation.fromIssueHintMaps(hints));
    }

    private static Map<String, List<Map<String, Object>>> groupedIssueHints(
            List<Map<String, Object>> hints,
            String field,
            String fallback) {
        if (hints == null || hints.isEmpty()) {
            return Map.of();
        }
        Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
        for (Map<String, Object> hint : hints) {
            String key = text(hint.get(field));
            if (key == null) {
                key = fallback;
            }
            grouped.computeIfAbsent(key, ignored -> new ArrayList<>())
                    .add(hint);
        }
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        grouped.forEach((key, keyHints) -> out.put(key, List.copyOf(keyHints)));
        return Collections.unmodifiableMap(out);
    }

    private static int messageCount(Map<String, ?> checks, String field) {
        return flatMessages(checks, field, false).size();
    }

    private static List<String> flatMessages(Map<String, ?> checks, String field, boolean prefixArea) {
        if (checks == null || checks.isEmpty()) {
            return List.of();
        }
        List<String> messages = new ArrayList<>();
        for (Map.Entry<String, ?> entry : normalizedEntries(checks)) {
            Map<String, Object> check = objectMap(entry.getValue());
            for (String message : stringList(check.get(field))) {
                messages.add(prefixArea ? entry.getKey() + ": " + message : message);
            }
        }
        return List.copyOf(messages);
    }

    private static List<String> flatCodes(Map<String, ?> checks, String messageField, String codeField) {
        if (checks == null || checks.isEmpty()) {
            return List.of();
        }
        List<String> codes = new ArrayList<>();
        String severity = codeField.startsWith("warning") ? "warning" : "blocking";
        for (Map.Entry<String, ?> entry : normalizedEntries(checks)) {
            codes.addAll(codesFor(entry.getKey(), severity, objectMap(entry.getValue()), messageField, codeField));
        }
        return List.copyOf(codes);
    }

    private static List<Map<String, Object>> issueHintsFor(
            String area,
            AgentServingReadinessReport.Severity severity,
            Map<String, Object> check,
            String messageField,
            String codeField) {
        List<String> messages = stringList(check.get(messageField));
        List<String> suppliedCodes = stringList(check.get(codeField));
        List<Map<String, Object>> hints = new ArrayList<>();
        for (int i = 0; i < messages.size(); i++) {
            hints.add(new AgentServingReadinessReport.IssueHint(
                    area,
                    severity,
                    messages.get(i),
                    itemAt(suppliedCodes, i),
                    null,
                    null,
                    null).toMap());
        }
        return List.copyOf(hints);
    }

    private static List<String> codesFor(
            String area,
            String severity,
            Map<String, Object> check,
            String messageField,
            String codeField) {
        List<String> messages = stringList(check.get(messageField));
        List<String> suppliedCodes = stringList(check.get(codeField));
        List<String> codes = new ArrayList<>();
        for (int i = 0; i < messages.size(); i++) {
            codes.add(AgentReadinessIssueCodes.resolve(itemAt(suppliedCodes, i), area, severity, messages.get(i)));
        }
        return List.copyOf(codes);
    }

    private static String itemAt(List<String> values, int index) {
        return values == null || index < 0 || index >= values.size() ? null : values.get(index);
    }

    private static List<Map.Entry<String, ?>> normalizedEntries(Map<String, ?> checks) {
        List<Map.Entry<String, ?>> entries = new ArrayList<>();
        for (Map.Entry<String, ?> entry : checks.entrySet()) {
            String key = text(entry.getKey());
            if (key != null) {
                entries.add(new AbstractMap.SimpleImmutableEntry<>(key, entry.getValue()));
            }
        }
        return entries;
    }

    private static List<String> stringList(Object value) {
        if (value instanceof List<?> list) {
            List<String> out = new ArrayList<>();
            for (Object item : list) {
                String text = text(item);
                if (text != null) {
                    out.add(text);
                }
            }
            return List.copyOf(out);
        }
        String text = text(value);
        return text == null ? List.of() : List.of(text);
    }

    private static List<Map<String, Object>> mapList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object item : list) {
            Map<String, Object> map = objectMap(item);
            if (!map.isEmpty()) {
                out.add(map);
            }
        }
        return List.copyOf(out);
    }

    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            String key = text(entry.getKey());
            if (key != null) {
                out.put(key, entry.getValue());
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }
}
