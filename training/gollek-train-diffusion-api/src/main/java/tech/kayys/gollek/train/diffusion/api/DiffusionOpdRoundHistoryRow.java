package tech.kayys.gollek.train.diffusion.api;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed row view of a normalized DiffusionOPD round-history entry.
 */
public record DiffusionOpdRoundHistoryRow(
        int round,
        String taskId,
        String teacherKey,
        String stageName,
        Double averageLoss,
        Map<String, Object> raw) {

    public static DiffusionOpdRoundHistoryRow fromMap(Map<String, Object> raw) {
        return new DiffusionOpdRoundHistoryRow(
                intValue(raw.get("round")),
                stringValue(raw.get("taskId")),
                stringValue(raw.get("teacherKey")),
                stringValue(raw.get("stageName")),
                doubleValue(raw.get("averageLoss")),
                Map.copyOf(new LinkedHashMap<>(raw)));
    }

    public Map<String, Object> asMap() {
        return raw;
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static int intValue(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value == null) {
            return 0;
        }
        return Integer.parseInt(String.valueOf(value));
    }

    private static Double doubleValue(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value == null) {
            return null;
        }
        return Double.parseDouble(String.valueOf(value));
    }
}
