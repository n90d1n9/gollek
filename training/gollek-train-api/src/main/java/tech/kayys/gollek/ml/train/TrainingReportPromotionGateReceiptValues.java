package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed accessors shared by promotion-gate receipt verifiers.
 */
final class TrainingReportPromotionGateReceiptValues {
    private TrainingReportPromotionGateReceiptValues() {
    }

    static Optional<String> stringValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.stringValue(map, key);
    }

    static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.objectValue(map, key);
    }

    static Optional<Path> pathValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.pathValue(map, key);
    }

    static Optional<List<String>> stringListValue(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        return TrainingReportMapValues.stringListValue(map, key, owner, failures);
    }

    static Optional<List<?>> iterableValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.iterableValue(map, key);
    }

    static Optional<Boolean> booleanValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.booleanValue(map, key);
    }

    static Optional<Long> longValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.longValue(map, key);
    }

    static boolean isSha256Hex(String value) {
        return TrainingReportMapValues.isSha256Hex(value);
    }

    static Map<String, Object> immutableMap(Map<?, ?> map) {
        return TrainingReportMapValues.immutableMap(map);
    }
}
