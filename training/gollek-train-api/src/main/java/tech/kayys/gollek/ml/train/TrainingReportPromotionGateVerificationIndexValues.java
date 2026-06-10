package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed accessors for raw verification-index artifact maps.
 */
final class TrainingReportPromotionGateVerificationIndexValues {
    private TrainingReportPromotionGateVerificationIndexValues() {
    }

    static Optional<String> stringValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.stringValue(map, key);
    }

    static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.objectValue(map, key);
    }

    static Optional<List<?>> iterableValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.iterableValue(map, key);
    }

    static Optional<Long> longValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.longValue(map, key);
    }

    static Optional<Boolean> booleanValue(Map<String, ?> map, String key) {
        return TrainingReportMapValues.booleanValue(map, key);
    }

    static Map<String, Object> immutableMap(Map<?, ?> map) {
        return TrainingReportMapValues.immutableMap(map);
    }
}
