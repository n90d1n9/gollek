package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.stringValue;

import java.util.List;
import java.util.Map;

/**
 * Field-level requirements for promotion-gate JSON-style map validation.
 */
final class TrainingReportPromotionGateMapRequirements {
    private TrainingReportPromotionGateMapRequirements() {
    }

    static void requireString(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (stringValue(map, key).isEmpty()) {
            failures.add(owner + " is missing string field " + key);
        }
    }

    static void requireBoolean(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (booleanValue(map, key).isEmpty()) {
            failures.add(owner + " is missing boolean field " + key);
        }
    }

    static Map<String, Object> requireObject(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Map<String, Object> value = objectValue(map, key).orElse(null);
        if (value == null) {
            failures.add(owner + " is missing object field " + key);
        }
        return value;
    }

    static void requireIterable(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (iterableValue(map, key).isEmpty()) {
            failures.add(owner + " is missing array field " + key);
        }
    }
}
