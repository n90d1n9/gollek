package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.stringValue;

import java.util.List;
import java.util.Map;

/**
 * Field-level requirements for verification-index artifact schema validation.
 */
final class TrainingReportPromotionGateVerificationIndexRequirements {
    private TrainingReportPromotionGateVerificationIndexRequirements() {
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

    static void requireNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (longValue(map, key).isEmpty()) {
            failures.add(owner + " is missing numeric field " + key);
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
