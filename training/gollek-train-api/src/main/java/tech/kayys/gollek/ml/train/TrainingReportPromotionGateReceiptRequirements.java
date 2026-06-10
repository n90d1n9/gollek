package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.isSha256Hex;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.stringListValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.stringValue;

import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Field-level requirements shared by promotion-gate receipt schema checks.
 */
final class TrainingReportPromotionGateReceiptRequirements {
    private TrainingReportPromotionGateReceiptRequirements() {
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

    static void requireSha256(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing SHA-256 field " + key);
            return;
        }
        if (!isSha256Hex(value.orElseThrow())) {
            failures.add(owner + " has invalid SHA-256 field " + key);
        }
    }

    static void requireInstant(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing instant field " + key);
            return;
        }
        try {
            Instant.parse(value.orElseThrow());
        } catch (DateTimeParseException error) {
            failures.add(owner + " has invalid instant field " + key);
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

    static void requireStringList(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (stringListValue(map, key, owner, failures).isEmpty()) {
            failures.add(owner + " is missing array field " + key);
        }
    }
}
