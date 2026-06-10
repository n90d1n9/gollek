package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.isSha256Hex;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.pathValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.stringValue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Field-level requirements for evidence manifest validation.
 */
final class TrainingReportPromotionGateEvidenceManifestRequirements {
    private TrainingReportPromotionGateEvidenceManifestRequirements() {
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

    static void requireDirectoryPath(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<Path> path = pathValue(map, key);
        if (path.isEmpty()) {
            failures.add(owner + " is missing directory path field " + key);
            return;
        }
        if (!Files.isDirectory(path.orElseThrow())) {
            failures.add(owner + " directory path field " + key + " is not a directory: "
                    + path.orElseThrow());
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

    static Map<String, Object> requireFileReference(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Map<String, Object> value = requireObject(map, key, owner, failures);
        if (value != null) {
            requireFileReferenceFields(value, owner + "." + key, failures);
        }
        return value;
    }

    static void requireFileReferenceFields(
            Map<String, ?> reference,
            String owner,
            List<String> failures) {
        requireString(reference, "file", owner, failures);
        requireSha256(reference, "sha256", owner, failures);
    }

    static void requireNonNegativeNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<Long> value = longValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing numeric field " + key);
            return;
        }
        if (value.orElseThrow().longValue() < 0L) {
            failures.add(owner + " has negative numeric field " + key);
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

    static void requireOptionalString(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (map.containsKey(key) && stringValue(map, key).isEmpty()) {
            failures.add(owner + " has invalid string field " + key);
        }
    }

    static void requireOptionalNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (map.containsKey(key) && longValue(map, key).isEmpty()) {
            failures.add(owner + " has invalid numeric field " + key);
        }
    }

    static void requireOptionalSha256(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (!map.containsKey(key)) {
            return;
        }
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty() || !isSha256Hex(value.orElseThrow())) {
            failures.add(owner + " has invalid SHA-256 field " + key);
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
