package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Reader/verifier for the promotion-gate verification index.
 */
final class TrainingReportPromotionGateVerificationIndexVerifier {
    private TrainingReportPromotionGateVerificationIndexVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection read(Path indexFile)
            throws IOException {
        Path resolvedIndexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedIndexFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification index JSON at "
                    + resolvedIndexFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification index at "
                    + resolvedIndexFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection(
                resolvedIndexFile,
                TrainerCheckpointIO.sha256Hex(resolvedIndexFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verify(
            Path indexFile,
            String expectedIndexSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection = read(indexFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedIndexSha256 == null || expectedIndexSha256.isBlank()
                ? null
                : expectedIndexSha256.trim();
        boolean indexSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.indexSha256());
        if (!indexSha256Matches) {
            failures.add("Verification index checksum mismatch for " + inspection.indexFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.indexSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        boolean referencesValid = verifyReferences(inspection, failures);
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification(
                inspection,
                normalizedExpectedSha256,
                indexSha256Matches,
                schemaValid,
                schemaValid && referencesValid,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> index = inspection.index();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT.equals(inspection.format())) {
            failures.add("Verification index format mismatch for " + inspection.indexFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT
                    + " but found " + inspection.format());
        }
        requireString(index, "packageDirectory", "verification index", failures);
        requireString(index, "reportDirectory", "verification index", failures);
        requireString(index, "decisionStatus", "verification index", failures);
        requireObject(index, "manifest", "verification index", failures);
        Map<String, Object> reports = requireObject(index, "reports", "verification index", failures);
        if (reports != null) {
            requireObject(reports, "json", "verification index reports", failures);
            requireObject(reports, "markdown", "verification index reports", failures);
            requireObject(reports, "junitXml", "verification index reports", failures);
            objectValue(reports, "receipt").ifPresent(receipt ->
                    verifyFileReferenceShape("receipt", receipt, "verification index reports", failures));
        }
        Map<String, Object> sourceSnapshots =
                requireObject(index, "sourceReportSnapshots", "verification index", failures);
        if (sourceSnapshots != null) {
            requireNumber(sourceSnapshots, "expected", "verification index sourceReportSnapshots", failures);
            requireNumber(sourceSnapshots, "present", "verification index sourceReportSnapshots", failures);
            requireIterable(sourceSnapshots, "snapshots", "verification index sourceReportSnapshots", failures);
        }
        return failures.size() == before;
    }

    private static boolean verifyReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> index = inspection.index();
        Map<String, Object> manifest = objectValue(index, "manifest").orElse(null);
        if (manifest != null) {
            verifyFileReference("manifest", manifest, null, failures);
        }

        Map<String, Object> reports = objectValue(index, "reports").orElse(null);
        if (reports != null) {
            for (String reportName : List.of("json", "markdown", "junitXml", "receipt")) {
                Map<String, Object> report = objectValue(reports, reportName).orElse(null);
                if (report != null) {
                    verifyFileReference("reports." + reportName, report, null, failures);
                }
            }
        }

        Map<String, Object> sourceSnapshots = objectValue(index, "sourceReportSnapshots").orElse(null);
        if (sourceSnapshots != null) {
            List<?> snapshots = iterableValue(sourceSnapshots, "snapshots").orElse(List.of());
            int snapshotIndex = 0;
            for (Object item : snapshots) {
                if (!(item instanceof Map<?, ?> snapshotMap)) {
                    failures.add("Verification index sourceReportSnapshots.snapshots["
                            + snapshotIndex + "] must be an object");
                    snapshotIndex++;
                    continue;
                }
                Map<String, Object> snapshot = immutableMap(snapshotMap);
                String artifact = stringValue(snapshot, "artifact").orElse(Integer.toString(snapshotIndex));
                verifyFileReference(
                        "sourceReportSnapshots." + artifact,
                        snapshot,
                        longValue(snapshot, "bytes").orElse(null),
                        failures);
                snapshotIndex++;
            }
        }
        return failures.size() == before;
    }

    private static boolean verifyFileReference(
            String label,
            Map<String, ?> reference,
            Long expectedBytes,
            List<String> failures) throws IOException {
        int before = failures.size();
        String file = stringValue(reference, "file").orElse(null);
        String expectedSha256 = stringValue(reference, "sha256").orElse(null);
        if (file == null || file.isBlank()) {
            failures.add("Verification index " + label + " is missing file");
            return false;
        }
        if (expectedSha256 == null || expectedSha256.isBlank()) {
            failures.add("Verification index " + label + " is missing sha256");
            return false;
        }
        Path path;
        try {
            path = Path.of(file).toAbsolutePath().normalize();
        } catch (InvalidPathException error) {
            failures.add("Verification index " + label + " has invalid file path: " + file);
            return false;
        }
        if (!Files.isRegularFile(path)) {
            failures.add("Verification index " + label + " file is missing: " + path);
            return false;
        }
        if (expectedBytes != null) {
            long actualBytes = Files.size(path);
            if (actualBytes != expectedBytes.longValue()) {
                failures.add("Verification index " + label + " byte count mismatch for " + path
                        + ": expected " + expectedBytes + " but found " + actualBytes);
            }
        }
        String actualSha256 = TrainerCheckpointIO.sha256Hex(path);
        if (!expectedSha256.equalsIgnoreCase(actualSha256)) {
            failures.add("Verification index " + label + " checksum mismatch for " + path
                    + ": expected " + expectedSha256 + " but found " + actualSha256);
        }
        return failures.size() == before;
    }

    private static void verifyFileReferenceShape(
            String label,
            Map<String, ?> reference,
            String owner,
            List<String> failures) {
        if (stringValue(reference, "file").isEmpty()) {
            failures.add(owner + " " + label + " is missing string field file");
        }
        if (stringValue(reference, "sha256").isEmpty()) {
            failures.add(owner + " " + label + " is missing string field sha256");
        }
    }

    private static void requireString(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (stringValue(map, key).isEmpty()) {
            failures.add(owner + " is missing string field " + key);
        }
    }

    private static Map<String, Object> requireObject(
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

    private static void requireNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (longValue(map, key).isEmpty()) {
            failures.add(owner + " is missing numeric field " + key);
        }
    }

    private static void requireIterable(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (iterableValue(map, key).isEmpty()) {
            failures.add(owner + " is missing array field " + key);
        }
    }

    private static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    private static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Map<?, ?> object) {
            return Optional.of(immutableMap(object));
        }
        return Optional.empty();
    }

    private static Optional<List<?>> iterableValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof List<?> list) {
            return Optional.of(List.copyOf(list));
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> values = new ArrayList<>();
            for (Object item : iterable) {
                values.add(item);
            }
            return Optional.of(List.copyOf(values));
        }
        return Optional.empty();
    }

    private static Optional<Long> longValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Optional.of(Long.parseLong(text));
            } catch (NumberFormatException ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
