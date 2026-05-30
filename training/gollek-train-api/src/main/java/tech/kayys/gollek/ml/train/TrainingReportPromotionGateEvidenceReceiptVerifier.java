package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Reader/verifier for the terminal promotion evidence receipt.
 */
final class TrainingReportPromotionGateEvidenceReceiptVerifier {
    private TrainingReportPromotionGateEvidenceReceiptVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection read(Path receiptFile)
            throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification evidence receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification evidence receipt at "
                    + resolvedReceiptFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection(
                resolvedReceiptFile,
                TrainerCheckpointIO.sha256Hex(resolvedReceiptFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection =
                read(receiptFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                ? null
                : expectedReceiptSha256.trim();
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Verification evidence receipt checksum mismatch for " + inspection.receiptFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification evidenceVerification = null;
        if (schemaValid && inspection.evidenceFile() != null) {
            try {
                evidenceVerification = TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifest(
                        inspection.evidenceFile(),
                        inspection.evidenceSha256());
                failures.addAll(evidenceVerification.failures());
                verifyMatchesRevalidation(inspection, evidenceVerification, failures);
            } catch (IOException error) {
                failures.add("Verification evidence receipt could not revalidate evidence manifest "
                        + inspection.evidenceFile() + ": " + error.getMessage());
            }
        }
        boolean evidenceRevalidated = evidenceVerification != null && evidenceVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                evidenceRevalidated,
                evidenceVerification,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification evidence receipt format mismatch for " + inspection.receiptFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT
                    + " but found " + inspection.format());
        }
        requireInstant(receipt, "generatedAt", "verification evidence receipt", failures);
        requireString(receipt, "evidenceFile", "verification evidence receipt", failures);
        requireSha256(receipt, "evidenceSha256", "verification evidence receipt", failures);
        requireBoolean(receipt, "passed", "verification evidence receipt", failures);
        requireBoolean(receipt, "evidenceSha256Matches", "verification evidence receipt", failures);
        requireBoolean(receipt, "schemaValid", "verification evidence receipt", failures);
        requireBoolean(receipt, "evidenceFilesSha256Match", "verification evidence receipt", failures);
        requireBoolean(receipt, "packageArtifactsSha256Match", "verification evidence receipt", failures);
        requireStringList(receipt, "failures", "verification evidence receipt", failures);
        Map<String, Object> verification = requireObject(receipt, "verification", "verification evidence receipt", failures);
        if (verification != null) {
            validateEmbeddedVerificationSchema(verification, failures);
        }
        return failures.size() == before;
    }

    private static void verifyMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification evidence receipt pass status is stale for " + inspection.receiptFile()
                    + ": receipt says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReceiptString(
                inspection,
                "evidenceSha256",
                revalidated.inspection().evidenceSha256(),
                failures);
        verifyReceiptBoolean(inspection, "evidenceSha256Matches", revalidated.evidenceSha256Matches(), failures);
        verifyReceiptBoolean(inspection, "schemaValid", revalidated.schemaValid(), failures);
        verifyReceiptBoolean(inspection, "evidenceFilesSha256Match", revalidated.evidenceFilesSha256Match(), failures);
        verifyReceiptBoolean(
                inspection,
                "packageArtifactsSha256Match",
                revalidated.packageArtifactsSha256Match(),
                failures);
        verifyReceiptFailures(
                inspection,
                inspection.receipt(),
                "Verification evidence receipt",
                revalidated.failures(),
                failures);
        objectValue(inspection.receipt(), "verification").ifPresent(verification ->
                verifyEmbeddedVerificationMatchesRevalidation(
                        inspection,
                        verification,
                        revalidated,
                        failures));
    }

    private static void validateEmbeddedVerificationSchema(
            Map<String, ?> verification,
            List<String> failures) {
        String owner = "verification evidence receipt verification";
        requireBoolean(verification, "passed", owner, failures);
        requireBoolean(verification, "evidenceSha256Matches", owner, failures);
        requireBoolean(verification, "schemaValid", owner, failures);
        requireBoolean(verification, "evidenceFilesSha256Match", owner, failures);
        requireBoolean(verification, "packageArtifactsSha256Match", owner, failures);
        requireSha256(verification, "actualEvidenceSha256", owner, failures);
        requireStringList(verification, "failures", owner, failures);
        requireObject(verification, "inspection", owner, failures);
    }

    private static void verifyEmbeddedVerificationMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            Map<String, ?> verification,
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification revalidated,
            List<String> failures) {
        String owner = "Verification evidence receipt embedded verification";
        verifyEmbeddedBoolean(inspection, verification, "passed", revalidated.passed(), owner, failures);
        verifyEmbeddedString(
                inspection,
                verification,
                "actualEvidenceSha256",
                revalidated.inspection().evidenceSha256(),
                owner,
                failures);
        verifyEmbeddedBoolean(
                inspection,
                verification,
                "evidenceSha256Matches",
                revalidated.evidenceSha256Matches(),
                owner,
                failures);
        verifyEmbeddedBoolean(inspection, verification, "schemaValid", revalidated.schemaValid(), owner, failures);
        verifyEmbeddedBoolean(
                inspection,
                verification,
                "evidenceFilesSha256Match",
                revalidated.evidenceFilesSha256Match(),
                owner,
                failures);
        verifyEmbeddedBoolean(
                inspection,
                verification,
                "packageArtifactsSha256Match",
                revalidated.packageArtifactsSha256Match(),
                owner,
                failures);
        verifyReceiptFailures(inspection, verification, owner, revalidated.failures(), failures);
    }

    private static void verifyReceiptBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            String key,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(inspection.receipt(), key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification evidence receipt " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    private static void verifyReceiptString(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            String key,
            String actual,
            List<String> failures) {
        Optional<String> recorded = stringValue(inspection.receipt(), key);
        if (recorded.isPresent() && !recorded.get().equalsIgnoreCase(actual)) {
            failures.add("Verification evidence receipt " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    private static void verifyEmbeddedBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            Map<String, ?> map,
            String key,
            boolean actual,
            String owner,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(map, key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add(owner + " " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    private static void verifyEmbeddedString(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            Map<String, ?> map,
            String key,
            String actual,
            String owner,
            List<String> failures) {
        Optional<String> recorded = stringValue(map, key);
        if (recorded.isPresent() && !recorded.get().equalsIgnoreCase(actual)) {
            failures.add(owner + " " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    private static void verifyReceiptFailures(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            Map<String, ?> map,
            String owner,
            List<String> actual,
            List<String> failures) {
        Optional<List<String>> recorded = stringListValue(map, "failures", owner, failures);
        if (recorded.isPresent() && !recorded.orElseThrow().equals(actual)) {
            failures.add(owner + " failures are stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.orElseThrow()
                    + " but revalidation says " + actual);
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

    private static void requireBoolean(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (booleanValue(map, key).isEmpty()) {
            failures.add(owner + " is missing boolean field " + key);
        }
    }

    private static void requireSha256(
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

    private static void requireInstant(
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

    private static void requireIterable(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (iterableValue(map, key).isEmpty()) {
            failures.add(owner + " is missing array field " + key);
        }
    }

    private static void requireStringList(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (stringListValue(map, key, owner, failures).isEmpty()) {
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

    private static Optional<List<String>> stringListValue(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<List<?>> values = iterableValue(map, key);
        if (values.isEmpty()) {
            return Optional.empty();
        }
        List<String> strings = new ArrayList<>();
        int index = 0;
        for (Object value : values.orElseThrow()) {
            if (value instanceof String text) {
                strings.add(text);
            } else {
                failures.add(owner + "." + key + "[" + index + "] must be a string");
            }
            index++;
        }
        return Optional.of(List.copyOf(strings));
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

    private static Optional<Boolean> booleanValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Boolean flag) {
            return Optional.of(flag);
        }
        if (value instanceof String text && !text.isBlank()) {
            String normalized = text.trim().toLowerCase();
            if ("true".equals(normalized)) {
                return Optional.of(Boolean.TRUE);
            }
            if ("false".equals(normalized)) {
                return Optional.of(Boolean.FALSE);
            }
        }
        return Optional.empty();
    }

    private static boolean isSha256Hex(String value) {
        if (value.length() != 64) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            boolean hex = (ch >= '0' && ch <= '9')
                    || (ch >= 'a' && ch <= 'f')
                    || (ch >= 'A' && ch <= 'F');
            if (!hex) {
                return false;
            }
        }
        return true;
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
