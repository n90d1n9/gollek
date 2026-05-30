package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Reader/verifier for the terminal promotion verification-report bundle receipt.
 */
final class TrainingReportPromotionGateVerificationReportBundleReceiptVerifier {
    private TrainingReportPromotionGateVerificationReportBundleReceiptVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection read(
            Path receiptFile) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification report bundle receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification report bundle receipt at "
                    + resolvedReceiptFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection(
                resolvedReceiptFile,
                TrainerCheckpointIO.sha256Hex(resolvedReceiptFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection =
                read(receiptFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                ? null
                : expectedReceiptSha256.trim();
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Verification report bundle receipt checksum mismatch for " + inspection.receiptFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification bundleVerification = null;
        if (schemaValid && inspection.reportDirectory() != null) {
            try {
                bundleVerification = TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundle(
                        inspection.reportDirectory(),
                        inspection.jsonReportSha256());
                failures.addAll(bundleVerification.failures());
                verifyMatchesRevalidation(inspection, bundleVerification, failures);
            } catch (IOException error) {
                failures.add("Verification report bundle receipt could not revalidate report bundle "
                        + inspection.reportDirectory() + ": " + error.getMessage());
            }
        }
        boolean reportBundleRevalidated = bundleVerification != null && bundleVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                reportBundleRevalidated,
                bundleVerification,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification report bundle receipt format mismatch for " + inspection.receiptFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT
                    + " but found " + inspection.format());
        }
        requireString(receipt, "reportDirectory", "verification report bundle receipt", failures);
        requireString(receipt, "jsonReportFile", "verification report bundle receipt", failures);
        requireString(receipt, "jsonReportSha256", "verification report bundle receipt", failures);
        requireString(receipt, "markdownFile", "verification report bundle receipt", failures);
        requireString(receipt, "markdownSha256", "verification report bundle receipt", failures);
        requireString(receipt, "junitXmlFile", "verification report bundle receipt", failures);
        requireString(receipt, "junitXmlSha256", "verification report bundle receipt", failures);
        requireBoolean(receipt, "passed", "verification report bundle receipt", failures);
        requireBoolean(receipt, "jsonReportVerified", "verification report bundle receipt", failures);
        requireBoolean(receipt, "markdownMatchesRendered", "verification report bundle receipt", failures);
        requireBoolean(receipt, "junitXmlMatchesRendered", "verification report bundle receipt", failures);
        requireIterable(receipt, "failures", "verification report bundle receipt", failures);
        requireObject(receipt, "verification", "verification report bundle receipt", failures);
        return failures.size() == before;
    }

    private static void verifyMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification report bundle receipt pass status is stale for "
                    + inspection.receiptFile() + ": receipt says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReceiptPath(inspection, "jsonReportFile", revalidated.inspection().json().reportFile(), failures);
        verifyReceiptString(inspection, "jsonReportSha256", revalidated.inspection().json().reportSha256(), failures);
        verifyReceiptPath(inspection, "markdownFile", revalidated.inspection().markdownFile(), failures);
        verifyReceiptString(inspection, "markdownSha256", revalidated.inspection().markdownSha256(), failures);
        verifyReceiptPath(inspection, "junitXmlFile", revalidated.inspection().junitXmlFile(), failures);
        verifyReceiptString(inspection, "junitXmlSha256", revalidated.inspection().junitXmlSha256(), failures);
        verifyReceiptBoolean(inspection, "jsonReportVerified", revalidated.jsonReportVerified(), failures);
        verifyReceiptBoolean(inspection, "markdownMatchesRendered", revalidated.markdownMatchesRendered(), failures);
        verifyReceiptBoolean(inspection, "junitXmlMatchesRendered", revalidated.junitXmlMatchesRendered(), failures);
    }

    private static void verifyReceiptBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            String key,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(inspection.receipt(), key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification report bundle receipt " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    private static void verifyReceiptPath(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            String key,
            Path actual,
            List<String> failures) {
        Optional<Path> recorded = pathValue(inspection.receipt(), key);
        Path normalizedActual = actual == null ? null : actual.toAbsolutePath().normalize();
        if (recorded.isPresent() && !recorded.get().equals(normalizedActual)) {
            failures.add("Verification report bundle receipt " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
                    + " but revalidation says " + normalizedActual);
        }
    }

    private static void verifyReceiptString(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            String key,
            String actual,
            List<String> failures) {
        Optional<String> recorded = stringValue(inspection.receipt(), key);
        if (recorded.isPresent() && !recorded.get().equalsIgnoreCase(actual)) {
            failures.add("Verification report bundle receipt " + key + " is stale for "
                    + inspection.receiptFile() + ": receipt says " + recorded.get()
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

    private static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    private static Optional<Path> pathValue(Map<String, ?> map, String key) {
        return stringValue(map, key).flatMap(value -> {
            try {
                return Optional.of(Path.of(value).toAbsolutePath().normalize());
            } catch (InvalidPathException ignored) {
                return Optional.empty();
            }
        });
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

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
