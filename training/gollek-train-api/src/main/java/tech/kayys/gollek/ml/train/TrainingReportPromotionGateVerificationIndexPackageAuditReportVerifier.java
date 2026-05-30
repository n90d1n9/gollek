package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Reader/verifier for persisted verification-index package-audit reports.
 */
final class TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier {
    private TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection read(
            Path reportFile) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReportFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate verification index package audit report JSON at "
                    + resolvedReportFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate verification index package audit report at "
                    + resolvedReportFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection(
                resolvedReportFile,
                TrainerCheckpointIO.sha256Hex(resolvedReportFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification verify(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection =
                read(reportFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReportSha256 == null || expectedReportSha256.isBlank()
                ? null
                : expectedReportSha256.trim();
        boolean reportSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.reportSha256());
        if (!reportSha256Matches) {
            failures.add("Verification index package audit report checksum mismatch for "
                    + inspection.reportFile() + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.reportSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit audit = null;
        if (schemaValid && inspection.indexFile() != null) {
            try {
                audit = TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(
                        inspection.indexFile(),
                        inspection.indexSha256());
                failures.addAll(audit.failures());
                verifyMatchesRevalidation(inspection, audit, failures);
            } catch (IOException error) {
                failures.add("Verification index package audit report could not revalidate index "
                        + inspection.indexFile() + ": " + error.getMessage());
            }
        }
        boolean auditRevalidated = audit != null && audit.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification(
                inspection,
                normalizedExpectedSha256,
                reportSha256Matches,
                schemaValid,
                auditRevalidated,
                audit,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> report = inspection.report();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification index package audit report format mismatch for " + inspection.reportFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT
                    + " but found " + inspection.format());
        }
        requireString(report, "indexFile", "verification index package audit report", failures);
        requireString(report, "indexSha256", "verification index package audit report", failures);
        requireBoolean(report, "passed", "verification index package audit report", failures);
        requireBoolean(report, "indexPassed", "verification index package audit report", failures);
        requireBoolean(report, "packagePassed", "verification index package audit report", failures);
        requireIterable(report, "failures", "verification index package audit report", failures);
        requireObject(report, "audit", "verification index package audit report", failures);
        return failures.size() == before;
    }

    private static void verifyMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification index package audit report pass status is stale for "
                    + inspection.reportFile() + ": report says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReportBoolean(inspection, "indexPassed", revalidated.indexVerification().passed(), failures);
        boolean packagePassed = revalidated.packageVerification() != null
                && revalidated.packageVerification().passed();
        verifyReportBoolean(inspection, "packagePassed", packagePassed, failures);
    }

    private static void verifyReportBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            String key,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(inspection.report(), key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification index package audit report " + key + " is stale for "
                    + inspection.reportFile() + ": report says " + recorded.get()
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
