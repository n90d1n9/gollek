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
 * Reader/verifier for persisted complete package verification JSON reports.
 */
final class TrainingReportPromotionGateVerificationReportVerifier {
    private TrainingReportPromotionGateVerificationReportVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportInspection read(Path reportFile)
            throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReportFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification report JSON at "
                    + resolvedReportFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification report at "
                    + resolvedReportFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportInspection(
                resolvedReportFile,
                TrainerCheckpointIO.sha256Hex(resolvedReportFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportVerification verify(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection = read(reportFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReportSha256 == null || expectedReportSha256.isBlank()
                ? null
                : expectedReportSha256.trim();
        boolean reportSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.reportSha256());
        if (!reportSha256Matches) {
            failures.add("Verification report checksum mismatch for " + inspection.reportFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.reportSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.PackageVerification packageVerification = null;
        if (schemaValid && inspection.packageDirectory() != null) {
            try {
                packageVerification = TrainingReportPromotionGateArtifactPackage.verifyComplete(
                        inspection.packageDirectory(),
                        inspection.manifestSha256(),
                        optionsFromInspection(inspection));
                failures.addAll(packageVerification.failures());
                verifyMatchesRevalidation(inspection, packageVerification, failures);
            } catch (IOException error) {
                failures.add("Verification report could not revalidate package "
                        + inspection.packageDirectory() + ": " + error.getMessage());
            }
        }
        boolean packageRevalidated = packageVerification != null && packageVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportVerification(
                inspection,
                normalizedExpectedSha256,
                reportSha256Matches,
                schemaValid,
                packageRevalidated,
                packageVerification,
                failures);
    }

    private static TrainingReportPromotionGateArtifactPackage.Options optionsFromInspection(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection) {
        Path manifestFile = inspection.manifestFile();
        if (manifestFile == null || manifestFile.getFileName() == null) {
            return TrainingReportPromotionGateArtifactPackage.Options.defaults();
        }
        return new TrainingReportPromotionGateArtifactPackage.Options(
                TrainingReportPromotionArtifacts.Options.defaults(),
                TrainingReportPromotionGateArtifacts.Options.defaults(),
                new TrainingReportPromotionGateArtifactManifest.Options(manifestFile.getFileName().toString()));
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> report = inspection.report();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT.equals(inspection.format())) {
            failures.add("Verification report format mismatch for " + inspection.reportFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT
                    + " but found " + inspection.format());
        }
        requireBoolean(report, "passed", "verification report", failures);
        requireIterable(report, "failures", "verification report", failures);
        Map<String, Object> manifestVerification =
                requireObject(report, "manifestVerification", "verification report", failures);
        if (manifestVerification != null) {
            requireString(manifestVerification, "actualManifestSha256", "verification report manifestVerification",
                    failures);
            requireBoolean(manifestVerification, "manifestSha256Matches", "verification report manifestVerification",
                    failures);
            requireBoolean(manifestVerification, "artifactBytesMatch", "verification report manifestVerification",
                    failures);
            requireBoolean(manifestVerification, "artifactSha256Match", "verification report manifestVerification",
                    failures);
            Map<String, Object> manifestInspection = requireObject(
                    manifestVerification,
                    "inspection",
                    "verification report manifestVerification",
                    failures);
            if (manifestInspection != null) {
                requireString(manifestInspection, "directory", "verification report manifest inspection", failures);
                requireString(manifestInspection, "manifestFile", "verification report manifest inspection", failures);
            }
        }
        Map<String, Object> sourceSnapshotVerification =
                requireObject(report, "sourceSnapshotVerification", "verification report", failures);
        if (sourceSnapshotVerification != null) {
            requireBoolean(sourceSnapshotVerification, "passed", "verification report sourceSnapshotVerification",
                    failures);
            requireIterable(sourceSnapshotVerification, "snapshots", "verification report sourceSnapshotVerification",
                    failures);
        }
        Map<String, Object> packageInspection = requireObject(report, "inspection", "verification report", failures);
        if (packageInspection != null) {
            requireString(packageInspection, "directory", "verification report inspection", failures);
        }
        return failures.size() == before;
    }

    private static void verifyMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            TrainingReportPromotionGateArtifactPackage.PackageVerification revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification report pass status is stale for " + inspection.reportFile()
                    + ": report says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "manifestSha256Matches",
                revalidated.manifestVerification().manifestSha256Matches(),
                failures);
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "artifactBytesMatch",
                revalidated.manifestVerification().artifactBytesMatch(),
                failures);
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "artifactSha256Match",
                revalidated.manifestVerification().artifactSha256Match(),
                failures);
        verifyReportBoolean(
                inspection,
                "sourceSnapshotVerification",
                "passed",
                revalidated.sourceSnapshotVerification().passed(),
                failures);
    }

    private static void verifyReportBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            String objectKey,
            String booleanKey,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = objectValue(inspection.report(), objectKey)
                .flatMap(object -> booleanValue(object, booleanKey));
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification report " + objectKey + "." + booleanKey + " is stale for "
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
