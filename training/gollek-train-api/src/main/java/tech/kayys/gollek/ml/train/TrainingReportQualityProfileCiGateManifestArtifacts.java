package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

final class TrainingReportQualityProfileCiGateManifestArtifacts {
    private TrainingReportQualityProfileCiGateManifestArtifacts() {
    }

    static List<TrainingReportQualityProfileCiGateManifest.ArtifactEntry> collect(
            Path directory,
            TrainingReportQualityProfileCiGate.Result result) throws IOException {
        List<TrainingReportQualityProfileCiGateManifest.ArtifactEntry> artifacts = new ArrayList<>();
        for (Map.Entry<String, TrainingReportQualityProfileValidationGate.Result> validation
                : result.validations().entrySet()) {
            addValidationArtifacts(artifacts, directory, validation, result);
        }
        addPromotionArtifacts(artifacts, directory, result);
        return List.copyOf(artifacts);
    }

    static List<String> verify(TrainingReportQualityProfileCiGateManifest.ManifestInspection inspection) {
        List<String> failures = new ArrayList<>();
        for (Map<String, Object> artifact : inspection.artifactMaps()) {
            verifyArtifact(inspection.directory(), artifact, failures);
        }
        return List.copyOf(failures);
    }

    static boolean isFailure(String failure) {
        if (failure == null || failure.isBlank()) {
            return false;
        }
        return failure.contains(" artifact has an unsafe or missing path")
                || failure.contains(" artifact is missing:")
                || failure.contains(" artifact byte size mismatch")
                || failure.contains(" artifact SHA-256 mismatch")
                || failure.contains(" artifact could not be verified:");
    }

    private static void addValidationArtifacts(
            List<TrainingReportQualityProfileCiGateManifest.ArtifactEntry> artifacts,
            Path directory,
            Map.Entry<String, TrainingReportQualityProfileValidationGate.Result> validation,
            TrainingReportQualityProfileCiGate.Result result) throws IOException {
        String reportName = validation.getKey();
        TrainingReportValidationArtifacts.ArtifactBundle raw = validation.getValue().artifacts();
        addArtifact(
                artifacts,
                directory,
                "validation." + reportName + ".json",
                "validation-json",
                reportName,
                raw.jsonFile());
        addArtifact(
                artifacts,
                directory,
                "validation." + reportName + ".markdown",
                "validation-markdown",
                reportName,
                raw.markdownFile());
        addArtifact(
                artifacts,
                directory,
                "validation." + reportName + ".junitXml",
                "validation-junit-xml",
                reportName,
                raw.junitXmlFile());

        TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle profile =
                result.validationArtifacts().get(reportName);
        if (profile != null) {
            addArtifact(
                    artifacts,
                    directory,
                    "profileValidation." + reportName + ".json",
                    "profile-validation-json",
                    reportName,
                    profile.jsonFile());
            addArtifact(
                    artifacts,
                    directory,
                    "profileValidation." + reportName + ".markdown",
                    "profile-validation-markdown",
                    reportName,
                    profile.markdownFile());
        }
    }

    private static void addPromotionArtifacts(
            List<TrainingReportQualityProfileCiGateManifest.ArtifactEntry> artifacts,
            Path directory,
            TrainingReportQualityProfileCiGate.Result result) throws IOException {
        TrainingReportPromotionArtifacts.ArtifactBundle promotion = result.promotion().artifacts();
        addArtifact(artifacts, directory, "promotion.json", "promotion-json", "", promotion.jsonFile());
        addArtifact(artifacts, directory, "promotion.markdown", "promotion-markdown", "", promotion.markdownFile());
        addArtifact(
                artifacts,
                directory,
                "profilePromotion.json",
                "profile-promotion-json",
                "",
                result.promotionArtifacts().jsonFile());
        addArtifact(
                artifacts,
                directory,
                "profilePromotion.markdown",
                "profile-promotion-markdown",
                "",
                result.promotionArtifacts().markdownFile());
    }

    private static void addArtifact(
            List<TrainingReportQualityProfileCiGateManifest.ArtifactEntry> artifacts,
            Path directory,
            String name,
            String kind,
            String reportName,
            Path file) throws IOException {
        Path resolvedFile = Objects.requireNonNull(file, "artifact file must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(resolvedFile);
        artifacts.add(new TrainingReportQualityProfileCiGateManifest.ArtifactEntry(
                name,
                kind,
                reportName,
                fingerprint.file(),
                manifestPath(directory, fingerprint.file()),
                fingerprint.bytes(),
                fingerprint.sha256()));
    }

    private static void verifyArtifact(Path directory, Map<String, Object> artifact, List<String> failures) {
        String name = stringValue(artifact.get("name"), "unknown");
        Path file = resolveManifestPath(directory, stringValue(artifact.get("file"), ""));
        long expectedBytes = longValue(artifact.get("bytes"), -1L);
        String expectedSha256 = normalizeChecksum(stringValue(artifact.get("sha256"), ""));
        if (file == null) {
            failures.add(name + " artifact has an unsafe or missing path.");
            return;
        }
        if (!Files.isRegularFile(file)) {
            failures.add(name + " artifact is missing: " + file);
            return;
        }
        try {
            verifyArtifactBytesAndChecksum(name, file, expectedBytes, expectedSha256, failures);
        } catch (IOException error) {
            failures.add(name + " artifact could not be verified: " + error.getMessage());
        }
    }

    private static void verifyArtifactBytesAndChecksum(
            String name,
            Path file,
            long expectedBytes,
            String expectedSha256,
            List<String> failures) throws IOException {
        TrainingReportArtifactFingerprint actual = TrainingReportArtifactFingerprint.of(file);
        if (expectedBytes != actual.bytes()) {
            failures.add(name + " artifact byte size mismatch for " + file
                    + ": expected " + expectedBytes + ", got " + actual.bytes());
        }
        if (expectedSha256 == null || !actual.sha256().equalsIgnoreCase(expectedSha256)) {
            failures.add(name + " artifact SHA-256 mismatch for " + file);
        }
    }

    private static Path resolveManifestPath(Path directory, String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return null;
        }
        try {
            Path path = Path.of(rawPath.trim());
            if (path.isAbsolute()) {
                return path.normalize();
            }
            Path normalized = directory.resolve(path).normalize();
            return normalized.startsWith(directory) ? normalized : null;
        } catch (RuntimeException error) {
            return null;
        }
    }

    private static String manifestPath(Path directory, Path file) {
        Path resolvedDirectory = directory.toAbsolutePath().normalize();
        Path resolvedFile = file.toAbsolutePath().normalize();
        if (resolvedFile.startsWith(resolvedDirectory)) {
            return resolvedDirectory.relativize(resolvedFile).toString();
        }
        return resolvedFile.toString();
    }

    private static long longValue(Object value, long fallback) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value instanceof String text) {
            try {
                return Long.parseLong(text.trim());
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }
}
