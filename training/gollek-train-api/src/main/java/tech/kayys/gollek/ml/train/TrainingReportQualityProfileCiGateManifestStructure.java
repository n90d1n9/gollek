package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class TrainingReportQualityProfileCiGateManifestStructure {
    private static final List<String> REQUIRED_FIELDS = List.of(
            "format",
            "profileId",
            "profile",
            "passed",
            "validationPassed",
            "promotionPassed",
            "artifactsVerified",
            "message",
            "validationCount",
            "artifactCount",
            "artifacts");

    private TrainingReportQualityProfileCiGateManifestStructure() {
    }

    static List<String> verify(TrainingReportQualityProfileCiGateManifest.ManifestInspection inspection) {
        List<String> failures = new ArrayList<>();
        Map<String, Object> manifest = inspection.manifest();
        List<Map<String, Object>> artifacts = inspection.artifactMaps();
        verifyRequiredFields(manifest, failures);
        verifyDeclaredCounts(manifest, artifacts, failures);
        verifyArtifactIdentity(artifacts, failures);
        return List.copyOf(failures);
    }

    static boolean isFailure(String failure) {
        if (failure == null || failure.isBlank()) {
            return false;
        }
        return failure.startsWith("Manifest is missing required field")
                || failure.startsWith("Manifest artifact count mismatch")
                || failure.startsWith("Manifest validation count mismatch")
                || failure.startsWith("Manifest artifact has blank name")
                || failure.startsWith("Manifest artifact `")
                || failure.startsWith("Manifest artifact name is duplicated")
                || failure.startsWith("Manifest artifact file is duplicated");
    }

    private static void verifyRequiredFields(Map<String, Object> manifest, List<String> failures) {
        for (String field : REQUIRED_FIELDS) {
            if (!manifest.containsKey(field)) {
                failures.add("Manifest is missing required field `" + field + "`.");
            }
        }
    }

    private static void verifyDeclaredCounts(
            Map<String, Object> manifest,
            List<Map<String, Object>> artifacts,
            List<String> failures) {
        long declaredArtifactCount = longValue(manifest.get("artifactCount"), -1L);
        if (declaredArtifactCount != artifacts.size()) {
            failures.add("Manifest artifact count mismatch: declared "
                    + declaredArtifactCount + ", found " + artifacts.size() + ".");
        }

        long declaredValidationCount = longValue(manifest.get("validationCount"), -1L);
        int reportCount = reportNames(artifacts).size();
        if (declaredValidationCount != reportCount) {
            failures.add("Manifest validation count mismatch: declared "
                    + declaredValidationCount + ", found " + reportCount + ".");
        }
    }

    private static void verifyArtifactIdentity(List<Map<String, Object>> artifacts, List<String> failures) {
        Set<String> names = new LinkedHashSet<>();
        Set<String> paths = new LinkedHashSet<>();
        for (Map<String, Object> artifact : artifacts) {
            String name = stringValue(artifact.get("name"), "");
            String kind = stringValue(artifact.get("kind"), "");
            String file = stringValue(artifact.get("file"), "");
            if (name.isBlank()) {
                failures.add("Manifest artifact has blank name.");
            } else if (!names.add(name)) {
                failures.add("Manifest artifact name is duplicated: `" + name + "`.");
            }
            if (kind.isBlank()) {
                failures.add("Manifest artifact `" + fallbackName(name) + "` has blank kind.");
            }
            if (file.isBlank()) {
                failures.add("Manifest artifact `" + fallbackName(name) + "` has blank file.");
            } else if (!paths.add(file)) {
                failures.add("Manifest artifact file is duplicated: `" + file + "`.");
            }
            if (longValue(artifact.get("bytes"), -1L) < 0L) {
                failures.add("Manifest artifact `" + fallbackName(name) + "` has invalid byte size.");
            }
            if (stringValue(artifact.get("sha256"), "").isBlank()) {
                failures.add("Manifest artifact `" + fallbackName(name) + "` has blank SHA-256.");
            }
        }
    }

    private static Set<String> reportNames(List<Map<String, Object>> artifacts) {
        Set<String> reports = new LinkedHashSet<>();
        for (Map<String, Object> artifact : artifacts) {
            String reportName = stringValue(artifact.get("reportName"), "");
            if (!reportName.isBlank()) {
                reports.add(reportName);
            }
        }
        return reports;
    }

    private static String fallbackName(String name) {
        return name == null || name.isBlank() ? "unknown" : name;
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
}
