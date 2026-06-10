package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.pathValue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Directory containment checks for evidence manifest file references.
 */
final class TrainingReportPromotionGateEvidenceManifestReferenceRoots {
    private TrainingReportPromotionGateEvidenceManifestReferenceRoots() {
    }

    static void validateEvidenceFiles(
            Map<String, ?> evidence,
            Map<String, ?> evidenceFiles,
            List<String> failures) {
        Optional<Path> packageDirectory = pathValue(evidence, "packageDirectory");
        Optional<Path> reportDirectory = pathValue(evidence, "reportDirectory");
        for (Map.Entry<String, ?> entry : evidenceFiles.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                continue;
            }
            String key = entry.getKey();
            boolean packageScoped = "manifest".equals(key);
            Optional<Path> root = packageScoped ? packageDirectory : reportDirectory;
            root.ifPresent(path -> validateReferencePathWithin(
                    path,
                    immutableMap(referenceMap),
                    "file",
                    "verification evidence evidenceFiles." + key,
                    packageScoped ? "packageDirectory" : "reportDirectory",
                    failures));
        }
    }

    static void validatePackageArtifacts(
            Map<String, ?> evidence,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        Optional<Path> packageDirectory = pathValue(evidence, "packageDirectory");
        if (packageDirectory.isEmpty()) {
            return;
        }
        for (Map.Entry<String, ?> entry : packageArtifacts.entrySet()) {
            if (entry.getValue() instanceof Map<?, ?> referenceMap) {
                validateReferencePathWithin(
                        packageDirectory.orElseThrow(),
                        immutableMap(referenceMap),
                        "file",
                        "verification evidence packageArtifacts." + entry.getKey(),
                        "packageDirectory",
                        failures);
            }
        }
    }

    private static void validateReferencePathWithin(
            Path root,
            Map<String, ?> reference,
            String key,
            String owner,
            String rootLabel,
            List<String> failures) {
        Optional<Path> path = pathValue(reference, key);
        if (path.isPresent() && !isWithin(root, path.orElseThrow())) {
            failures.add(owner + "." + key + " is outside " + rootLabel + ": " + path.orElseThrow());
        }
    }

    private static boolean isWithin(Path root, Path candidate) {
        return candidate.toAbsolutePath().normalize().startsWith(root.toAbsolutePath().normalize());
    }
}
