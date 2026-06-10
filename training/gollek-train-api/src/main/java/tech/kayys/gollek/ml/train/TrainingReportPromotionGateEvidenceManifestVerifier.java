package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.stringValue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader/verifier for the promotion-gate evidence manifest.
 */
final class TrainingReportPromotionGateEvidenceManifestVerifier {
    private TrainingReportPromotionGateEvidenceManifestVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection read(Path evidenceFile)
            throws IOException {
        Path resolvedEvidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedEvidenceFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification evidence JSON at "
                    + resolvedEvidenceFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification evidence at "
                    + resolvedEvidenceFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection(
                resolvedEvidenceFile,
                TrainingReportArtifactFingerprint.of(resolvedEvidenceFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verify(
            Path evidenceFile,
            String expectedEvidenceSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection =
                read(evidenceFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedEvidenceSha256 == null || expectedEvidenceSha256.isBlank()
                ? null
                : expectedEvidenceSha256.trim();
        boolean evidenceSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.evidenceSha256());
        if (!evidenceSha256Matches) {
            failures.add("Verification evidence checksum mismatch for " + inspection.evidenceFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.evidenceSha256());
        }
        boolean schemaValid = TrainingReportPromotionGateEvidenceManifestSchema.verify(inspection, failures);
        boolean evidenceFilesValid = verifyEvidenceFilesReferences(inspection, failures);
        boolean packageArtifactsValid = verifyPackageArtifactReferences(inspection, failures);
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification(
                inspection,
                normalizedExpectedSha256,
                evidenceSha256Matches,
                schemaValid,
                schemaValid && evidenceFilesValid,
                schemaValid && packageArtifactsValid,
                failures);
    }

    private static boolean verifyEvidenceFilesReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> evidenceFiles = objectValue(inspection.evidence(), "evidenceFiles").orElse(Map.of());
        for (Map.Entry<String, Object> entry : evidenceFiles.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                failures.add("Verification evidence evidenceFiles." + entry.getKey() + " must be an object");
                continue;
            }
            verifyFileReference(
                    "evidenceFiles." + entry.getKey(),
                    immutableMap(referenceMap),
                    null,
                    failures);
        }
        return failures.size() == before;
    }

    private static boolean verifyPackageArtifactReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> packageArtifacts =
                objectValue(inspection.evidence(), "packageArtifacts").orElse(Map.of());
        for (Map.Entry<String, Object> entry : packageArtifacts.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                failures.add("Verification evidence packageArtifacts." + entry.getKey() + " must be an object");
                continue;
            }
            Map<String, Object> reference = immutableMap(referenceMap);
            verifyFileReference(
                    "packageArtifacts." + entry.getKey(),
                    reference,
                    longValue(reference, "bytes").orElse(null),
                    failures);
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
            failures.add("Verification evidence " + label + " is missing file");
            return false;
        }
        if (expectedSha256 == null || expectedSha256.isBlank()) {
            failures.add("Verification evidence " + label + " is missing sha256");
            return false;
        }
        Path path;
        try {
            path = Path.of(file).toAbsolutePath().normalize();
        } catch (InvalidPathException error) {
            failures.add("Verification evidence " + label + " has invalid file path: " + file);
            return false;
        }
        if (!Files.isRegularFile(path)) {
            failures.add("Verification evidence " + label + " file is missing: " + path);
            return false;
        }
        TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(path);
        if (expectedBytes != null) {
            if (fingerprint.bytes() != expectedBytes.longValue()) {
                failures.add("Verification evidence " + label + " byte count mismatch for " + path
                        + ": expected " + expectedBytes + " but found " + fingerprint.bytes());
            }
        }
        if (!expectedSha256.equalsIgnoreCase(fingerprint.sha256())) {
            failures.add("Verification evidence " + label + " checksum mismatch for " + path
                    + ": expected " + expectedSha256 + " but found " + fingerprint.sha256());
        }
        return failures.size() == before;
    }
}
