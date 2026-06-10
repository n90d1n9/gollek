package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.stringValue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Integrity checks for file references recorded in a verification index.
 */
final class TrainingReportPromotionGateVerificationIndexFileReferences {
    private TrainingReportPromotionGateVerificationIndexFileReferences() {
    }

    static boolean verify(
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
        TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(path);
        if (expectedBytes != null && fingerprint.bytes() != expectedBytes.longValue()) {
            failures.add("Verification index " + label + " byte count mismatch for " + path
                    + ": expected " + expectedBytes + " but found " + fingerprint.bytes());
        }
        if (!expectedSha256.equalsIgnoreCase(fingerprint.sha256())) {
            failures.add("Verification index " + label + " checksum mismatch for " + path
                    + ": expected " + expectedSha256 + " but found " + fingerprint.sha256());
        }
        return failures.size() == before;
    }

    static void verifyShape(
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
}
