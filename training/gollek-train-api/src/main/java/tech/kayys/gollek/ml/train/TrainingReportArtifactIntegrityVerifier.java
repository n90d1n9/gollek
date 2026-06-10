package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

final class TrainingReportArtifactIntegrityVerifier {
    private TrainingReportArtifactIntegrityVerifier() {
    }

    record Result(boolean bytesMatch, boolean sha256Match) {
    }

    static Result verifyDetailedArtifact(
            String name,
            Path file,
            long expectedBytes,
            String expectedSha256,
            List<String> failures) {
        Objects.requireNonNull(failures, "failures must not be null");
        TrainingReportArtifactFingerprint actual = readFingerprint(name, file, failures);
        if (actual == null) {
            return new Result(false, false);
        }

        boolean bytesMatch = true;
        boolean sha256Match = true;
        if (actual.bytes() != expectedBytes) {
            failures.add(name + " artifact byte size mismatch for " + actual.file()
                    + " (expected " + expectedBytes + " bytes, got " + actual.bytes() + " bytes)");
            bytesMatch = false;
        }
        if (expectedSha256 == null || !actual.sha256().equalsIgnoreCase(expectedSha256)) {
            failures.add(name + " artifact SHA-256 mismatch for " + actual.file()
                    + " (expected " + expectedSha256 + ", got " + actual.sha256() + ")");
            sha256Match = false;
        }
        return new Result(bytesMatch, sha256Match);
    }

    private static TrainingReportArtifactFingerprint readFingerprint(String name, Path file, List<String> failures) {
        try {
            return TrainingReportArtifactFingerprint.of(file);
        } catch (IOException error) {
            failures.add(name + " artifact fingerprint could not be read for " + file
                    + ": " + error.getMessage());
            return null;
        }
    }
}
