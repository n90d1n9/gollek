package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class TrainingReportPromotionGateArtifactPackageFingerprints {
    private TrainingReportPromotionGateArtifactPackageFingerprints() {
    }

    static String sha256(Path file) throws IOException {
        return TrainingReportArtifactFingerprint.of(file).sha256();
    }

    static void verifySourceReportSnapshot(
            TrainingReportPromotionGateArtifactPackage.SourceReportSnapshot snapshot,
            List<String> failures) throws IOException {
        TrainingReportPromotionArtifacts.SourceReport sourceReport = snapshot.sourceReport();
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact = snapshot.artifact();
        if (!snapshot.manifestBytesMatchSource()) {
            failures.add("Packaged source report snapshot byte size does not match review provenance for "
                    + sourceReport.role() + " " + sourceReport.name());
        }
        if (!snapshot.manifestSha256MatchesSource()) {
            failures.add("Packaged source report snapshot SHA-256 does not match review provenance for "
                    + sourceReport.role() + " " + sourceReport.name());
        }
        if (!Files.isRegularFile(artifact.file())) {
            failures.add("Packaged source report snapshot is missing for "
                    + sourceReport.role() + " " + sourceReport.name() + ": " + artifact.file());
            return;
        }
        TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(artifact.file());
        if (fingerprint.bytes() != artifact.bytes()) {
            failures.add("Packaged source report snapshot byte size mismatch for " + artifact.file()
                    + " (expected " + artifact.bytes() + " bytes, got " + fingerprint.bytes() + " bytes)");
        }
        if (!artifact.sha256().equalsIgnoreCase(fingerprint.sha256())) {
            failures.add("Packaged source report snapshot SHA-256 mismatch for " + artifact.file()
                    + " (expected " + artifact.sha256() + ", got " + fingerprint.sha256() + ")");
        }
    }

    static void requireUnchangedNonRefreshableArtifact(
            TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact) throws IOException {
        if (!Files.isRegularFile(artifact.file())) {
            throw new IOException("Cannot refresh promotion package because non-refreshable artifact is missing: "
                    + artifact.name() + " (" + artifact.file() + ")");
        }
        TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(artifact.file());
        if (fingerprint.bytes() != artifact.bytes()) {
            throw new IOException("Cannot refresh promotion package because non-refreshable artifact byte size "
                    + "changed: " + artifact.name() + " (" + artifact.file() + ")");
        }
        if (!artifact.sha256().equalsIgnoreCase(fingerprint.sha256())) {
            throw new IOException("Cannot refresh promotion package because non-refreshable artifact SHA-256 "
                    + "changed: " + artifact.name() + " (" + artifact.file() + ")");
        }
    }
}
