package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Shared integrity checks for trainer checkpoint files and manifests.
 */
final class TrainerCheckpointFileIntegrity {
    private TrainerCheckpointFileIntegrity() {
    }

    static String modelCheckpointMismatch(Properties metadata, Path checkpointFile) {
        if (checkpointFile == null || !Files.isRegularFile(checkpointFile)) {
            return null;
        }

        String sizeMismatch = byteSizeMismatch(
                metadata.getProperty("modelCheckpointBytes"),
                "model checkpoint",
                checkpointFile,
                "model checkpoint byte size metadata");
        if (sizeMismatch != null) {
            return sizeMismatch;
        }
        return sha256Mismatch(
                metadata.getProperty("modelCheckpointSha256"),
                "model checkpoint",
                checkpointFile);
    }

    static String manifestArtifactMismatch(
            Properties manifest,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        return manifestArtifactMismatch(manifest, artifactName, artifactFile, supportedManifestVersion, false);
    }

    static String requiredManifestArtifactMismatch(
            Properties manifest,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        return manifestArtifactMismatch(manifest, artifactName, artifactFile, supportedManifestVersion, true);
    }

    private static String manifestArtifactMismatch(
            Properties manifest,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion,
            boolean requireManifestEntry) {
        String versionMismatch = formatVersionMismatch(
                manifest.getProperty("formatVersion"),
                "checkpoint manifest",
                supportedManifestVersion);
        if (versionMismatch != null) {
            return versionMismatch;
        }
        if (artifactFile == null || !Files.isRegularFile(artifactFile)) {
            return null;
        }

        String prefix = "artifact." + artifactName + '.';
        String expectedFile = manifest.getProperty(prefix + "file");
        String expectedBytes = manifest.getProperty(prefix + "bytes");
        String expectedSha256 = manifest.getProperty(prefix + "sha256");
        if (expectedFile == null && expectedBytes == null && expectedSha256 == null) {
            return requireManifestEntry
                    ? artifactName + " checkpoint is missing from checkpoint manifest"
                    : null;
        }
        if (expectedFile != null && !expectedFile.equals(artifactFile.getFileName().toString())) {
            return artifactName + " checkpoint file mismatch (expected "
                    + expectedFile + ", got " + artifactFile.getFileName() + ")";
        }

        String sizeMismatch = byteSizeMismatch(
                expectedBytes,
                artifactName + " checkpoint",
                artifactFile,
                artifactName + " checkpoint byte size metadata");
        if (sizeMismatch != null) {
            return sizeMismatch;
        }
        return sha256Mismatch(expectedSha256, artifactName + " checkpoint", artifactFile);
    }

    static String formatVersionMismatch(String rawVersion, String label, int supportedVersion) {
        if (rawVersion == null) {
            return null;
        }
        try {
            int version = Integer.parseInt(rawVersion.trim());
            if (version == supportedVersion) {
                return null;
            }
            return "unsupported " + label + " format version " + version
                    + " (supported: " + supportedVersion + ")";
        } catch (NumberFormatException error) {
            return "invalid " + label + " format version: " + rawVersion;
        }
    }

    private static String byteSizeMismatch(
            String expectedBytes,
            String label,
            Path artifactFile,
            String metadataLabel) {
        if (expectedBytes == null) {
            return null;
        }

        long expectedSize;
        try {
            expectedSize = Long.parseLong(expectedBytes.trim());
        } catch (NumberFormatException error) {
            return "invalid " + metadataLabel + ": " + expectedBytes;
        }
        try {
            long actualSize = Files.size(artifactFile);
            if (expectedSize == actualSize) {
                return null;
            }
            return label + " size mismatch (expected "
                    + expectedSize + " bytes, got " + actualSize + " bytes)";
        } catch (IOException error) {
            return label + " size could not be read: " + error.getMessage();
        }
    }

    private static String sha256Mismatch(String expectedSha256, String label, Path artifactFile) {
        if (expectedSha256 == null || expectedSha256.isBlank()) {
            return null;
        }
        try {
            String actualSha256 = TrainerCheckpointIO.sha256Hex(artifactFile);
            if (expectedSha256.equalsIgnoreCase(actualSha256)) {
                return null;
            }
            return label + " SHA-256 mismatch (expected " + expectedSha256 + ", got " + actualSha256 + ")";
        } catch (IOException error) {
            return label + " SHA-256 could not be read: " + error.getMessage();
        }
    }
}
