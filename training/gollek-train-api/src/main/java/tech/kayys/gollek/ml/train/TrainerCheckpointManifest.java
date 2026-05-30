package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Properties;

/**
 * Builds the checkpoint manifest used to verify resumed trainer artifacts.
 */
final class TrainerCheckpointManifest {
    private TrainerCheckpointManifest() {
    }

    static Properties build(
            Map<String, Path> artifacts,
            int formatVersion,
            Instant generatedAt) throws IOException {
        return TrainerArtifactManifest.build(artifacts, formatVersion, generatedAt);
    }

    static void write(
            Path manifestFile,
            Map<String, Path> artifacts,
            int formatVersion,
            Instant generatedAt) throws IOException {
        if (manifestFile == null) {
            return;
        }
        TrainerArtifactManifest.write(
                manifestFile,
                artifacts,
                formatVersion,
                generatedAt,
                "Gollek canonical trainer checkpoint manifest");
    }

    static CompatibilityCheck checkArtifact(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        return checkArtifact(manifestFile, artifactName, artifactFile, supportedManifestVersion, false);
    }

    static CompatibilityCheck checkRequiredArtifact(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        return checkArtifact(manifestFile, artifactName, artifactFile, supportedManifestVersion, true);
    }

    private static CompatibilityCheck checkArtifact(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion,
            boolean requireManifestEntry) {
        if (manifestFile == null || artifactFile == null || !Files.isRegularFile(artifactFile)) {
            return CompatibilityCheck.ok(false, false, null);
        }
        if (!Files.isRegularFile(manifestFile)) {
            return CompatibilityCheck.ok(false, true, null);
        }

        Properties manifest = new Properties();
        try (Reader reader = Files.newBufferedReader(manifestFile, StandardCharsets.UTF_8)) {
            manifest.load(reader);
        } catch (IOException error) {
            String message = "checkpoint manifest could not be read: " + error.getMessage();
            return CompatibilityCheck.incompatible(message, false, false, error.getMessage());
        }

        if (requireManifestEntry && !hasArtifactEntry(manifest, artifactName)) {
            return CompatibilityCheck.incompatible(
                    artifactName + " checkpoint is missing from checkpoint manifest",
                    true,
                    false,
                    null,
                    true);
        }

        String mismatch = requireManifestEntry
                ? TrainerCheckpointFileIntegrity.requiredManifestArtifactMismatch(
                        manifest,
                        artifactName,
                        artifactFile,
                        supportedManifestVersion)
                : TrainerCheckpointFileIntegrity.manifestArtifactMismatch(
                        manifest,
                        artifactName,
                        artifactFile,
                        supportedManifestVersion);
        if (mismatch != null) {
            return CompatibilityCheck.incompatible(mismatch, true, false, null);
        }
        return CompatibilityCheck.ok(true, false, null);
    }

    private static boolean hasArtifactEntry(Properties manifest, String artifactName) {
        String prefix = "artifact." + artifactName + '.';
        return manifest.getProperty(prefix + "file") != null
                || manifest.getProperty(prefix + "bytes") != null
                || manifest.getProperty(prefix + "sha256") != null;
    }

    record CompatibilityCheck(
            TrainerCheckpointCompatibilityReport report,
            boolean loaded,
            boolean missing,
            String loadError,
            boolean manifestEntryMissing) {
        static CompatibilityCheck ok(boolean loaded, boolean missing, String loadError) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.ok(),
                    loaded,
                    missing,
                    loadError,
                    false);
        }

        static CompatibilityCheck incompatible(
                String error,
                boolean loaded,
                boolean missing,
                String loadError) {
            return incompatible(error, loaded, missing, loadError, false);
        }

        static CompatibilityCheck incompatible(
                String error,
                boolean loaded,
                boolean missing,
                String loadError,
                boolean manifestEntryMissing) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.incompatible(error),
                    loaded,
                    missing,
                    loadError,
                    manifestEntryMissing);
        }
    }
}
