package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
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
        Objects.requireNonNull(artifacts, "artifacts must not be null");
        Objects.requireNonNull(generatedAt, "generatedAt must not be null");

        Properties manifest = new Properties();
        manifest.setProperty("formatVersion", Integer.toString(formatVersion));
        manifest.setProperty("generatedAt", generatedAt.toString());
        for (Map.Entry<String, Path> artifact : artifacts.entrySet()) {
            addArtifact(manifest, artifact.getKey(), artifact.getValue());
        }
        return manifest;
    }

    static void write(
            Path manifestFile,
            Map<String, Path> artifacts,
            int formatVersion,
            Instant generatedAt) throws IOException {
        if (manifestFile == null) {
            return;
        }
        Properties manifest = build(artifacts, formatVersion, generatedAt);
        TrainerCheckpointIO.writePropertiesAtomically(
                manifestFile,
                manifest,
                "Gollek canonical trainer checkpoint manifest");
    }

    static CompatibilityCheck checkArtifact(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
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

        String mismatch = TrainerCheckpointFileIntegrity.manifestArtifactMismatch(
                manifest,
                artifactName,
                artifactFile,
                supportedManifestVersion);
        if (mismatch != null) {
            return CompatibilityCheck.incompatible(mismatch, true, false, null);
        }
        return CompatibilityCheck.ok(true, false, null);
    }

    private static void addArtifact(Properties manifest, String artifactName, Path artifactFile) throws IOException {
        if (artifactFile == null || !Files.isRegularFile(artifactFile)) {
            return;
        }
        String prefix = "artifact." + artifactName + '.';
        manifest.setProperty(prefix + "file", artifactFile.getFileName().toString());
        manifest.setProperty(prefix + "bytes", Long.toString(Files.size(artifactFile)));
        manifest.setProperty(prefix + "sha256", TrainerCheckpointIO.sha256Hex(artifactFile));
    }

    record CompatibilityCheck(
            TrainerCheckpointCompatibilityReport report,
            boolean loaded,
            boolean missing,
            String loadError) {
        static CompatibilityCheck ok(boolean loaded, boolean missing, String loadError) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.ok(),
                    loaded,
                    missing,
                    loadError);
        }

        static CompatibilityCheck incompatible(
                String error,
                boolean loaded,
                boolean missing,
                String loadError) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.incompatible(error),
                    loaded,
                    missing,
                    loadError);
        }
    }
}
