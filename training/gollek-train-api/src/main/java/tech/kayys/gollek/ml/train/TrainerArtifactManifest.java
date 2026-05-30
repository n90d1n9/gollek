package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.TreeSet;

/**
 * Shared manifest primitives for trainer artifacts that need file, size and SHA-256 provenance.
 */
final class TrainerArtifactManifest {
    private TrainerArtifactManifest() {
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
            Instant generatedAt,
            String comment) throws IOException {
        if (manifestFile == null) {
            return;
        }
        TrainerCheckpointIO.writePropertiesAtomically(
                manifestFile,
                build(artifacts, formatVersion, generatedAt),
                comment);
    }

    static Inspection read(Path manifestFile) throws IOException {
        Path resolvedManifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                .toAbsolutePath()
                .normalize();
        Properties properties = new Properties();
        try (Reader reader = Files.newBufferedReader(resolvedManifestFile, StandardCharsets.UTF_8)) {
            properties.load(reader);
        }
        return new Inspection(
                resolvedManifestFile,
                formatVersion(properties),
                generatedAt(properties),
                TrainerCheckpointIO.sha256Hex(resolvedManifestFile),
                stringProperties(properties),
                artifactEntries(properties));
    }

    private static void addArtifact(Properties manifest, String artifactName, Path artifactFile) throws IOException {
        if (artifactName == null
                || artifactName.isBlank()
                || artifactFile == null
                || !Files.isRegularFile(artifactFile)) {
            return;
        }
        String prefix = "artifact." + artifactName.trim() + '.';
        manifest.setProperty(prefix + "file", artifactFile.getFileName().toString());
        manifest.setProperty(prefix + "bytes", Long.toString(Files.size(artifactFile)));
        manifest.setProperty(prefix + "sha256", TrainerCheckpointIO.sha256Hex(artifactFile));
    }

    private static int formatVersion(Properties properties) throws IOException {
        String raw = properties.getProperty("formatVersion");
        if (raw == null || raw.isBlank()) {
            throw new IOException("artifact manifest is missing formatVersion");
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException error) {
            throw new IOException("invalid artifact manifest format version: " + raw, error);
        }
    }

    private static Instant generatedAt(Properties properties) throws IOException {
        String raw = properties.getProperty("generatedAt");
        if (raw == null || raw.isBlank()) {
            throw new IOException("artifact manifest is missing generatedAt");
        }
        try {
            return Instant.parse(raw.trim());
        } catch (RuntimeException error) {
            throw new IOException("invalid artifact manifest generatedAt: " + raw, error);
        }
    }

    private static Map<String, String> stringProperties(Properties properties) {
        Map<String, String> values = new LinkedHashMap<>();
        for (String name : new TreeSet<>(properties.stringPropertyNames())) {
            values.put(name, properties.getProperty(name));
        }
        return Map.copyOf(values);
    }

    private static Map<String, ArtifactEntry> artifactEntries(Properties properties) throws IOException {
        Map<String, ArtifactEntry> entries = new LinkedHashMap<>();
        for (String name : new TreeSet<>(properties.stringPropertyNames())) {
            if (!name.startsWith("artifact.") || !name.endsWith(".file")) {
                continue;
            }
            String artifactName = name.substring("artifact.".length(), name.length() - ".file".length());
            String prefix = "artifact." + artifactName + '.';
            String fileName = properties.getProperty(prefix + "file");
            String bytes = properties.getProperty(prefix + "bytes");
            String sha256 = properties.getProperty(prefix + "sha256");
            if (fileName == null || fileName.isBlank()) {
                throw new IOException("artifact manifest entry is missing file for " + artifactName);
            }
            if (bytes == null || bytes.isBlank()) {
                throw new IOException("artifact manifest entry is missing bytes for " + artifactName);
            }
            if (sha256 == null || sha256.isBlank()) {
                throw new IOException("artifact manifest entry is missing sha256 for " + artifactName);
            }
            long parsedBytes;
            try {
                parsedBytes = Long.parseLong(bytes.trim());
            } catch (NumberFormatException error) {
                throw new IOException("invalid artifact manifest byte size for " + artifactName + ": " + bytes, error);
            }
            if (parsedBytes < 0) {
                throw new IOException("invalid artifact manifest byte size for " + artifactName + ": " + bytes);
            }
            entries.put(artifactName, new ArtifactEntry(
                    artifactName,
                    safeArtifactFileName(fileName, artifactName),
                    parsedBytes,
                    validSha256(sha256, artifactName)));
        }
        return Map.copyOf(entries);
    }

    private static String safeArtifactFileName(String rawFileName, String artifactName) throws IOException {
        String fileName = rawFileName.trim();
        Path path = Path.of(fileName);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(fileName) || "..".equals(fileName)) {
            throw new IOException("unsafe artifact manifest file name for " + artifactName + ": " + rawFileName);
        }
        return fileName;
    }

    private static String validSha256(String rawSha256, String artifactName) throws IOException {
        String sha256 = rawSha256.trim().toLowerCase(java.util.Locale.ROOT);
        if (sha256.length() != 64) {
            throw new IOException("invalid artifact manifest sha256 for " + artifactName + ": " + rawSha256);
        }
        for (int index = 0; index < sha256.length(); index++) {
            char value = sha256.charAt(index);
            if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
                throw new IOException("invalid artifact manifest sha256 for " + artifactName + ": " + rawSha256);
            }
        }
        return sha256;
    }

    record ArtifactEntry(String name, String fileName, long bytes, String sha256) {
        ArtifactEntry {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            if (fileName == null || fileName.isBlank()) {
                throw new IllegalArgumentException("fileName must not be blank");
            }
            if (bytes < 0) {
                throw new IllegalArgumentException("bytes must be non-negative");
            }
            if (sha256 == null || sha256.isBlank()) {
                throw new IllegalArgumentException("sha256 must not be blank");
            }
        }

        Path resolve(Path directory) {
            return directory.resolve(fileName).toAbsolutePath().normalize();
        }
    }

    record Inspection(
            Path manifestFile,
            int formatVersion,
            Instant generatedAt,
            String sha256,
            Map<String, String> properties,
            Map<String, ArtifactEntry> artifacts) {
        Inspection {
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            generatedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
            if (sha256 == null || sha256.isBlank()) {
                throw new IllegalArgumentException("sha256 must not be blank");
            }
            properties = Map.copyOf(Objects.requireNonNull(properties, "properties must not be null"));
            artifacts = Map.copyOf(Objects.requireNonNull(artifacts, "artifacts must not be null"));
        }
    }
}
