package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Stable file identity used by trainer report receipts and artifact manifests.
 */
public record TrainingReportArtifactFingerprint(
        Path file,
        long bytes,
        String sha256) {
    public TrainingReportArtifactFingerprint {
        file = Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize();
        if (bytes < 0L) {
            throw new IllegalArgumentException("bytes must be non-negative");
        }
        sha256 = requireSha256(sha256, "sha256");
    }

    public static TrainingReportArtifactFingerprint of(Path file) throws IOException {
        Path resolved = Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize();
        return new TrainingReportArtifactFingerprint(
                resolved,
                Files.size(resolved),
                TrainerCheckpointIO.sha256Hex(resolved));
    }

    public static TrainingReportArtifactFingerprint fromMap(Map<?, ?> map, String fieldName) {
        Objects.requireNonNull(map, fieldName + " must not be null");
        Path file = pathValue(map.get("file"), fieldName + ".file");
        long bytes = longValue(map.get("bytes"), fieldName + ".bytes");
        String sha256 = stringValue(map.get("sha256"), fieldName + ".sha256");
        return new TrainingReportArtifactFingerprint(file, bytes, sha256);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("file", file.toString());
        map.put("bytes", bytes);
        map.put("sha256", sha256);
        return Map.copyOf(map);
    }

    public void verifyMatches(
            String name,
            TrainingReportArtifactFingerprint actual,
            java.util.List<String> failures) {
        Objects.requireNonNull(actual, "actual must not be null");
        Objects.requireNonNull(failures, "failures must not be null");
        if (!file.equals(actual.file())) {
            failures.add(name + " fingerprint file mismatch: expected " + file + ", got " + actual.file());
        }
        if (bytes != actual.bytes()) {
            failures.add(name + " fingerprint byte size mismatch for " + file
                    + ": expected " + bytes + ", got " + actual.bytes());
        }
        if (!sha256.equalsIgnoreCase(actual.sha256())) {
            failures.add(name + " fingerprint SHA-256 mismatch for " + file);
        }
    }

    private static Path pathValue(Object value, String fieldName) {
        if (!(value instanceof String text) || text.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must be a non-blank path string");
        }
        try {
            return Path.of(text.trim()).toAbsolutePath().normalize();
        } catch (InvalidPathException error) {
            throw new IllegalArgumentException(fieldName + " is not a valid path: " + text, error);
        }
    }

    private static long longValue(Object value, String fieldName) {
        if (value instanceof Number number) {
            long resolved = number.longValue();
            if (resolved >= 0L) {
                return resolved;
            }
        }
        throw new IllegalArgumentException(fieldName + " must be a non-negative number");
    }

    private static String stringValue(Object value, String fieldName) {
        if (value instanceof String text && !text.isBlank()) {
            return requireSha256(text, fieldName);
        }
        throw new IllegalArgumentException(fieldName + " must be a SHA-256 string");
    }

    private static String requireSha256(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (!normalized.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(fieldName + " must be a 64-character SHA-256 hex string");
        }
        return normalized;
    }
}
