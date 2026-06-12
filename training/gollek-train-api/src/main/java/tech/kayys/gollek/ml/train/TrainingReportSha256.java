package tech.kayys.gollek.ml.train;

import java.util.Locale;

/**
 * Shared SHA-256 normalization rules for training report artifact and receipt verification.
 */
final class TrainingReportSha256 {
    private TrainingReportSha256() {
    }

    static String normalizeOptional(String value, String fieldName) {
        return value == null || value.isBlank() ? null : require(value, fieldName);
    }

    static String require(String value, String fieldName) {
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
