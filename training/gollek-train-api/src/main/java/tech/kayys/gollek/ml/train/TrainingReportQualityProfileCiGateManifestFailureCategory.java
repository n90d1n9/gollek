package tech.kayys.gollek.ml.train;

import java.util.Locale;
import java.util.Optional;

/**
 * Stable failure buckets for quality-profile CI manifest verification.
 */
public enum TrainingReportQualityProfileCiGateManifestFailureCategory {
    CHECKSUM("checksum"),
    FORMAT("format"),
    PROFILE("profile"),
    MARKDOWN("markdown"),
    STRUCTURE("structure"),
    ARTIFACT("artifact"),
    UNKNOWN("unknown");

    private final String id;

    TrainingReportQualityProfileCiGateManifestFailureCategory(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static Optional<TrainingReportQualityProfileCiGateManifestFailureCategory> fromId(String id) {
        if (id == null || id.isBlank()) {
            return Optional.empty();
        }
        String normalized = id.trim().toLowerCase(Locale.ROOT);
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category : values()) {
            if (category.id.equals(normalized)) {
                return Optional.of(category);
            }
        }
        return Optional.empty();
    }
}
