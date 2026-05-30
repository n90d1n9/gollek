package tech.kayys.gollek.spi.feature;

import java.util.Locale;
import java.util.Set;

/**
 * Common feature adapter kind names.
 *
 * <p>The list is descriptive, not closed. Custom projects may publish new
 * lower-case kind names when a domain does not fit these constants.</p>
 */
public final class FeatureAdapterKinds {
    public static final String PIPELINE = "pipeline";
    public static final String RUNNER = "runner";
    public static final String BACKEND = "backend";
    public static final String TRAINING = "training";
    public static final String OPTIMIZATION = "optimization";
    public static final String QUANTIZATION = "quantization";
    public static final String CONVERTER = "converter";
    public static final String EXPORTER = "exporter";
    public static final String DATASET = "dataset";
    public static final String EVALUATOR = "evaluator";
    public static final String TOOLING = "tooling";
    public static final String ADAPTER = "adapter";
    public static final String CAPABILITY = "capability";

    public static final Set<String> COMMON_KINDS = Set.of(
            PIPELINE,
            RUNNER,
            BACKEND,
            TRAINING,
            OPTIMIZATION,
            QUANTIZATION,
            CONVERTER,
            EXPORTER,
            DATASET,
            EVALUATOR,
            TOOLING,
            ADAPTER,
            CAPABILITY);

    private FeatureAdapterKinds() {
    }

    public static String normalize(String kind) {
        String normalized = kind == null ? "" : kind.trim()
                .toLowerCase(Locale.ROOT)
                .replace('_', '-')
                .replaceAll("[^a-z0-9.-]+", "-")
                .replaceAll("[-.]{2,}", "-")
                .replaceAll("^[-.]+|[-.]+$", "");
        return normalized.isBlank() ? CAPABILITY : normalized;
    }

    public static boolean isCommon(String kind) {
        return COMMON_KINDS.contains(normalize(kind));
    }
}
