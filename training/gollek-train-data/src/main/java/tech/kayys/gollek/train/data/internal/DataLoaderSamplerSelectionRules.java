package tech.kayys.gollek.train.data.internal;

import java.util.Objects;

public final class DataLoaderSamplerSelectionRules {
    private DataLoaderSamplerSelectionRules() {
    }

    public static <T> T requireSampler(T sampler) {
        return Objects.requireNonNull(sampler, "sampler must not be null");
    }

    public static <T> T requireBatchSampler(T batchSampler) {
        return Objects.requireNonNull(batchSampler, "batchSampler must not be null");
    }

    public static void requireExclusive(Object sampler, Object batchSampler) {
        if (sampler != null && batchSampler != null) {
            throw new IllegalArgumentException("sampler and batchSampler cannot both be configured");
        }
    }
}
