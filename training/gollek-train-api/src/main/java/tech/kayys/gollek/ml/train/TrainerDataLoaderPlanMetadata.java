package tech.kayys.gollek.ml.train;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.train.data.DataLoaderPlan;
import tech.kayys.gollek.train.data.DataLoaderPrefetchPlan;
import tech.kayys.gollek.train.data.PrefetchingIterable;

/**
 * Captures non-consuming loader plan metadata for trainer reports.
 */
final class TrainerDataLoaderPlanMetadata {
    private TrainerDataLoaderPlanMetadata() {
    }

    static Map<String, Object> capture(
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataLoaderPlanMetadataCaptured", Boolean.TRUE);
        putLoader(metadata, "trainLoaderPlan", trainLoader);
        putLoader(metadata, "validationLoaderPlan", validationLoader);
        return Map.copyOf(metadata);
    }

    private static void putLoader(
            Map<String, Object> metadata,
            String prefix,
            Iterable<Batch> loader) {
        if (loader == null) {
            markUnavailable(metadata, prefix, "no-loader", null);
            return;
        }

        metadata.put(prefix + ".loaderType", loader.getClass().getName());
        if (loader instanceof PrefetchingIterable<?> prefetched) {
            putPrefetchPlan(metadata, prefix, prefetched.plan());
            Iterable<?> source = prefetched.source();
            metadata.put(prefix + ".prefetch.sourceLoaderType", source.getClass().getName());
            putNonPrefetchedLoader(metadata, prefix, castBatchIterable(source));
            return;
        }
        putPrefetchPlan(metadata, prefix, DataLoaderPrefetchPlan.disabled());
        putNonPrefetchedLoader(metadata, prefix, loader);
    }

    private static void putNonPrefetchedLoader(
            Map<String, Object> metadata,
            String prefix,
            Iterable<Batch> loader) {
        if (loader instanceof DataLoader.TensorDataLoader tensorLoader) {
            putPlan(metadata, prefix, tensorLoader.plan());
            metadata.put(prefix + ".derivedFromBatchCollection", Boolean.FALSE);
            return;
        }
        if (loader instanceof Collection<?> collection) {
            putCollectionPlan(metadata, prefix, collection);
            return;
        }

        markUnavailable(metadata, prefix, "unsupported-loader", loader.getClass().getName());
    }

    private static void putCollectionPlan(
            Map<String, Object> metadata,
            String prefix,
            Collection<?> collection) {
        if (collection.isEmpty()) {
            markUnavailable(metadata, prefix, "empty-batch-collection", collection.getClass().getName());
            return;
        }

        int batchCount = collection.size();
        long sampleCount = 0L;
        int maxBatchSize = 0;
        for (Object item : collection) {
            if (!(item instanceof Batch batch)) {
                markUnavailable(metadata, prefix, "collection-item-not-batch", collection.getClass().getName());
                metadata.put(prefix + ".invalidItemType", item == null ? "null" : item.getClass().getName());
                return;
            }
            int batchSamples = safeInt(TrainerBatchGuards.sampleCount(batch), "batch sample count");
            sampleCount += batchSamples;
            maxBatchSize = Math.max(maxBatchSize, batchSamples);
        }

        DataLoaderPlan plan = new DataLoaderPlan(
                "batch-collection",
                safeInt(sampleCount, "sample count"),
                safeInt(sampleCount, "sample count"),
                Math.max(1, maxBatchSize),
                batchCount,
                false,
                false,
                false,
                null,
                false,
                0L);
        putPlan(metadata, prefix, plan);
        metadata.put(prefix + ".derivedFromBatchCollection", Boolean.TRUE);
    }

    private static void putPlan(
            Map<String, Object> metadata,
            String prefix,
            DataLoaderPlan plan) {
        metadata.put(prefix + ".available", Boolean.TRUE);
        metadata.put(prefix + ".skipped", Boolean.FALSE);
        metadata.putAll(plan.toMetadata(prefix));
        metadata.put(prefix + ".summary", plan.summary());
    }

    private static void putPrefetchPlan(
            Map<String, Object> metadata,
            String prefix,
            DataLoaderPrefetchPlan plan) {
        metadata.putAll(plan.toMetadata(prefix + ".prefetch"));
    }

    private static void markUnavailable(
            Map<String, Object> metadata,
            String prefix,
            String reason,
            String loaderType) {
        metadata.put(prefix + ".available", Boolean.FALSE);
        metadata.put(prefix + ".skipped", Boolean.TRUE);
        metadata.put(prefix + ".skipReason", reason);
        if (loaderType != null) {
            metadata.put(prefix + ".loaderType", loaderType);
        }
    }

    private static int safeInt(long value, String fieldName) {
        if (value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(fieldName + " exceeds int range: " + value);
        }
        return (int) value;
    }

    @SuppressWarnings("unchecked")
    private static Iterable<Batch> castBatchIterable(Iterable<?> iterable) {
        return (Iterable<Batch>) iterable;
    }
}
