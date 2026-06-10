package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.data.DataLoaderPrefetchPlan;

/**
 * Adds concrete prefetch tuning findings when runtime input timings identify loader iteration as the bottleneck.
 */
final class TrainingReportRuntimeInputPrefetchAdvisor {
    private static final int MIN_RECOMMENDED_BUFFERED_ITEMS = DataLoaderPrefetchPlan.DEFAULT_BUFFER_SIZE;

    private TrainingReportRuntimeInputPrefetchAdvisor() {
    }

    static void addFindings(
            List<TrainingReportRuntimeInputProfileGate.Finding> findings,
            Map<String, ?> metadata,
            Map<String, ?> inputProfile) {
        if (findings == null || metadata == null || inputProfile == null) {
            return;
        }
        if (!"train".equals(stringValue(inputProfile.get("dominantScope")))) {
            return;
        }

        Map<String, ?> train = mapValue(inputProfile.get("train"));
        if (!"next".equals(stringValue(train.get("dominantStage")))) {
            return;
        }

        String prefix = "trainLoaderPlan";
        String enabledKey = prefix + ".prefetch.enabled";
        if (!metadata.containsKey(enabledKey)) {
            return;
        }

        boolean enabled = booleanValue(metadata.get(enabledKey));
        if (!enabled) {
            findings.add(new TrainingReportRuntimeInputProfileGate.Finding(
                    "runtime-input-train-prefetch-disabled",
                    "warning",
                    "Training input `next()` dominates while train loader prefetching is disabled.",
                    "Wrap the train loader with `DataLoader.prefetch(" + MIN_RECOMMENDED_BUFFERED_ITEMS
                            + ")` or `loader.prefetch(" + MIN_RECOMMENDED_BUFFERED_ITEMS
                            + ")` so batch materialization overlaps trainer compute.",
                    evidence(metadata, inputProfile, prefix)));
            return;
        }

        int maxBufferedItems = intValue(metadata.get(prefix + ".prefetch.maxBufferedItems"), 0);
        if (maxBufferedItems > 0 && maxBufferedItems < MIN_RECOMMENDED_BUFFERED_ITEMS) {
            findings.add(new TrainingReportRuntimeInputProfileGate.Finding(
                    "runtime-input-train-prefetch-buffer-too-small",
                    "warning",
                    "Training input `next()` dominates while the prefetch buffer can hold only "
                            + maxBufferedItems + " item(s).",
                    "Increase train loader prefetch buffering to at least "
                            + MIN_RECOMMENDED_BUFFERED_ITEMS
                            + " item(s), then compare the next runtime profile before changing model math.",
                    evidence(metadata, inputProfile, prefix)));
        }
    }

    private static Map<String, Object> evidence(
            Map<String, ?> metadata,
            Map<String, ?> inputProfile,
            String prefix) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("recommendedPrefetchBufferSize", MIN_RECOMMENDED_BUFFERED_ITEMS);
        copy(inputProfile, evidence, "dominantScope");
        copy(inputProfile, evidence, "dominantScopePercent");
        copy(inputProfile, evidence, "dominantScopeTotalMillis");
        Map<String, ?> train = mapValue(inputProfile.get("train"));
        copy(train, evidence, "dominantStage");
        copy(train, evidence, "dominantStagePercent");
        copy(train, evidence, "dominantStageTotalMillis");
        copy(metadata, evidence, prefix + ".batchCount");
        copy(metadata, evidence, prefix + ".prefetch.enabled");
        copy(metadata, evidence, prefix + ".prefetch.bufferSize");
        copy(metadata, evidence, prefix + ".prefetch.workerCount");
        copy(metadata, evidence, prefix + ".prefetch.maxBufferedItems");
        copy(metadata, evidence, prefix + ".prefetch.summary");
        return Map.copyOf(evidence);
    }

    private static void copy(Map<String, ?> source, Map<String, Object> target, String key) {
        if (source.containsKey(key)) {
            target.put(key, source.get(key));
        }
    }

    private static Map<String, ?> mapValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> typed = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                typed.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            return Map.copyOf(typed);
        }
        return Map.of();
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }
}
