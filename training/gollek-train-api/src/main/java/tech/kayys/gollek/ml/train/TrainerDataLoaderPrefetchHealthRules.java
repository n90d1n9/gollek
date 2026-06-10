package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainerDataLoaderPrefetchHealthRules {
    private static final String KIND = "data-loader";
    private static final int MIN_PREFETCH_ADVICE_BATCHES = 4;
    private static final int MIN_PREFETCH_BUFFERED_ITEMS = 2;

    private TrainerDataLoaderPrefetchHealthRules() {
    }

    static void addIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues,
            String prefix,
            String phase,
            int batchCount) {
        if (!"train".equals(phase) || batchCount < MIN_PREFETCH_ADVICE_BATCHES) {
            return;
        }

        String enabledKey = prefix + ".prefetch.enabled";
        if (!metadata.containsKey(enabledKey)) {
            return;
        }

        boolean prefetchEnabled = booleanValue(metadata.get(enabledKey));
        if (!prefetchEnabled) {
            issues.add(warning(
                    code(phase, "prefetch-disabled"),
                    phase,
                    phase + " loader prefetching is disabled for a multi-batch epoch",
                    "enable bounded prefetching with DataLoader.prefetch(...) or loader.prefetch(...) "
                            + "when input loading stalls accelerator or CPU training",
                    evidence(prefix, metadata)));
            return;
        }

        int maxBufferedItems = intValue(metadata.get(prefix + ".prefetch.maxBufferedItems"), 0);
        if (maxBufferedItems > 0 && maxBufferedItems < MIN_PREFETCH_BUFFERED_ITEMS) {
            issues.add(warning(
                    code(phase, "prefetch-buffer-too-small"),
                    phase,
                    phase + " loader prefetch buffer can hold only " + maxBufferedItems + " item(s)",
                    "increase the prefetch buffer to at least " + MIN_PREFETCH_BUFFERED_ITEMS
                            + " items for steadier input throughput on multi-batch epochs",
                    evidence(prefix, metadata)));
        }
    }

    private static TrainerHealthIssue warning(
            String code,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return TrainerHealthIssue.warning(KIND, code, artifact, message, action, evidence);
    }

    private static Map<String, Object> evidence(String prefix, Map<String, Object> metadata) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        copy(metadata, evidence, prefix + ".kind");
        copy(metadata, evidence, prefix + ".available");
        copy(metadata, evidence, prefix + ".sampleCount");
        copy(metadata, evidence, prefix + ".batchSize");
        copy(metadata, evidence, prefix + ".batchCount");
        copy(metadata, evidence, prefix + ".prefetch.enabled");
        copy(metadata, evidence, prefix + ".prefetch.bufferSize");
        copy(metadata, evidence, prefix + ".prefetch.workerCount");
        copy(metadata, evidence, prefix + ".prefetch.maxBufferedItems");
        copy(metadata, evidence, prefix + ".prefetch.summary");
        copy(metadata, evidence, prefix + ".prefetch.sourceLoaderType");
        return Map.copyOf(evidence);
    }

    private static void copy(Map<String, Object> source, Map<String, Object> target, String key) {
        if (source.containsKey(key)) {
            target.put(key, source.get(key));
        }
    }

    private static String code(String phase, String suffix) {
        return "data-loader-" + phase + "-" + suffix;
    }
}
