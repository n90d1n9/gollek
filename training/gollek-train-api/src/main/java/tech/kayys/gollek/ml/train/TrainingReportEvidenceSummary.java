package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

final class TrainingReportEvidenceSummary {
    private static final int DEFAULT_MAX_FIELDS = 6;

    private TrainingReportEvidenceSummary() {
    }

    static String compact(Object evidence) {
        return compact(evidence, DEFAULT_MAX_FIELDS);
    }

    static String compact(Object evidence, int maxFields) {
        if (!(evidence instanceof Map<?, ?> map) || map.isEmpty() || maxFields <= 0) {
            return "";
        }
        List<Map.Entry<?, ?>> entries = new ArrayList<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                entries.add(entry);
            }
        }
        entries.sort(Comparator.comparing(entry -> String.valueOf(entry.getKey())));
        List<String> fields = new ArrayList<>(entries.size());
        for (Map.Entry<?, ?> entry : entries) {
            if (fields.size() == maxFields) {
                break;
            }
            fields.add(entry.getKey() + "=" + entry.getValue());
        }
        return String.join(", ", fields);
    }
}
