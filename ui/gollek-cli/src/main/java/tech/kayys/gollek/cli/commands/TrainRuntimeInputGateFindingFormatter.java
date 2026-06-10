package tech.kayys.gollek.cli.commands;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGate;

/**
 * Renders runtime input gate findings for terminal summaries without coupling CLI commands to report internals.
 */
final class TrainRuntimeInputGateFindingFormatter {
    private static final List<String> PREFETCH_EVIDENCE_KEYS = List.of(
            "recommendedPrefetchBufferSize",
            "trainLoaderPlan.prefetch.enabled",
            "trainLoaderPlan.prefetch.maxBufferedItems",
            "trainLoaderPlan.prefetch.summary");

    private TrainRuntimeInputGateFindingFormatter() {
    }

    static void print(TrainingReportRuntimeInputProfileGate.Finding finding) {
        System.out.println("- " + finding.severity() + " " + finding.code() + ": " + finding.message());
        if (!finding.action().isBlank()) {
            System.out.println("  action: " + finding.action());
        }
        String evidence = prefetchEvidence(finding.evidence());
        if (!evidence.isBlank()) {
            System.out.println("  evidence: " + evidence);
        }
    }

    private static String prefetchEvidence(Map<String, Object> evidence) {
        if (evidence == null || !evidence.containsKey("recommendedPrefetchBufferSize")) {
            return "";
        }
        List<String> values = new ArrayList<>();
        for (String key : PREFETCH_EVIDENCE_KEYS) {
            if (evidence.containsKey(key)) {
                values.add(key + "=" + evidence.get(key));
            }
        }
        return String.join(", ", values);
    }
}
