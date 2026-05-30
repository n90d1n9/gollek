package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Converts trainer diagnostics into a typed, user-facing action plan.
 */
public final class TrainingReportAdvisor {
    private TrainingReportAdvisor() {
    }

    public static TrainingReportActionPlan actionPlan(Map<String, ?> report) {
        List<TrainingReportDiagnostics.Finding> findings = findings(report);
        return actionPlan(findings);
    }

    public static TrainingReportActionPlan actionPlan(List<TrainingReportDiagnostics.Finding> findings) {
        List<TrainingReportDiagnostics.Finding> safeFindings = findings == null ? List.of() : List.copyOf(findings);
        return new TrainingReportActionPlan(
                TrainingReportDiagnostics.summarize(safeFindings),
                recommendations(safeFindings));
    }

    public static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        return recommendations(findings(report));
    }

    public static List<TrainingReportRecommendation> recommendations(
            List<TrainingReportDiagnostics.Finding> findings) {
        if (findings == null || findings.isEmpty()) {
            return List.of();
        }
        return findings.stream()
                .map(TrainingReportRecommendation::fromFinding)
                .toList();
    }

    private static List<TrainingReportDiagnostics.Finding> findings(Map<String, ?> report) {
        Objects.requireNonNull(report, "report must not be null");
        if (report.containsKey("diagnostics")) {
            return TrainingReportReader.diagnosticFindings(report);
        }
        return TrainingReportDiagnostics.analyze(report);
    }
}
