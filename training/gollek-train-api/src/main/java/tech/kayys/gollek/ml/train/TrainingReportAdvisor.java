package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Converts trainer diagnostics into a typed, user-facing action plan.
 */
public final class TrainingReportAdvisor {
    public static final String COMPARISON_ACTION_PLAN_SCHEMA =
            TrainingReportComparisonActionPlanExport.SCHEMA;

    private TrainingReportAdvisor() {
    }

    public static String comparisonActionPlanSchema() {
        return COMPARISON_ACTION_PLAN_SCHEMA;
    }

    public static TrainingReportActionPlan actionPlan(Map<String, ?> report) {
        List<TrainingReportDiagnostics.Finding> findings = findings(report);
        return new TrainingReportActionPlan(
                TrainingReportDiagnostics.summarize(findings),
                recommendations(report, findings));
    }

    public static TrainingReportActionPlan actionPlan(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        List<TrainingReportDiagnostics.Finding> findings = findings(candidateReport);
        return new TrainingReportActionPlan(
                TrainingReportDiagnostics.summarize(findings),
                recommendations(baselineReport, candidateReport, findings));
    }

    public static TrainingReportRuntimeRegressionSummary runtimeRegressionSummary(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return TrainingReportRuntimeRegressionAdvisor.summary(
                TrainingReportReader.runtimeProfileView(baselineReport),
                TrainingReportReader.runtimeProfileView(candidateReport));
    }

    public static Map<String, Object> runtimeRegression(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return runtimeRegressionSummary(baselineReport, candidateReport).toMap();
    }

    public static String runtimeRegressionMarkdown(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return TrainingReportRuntimeRegressionSummaryMarkdown.render(
                runtimeRegressionSummary(baselineReport, candidateReport));
    }

    public static String comparisonActionPlanMarkdown(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return TrainingReportActionPlanMarkdown.render(
                actionPlan(baselineReport, candidateReport),
                runtimeRegressionSummary(baselineReport, candidateReport));
    }

    public static Map<String, Object> comparisonActionPlan(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return comparisonActionPlanExport(baselineReport, candidateReport).toMap();
    }

    public static TrainingReportComparisonActionPlanExport comparisonActionPlanExport(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        TrainingReportActionPlan actionPlan = actionPlan(baselineReport, candidateReport);
        TrainingReportRuntimeRegressionSummary runtimeRegression =
                runtimeRegressionSummary(baselineReport, candidateReport);
        return TrainingReportComparisonActionPlanExport.of(actionPlan, runtimeRegression);
    }

    public static TrainingReportActionPlan actionPlan(List<TrainingReportDiagnostics.Finding> findings) {
        List<TrainingReportDiagnostics.Finding> safeFindings = findings == null ? List.of() : List.copyOf(findings);
        return new TrainingReportActionPlan(
                TrainingReportDiagnostics.summarize(safeFindings),
                recommendations(safeFindings));
    }

    public static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        return recommendations(report, findings(report));
    }

    public static List<TrainingReportRecommendation> recommendations(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        return recommendations(baselineReport, candidateReport, findings(candidateReport));
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

    private static List<TrainingReportRecommendation> recommendations(
            Map<String, ?> report,
            List<TrainingReportDiagnostics.Finding> findings) {
        List<TrainingReportRecommendation> recommendations = new ArrayList<>(recommendations(findings));
        recommendations.addAll(TrainingReportAccelerationAdvisor.recommendations(report));
        recommendations.addAll(TrainingReportThroughputAdvisor.recommendations(report));
        recommendations.addAll(TrainingReportRuntimeProfileAdvisor.recommendations(report));
        return List.copyOf(recommendations);
    }

    private static List<TrainingReportRecommendation> recommendations(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport,
            List<TrainingReportDiagnostics.Finding> findings) {
        List<TrainingReportRecommendation> recommendations = new ArrayList<>(recommendations(candidateReport, findings));
        recommendations.addAll(TrainingReportRuntimeRegressionAdvisor.recommendations(baselineReport, candidateReport));
        return List.copyOf(recommendations);
    }

    private static List<TrainingReportDiagnostics.Finding> findings(Map<String, ?> report) {
        Objects.requireNonNull(report, "report must not be null");
        if (report.containsKey("diagnostics")) {
            return TrainingReportReader.diagnosticFindings(report);
        }
        return TrainingReportDiagnostics.analyze(report);
    }
}
