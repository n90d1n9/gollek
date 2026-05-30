package tech.kayys.gollek.train.diffusion.opd;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioComparison;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioComparisonEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioDeltaDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioDriftDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioExecutiveSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioPolicyProfile;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioRemediationBucket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioRemediationSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowActionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowBatchManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowExecutionPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryDashboardEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolio;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioComparison;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioComparisonEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioDeltaDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioDriftDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioExecutiveSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioPolicyProfile;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioRemediationBucket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioRemediationSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowActionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowBatchManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowDispatchEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowExecutionPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryDashboardEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolio;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryPlaybook;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryDelta;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryRecommendation;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryRow;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistorySnapshotComparison;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistorySnapshot;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryStability;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryStatus;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryStatusPolicy;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryTimeline;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryTrend;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSectionView;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignApprovalPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignClosureReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionLog;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionLogEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilitySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionReview;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignEngineHandoffEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleasePacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignIncidentReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignApprovalStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignRolloutPack;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignResolutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignResolutionStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdWorkflowApprovalGate;

/**
 * Public typed query surface for normalized Diffusion OPD reports.
 *
 * <p>The larger campaign and release-delivery orchestration lanes are intentionally split into
 * package-private helpers in this package so this class can stay focused on the API entrypoints
 * and shared summary logic.
 */
public final class DiffusionOpdReportQueries {

    private DiffusionOpdReportQueries() {
    }

    /**
     * Returns the normalized typed round-history rows from one OPD report.
     */
    public static List<DiffusionOpdRoundHistoryRow> roundHistoryRows(DiffusionOpdReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return report.roundHistory().stream()
                .map(DiffusionOpdRoundHistoryRow::fromMap)
                .toList();
    }

    /**
     * Returns the subset of round-history rows belonging to one task id.
     */
    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForTask(
            DiffusionOpdReport report,
            String taskId) {
        return roundHistoryRows(report).stream()
                .filter(row -> taskId.equals(row.taskId()))
                .toList();
    }

    /**
     * Returns the subset of round-history rows belonging to one teacher key.
     */
    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return roundHistoryRows(report).stream()
                .filter(row -> teacherKey.equals(row.teacherKey()))
                .toList();
    }

    /**
     * Returns the subset of round-history rows belonging to one stage name.
     */
    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForStage(
            DiffusionOpdReport report,
            String stageName) {
        return roundHistoryRows(report).stream()
                .filter(row -> stageName.equals(row.stageName()))
                .toList();
    }

    /**
     * Summarizes the round-history subset for one task into the standard aggregate view.
     */
    public static DiffusionOpdRoundHistorySummary summarizeTaskHistory(
            DiffusionOpdReport report,
            String taskId) {
        return summarizeRows(roundHistoryForTask(report, taskId));
    }

    /**
     * Summarizes the round-history subset for one teacher into the standard aggregate view.
     */
    public static DiffusionOpdRoundHistorySummary summarizeTeacherHistory(
            DiffusionOpdReport report,
            String teacherKey) {
        return summarizeRows(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Summarizes the round-history subset for one stage into the standard aggregate view.
     */
    public static DiffusionOpdRoundHistorySummary summarizeStageHistory(
            DiffusionOpdReport report,
            String stageName) {
        return summarizeRows(roundHistoryForStage(report, stageName));
    }

    /**
     * Returns the mean loss from the summarized task history.
     */
    public static Double meanLossForTask(DiffusionOpdReport report, String taskId) {
        return summarizeTaskHistory(report, taskId).meanLoss();
    }

    /**
     * Returns the mean loss from the summarized teacher history.
     */
    public static Double meanLossForTeacher(DiffusionOpdReport report, String teacherKey) {
        return summarizeTeacherHistory(report, teacherKey).meanLoss();
    }

    /**
     * Returns the mean loss from the summarized stage history.
     */
    public static Double meanLossForStage(DiffusionOpdReport report, String stageName) {
        return summarizeStageHistory(report, stageName).meanLoss();
    }

    /**
     * Returns the maximum loss observed in the filtered task history.
     */
    public static Double maxLossForTask(DiffusionOpdReport report, String taskId) {
        return maxLoss(roundHistoryForTask(report, taskId));
    }

    /**
     * Returns the maximum loss observed in the filtered teacher history.
     */
    public static Double maxLossForTeacher(DiffusionOpdReport report, String teacherKey) {
        return maxLoss(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Returns the maximum loss observed in the filtered stage history.
     */
    public static Double maxLossForStage(DiffusionOpdReport report, String stageName) {
        return maxLoss(roundHistoryForStage(report, stageName));
    }

    /**
     * Returns the last round index present in the filtered task history.
     */
    public static Integer lastRoundForTask(DiffusionOpdReport report, String taskId) {
        return lastRound(roundHistoryForTask(report, taskId));
    }

    /**
     * Returns the last round index present in the filtered teacher history.
     */
    public static Integer lastRoundForTeacher(DiffusionOpdReport report, String teacherKey) {
        return lastRound(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Returns the last round index present in the filtered stage history.
     */
    public static Integer lastRoundForStage(DiffusionOpdReport report, String stageName) {
        return lastRound(roundHistoryForStage(report, stageName));
    }

    /**
     * Returns the latest row present in the filtered task history.
     */
    public static DiffusionOpdRoundHistoryRow latestTaskRow(
            DiffusionOpdReport report,
            String taskId) {
        return latestRow(roundHistoryForTask(report, taskId));
    }

    /**
     * Returns the latest row present in the filtered teacher history.
     */
    public static DiffusionOpdRoundHistoryRow latestTeacherRow(
            DiffusionOpdReport report,
            String teacherKey) {
        return latestRow(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Returns the latest row present in the filtered stage history.
     */
    public static DiffusionOpdRoundHistoryRow latestStageRow(
            DiffusionOpdReport report,
            String stageName) {
        return latestRow(roundHistoryForStage(report, stageName));
    }

    /**
     * Computes the first-to-last delta across the filtered task history.
     */
    public static DiffusionOpdRoundHistoryDelta taskDelta(
            DiffusionOpdReport report,
            String taskId) {
        return delta(roundHistoryForTask(report, taskId));
    }

    /**
     * Computes the first-to-last delta across the filtered teacher history.
     */
    public static DiffusionOpdRoundHistoryDelta teacherDelta(
            DiffusionOpdReport report,
            String teacherKey) {
        return delta(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Computes the first-to-last delta across the filtered stage history.
     */
    public static DiffusionOpdRoundHistoryDelta stageDelta(
            DiffusionOpdReport report,
            String stageName) {
        return delta(roundHistoryForStage(report, stageName));
    }

    /**
     * Computes the trend view over the filtered task history using the requested trailing window.
     */
    public static DiffusionOpdRoundHistoryTrend taskTrend(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return trend(roundHistoryForTask(report, taskId), windowSize);
    }

    /**
     * Computes the trend view over the filtered teacher history using the requested trailing window.
     */
    public static DiffusionOpdRoundHistoryTrend teacherTrend(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return trend(roundHistoryForTeacher(report, teacherKey), windowSize);
    }

    /**
     * Computes the trend view over the filtered stage history using the requested trailing window.
     */
    public static DiffusionOpdRoundHistoryTrend stageTrend(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return trend(roundHistoryForStage(report, stageName), windowSize);
    }

    /**
     * Computes the stability view over the filtered task history using the requested trailing
     * window.
     */
    public static DiffusionOpdRoundHistoryStability taskStability(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return stability(roundHistoryForTask(report, taskId), windowSize);
    }

    /**
     * Computes the stability view over the filtered teacher history using the requested trailing
     * window.
     */
    public static DiffusionOpdRoundHistoryStability teacherStability(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return stability(roundHistoryForTeacher(report, teacherKey), windowSize);
    }

    /**
     * Computes the stability view over the filtered stage history using the requested trailing
     * window.
     */
    public static DiffusionOpdRoundHistoryStability stageStability(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stability(roundHistoryForStage(report, stageName), windowSize);
    }

    /**
     * Computes the status view over the filtered task history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskStatus(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return taskStatus(report, taskId, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the status view over the filtered task history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskStatus(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForTask(report, taskId), windowSize, policy);
    }

    /**
     * Computes the status view over the filtered teacher history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus teacherStatus(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return teacherStatus(report, teacherKey, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the status view over the filtered teacher history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryStatus teacherStatus(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForTeacher(report, teacherKey), windowSize, policy);
    }

    /**
     * Computes the status view over the filtered stage history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus stageStatus(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stageStatus(report, stageName, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the status view over the filtered stage history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryStatus stageStatus(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForStage(report, stageName), windowSize, policy);
    }

    /**
     * Computes the recommendation view over the filtered task history using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskRecommendation(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return taskRecommendation(report, taskId, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the recommendation view over the filtered task history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskRecommendation(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(taskStatus(report, taskId, windowSize, policy));
    }

    /**
     * Computes the recommendation view over the filtered teacher history using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation teacherRecommendation(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return teacherRecommendation(report, teacherKey, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the recommendation view over the filtered teacher history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation teacherRecommendation(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(teacherStatus(report, teacherKey, windowSize, policy));
    }

    /**
     * Computes the recommendation view over the filtered stage history using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation stageRecommendation(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stageRecommendation(report, stageName, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the recommendation view over the filtered stage history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation stageRecommendation(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(stageStatus(report, stageName, windowSize, policy));
    }

    /**
     * Computes the playbook view over the filtered task history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskPlaybook(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return taskPlaybook(report, taskId, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the playbook view over the filtered task history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskPlaybook(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(taskRecommendation(report, taskId, windowSize, policy));
    }

    /**
     * Computes the playbook view over the filtered teacher history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook teacherPlaybook(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return teacherPlaybook(report, teacherKey, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the playbook view over the filtered teacher history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook teacherPlaybook(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(teacherRecommendation(report, teacherKey, windowSize, policy));
    }

    /**
     * Computes the playbook view over the filtered stage history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook stagePlaybook(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stagePlaybook(report, stageName, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the playbook view over the filtered stage history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook stagePlaybook(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(stageRecommendation(report, stageName, windowSize, policy));
    }

    /**
     * Computes the snapshot view over the filtered task history using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskSnapshot(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return taskSnapshot(report, taskId, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the snapshot view over the filtered task history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskSnapshot(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForTask(report, taskId), windowSize, policy);
    }

    /**
     * Computes the snapshot view over the filtered teacher history using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot teacherSnapshot(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return teacherSnapshot(report, teacherKey, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the snapshot view over the filtered teacher history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistorySnapshot teacherSnapshot(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForTeacher(report, teacherKey), windowSize, policy);
    }

    /**
     * Computes the snapshot view over the filtered stage history using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot stageSnapshot(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stageSnapshot(report, stageName, windowSize, DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the snapshot view over the filtered stage history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot stageSnapshot(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForStage(report, stageName), windowSize, policy);
    }

    /**
     * Computes the timeline view over the filtered task history using the default status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline taskTimeline(
            DiffusionOpdReport report,
            String taskId,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize) {
        return taskTimeline(
                report,
                taskId,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the timeline view over the filtered task history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline taskTimeline(
            DiffusionOpdReport report,
            String taskId,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForTask(report, taskId), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Computes the timeline view over the filtered teacher history using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryTimeline teacherTimeline(
            DiffusionOpdReport report,
            String teacherKey,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForTeacher(report, teacherKey), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Computes the timeline view over the filtered stage history using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline stageTimeline(
            DiffusionOpdReport report,
            String stageName,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForStage(report, stageName), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Compares the baseline and current task snapshots using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTaskSnapshots(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String taskId,
            int windowSize) {
        return compareTaskSnapshots(
                baselineReport,
                currentReport,
                taskId,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Compares the baseline and current task snapshots using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTaskSnapshots(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                taskSnapshot(baselineReport, taskId, windowSize, policy),
                taskSnapshot(currentReport, taskId, windowSize, policy));
    }

    /**
     * Compares the baseline and current teacher snapshots using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTeacherSnapshots(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                teacherSnapshot(baselineReport, teacherKey, windowSize, policy),
                teacherSnapshot(currentReport, teacherKey, windowSize, policy));
    }

    /**
     * Compares the baseline and current stage snapshots using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareStageSnapshots(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                stageSnapshot(baselineReport, stageName, windowSize, policy),
                stageSnapshot(currentReport, stageName, windowSize, policy));
    }

    /**
     * Compares the baseline and current task portfolios using the default status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioComparison compareTaskPortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize) {
        return compareTaskPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Compares the baseline and current task portfolios using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioComparison compareTaskPortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareGroupedPortfolios(
                taskPortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                taskPortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Compares the baseline and current teacher portfolios using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioComparison compareTeacherPortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareGroupedPortfolios(
                teacherPortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                teacherPortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Compares the baseline and current stage portfolios using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioComparison compareStagePortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareGroupedPortfolios(
                stagePortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                stagePortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Compares the baseline and current task-teacher paired portfolios using the supplied status
     * policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioComparison compareTaskTeacherPortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return comparePairedPortfolios(
                taskTeacherPortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                taskTeacherPortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Compares the baseline and current task-stage paired portfolios using the supplied status
     * policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioComparison compareTaskStagePortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return comparePairedPortfolios(
                taskStagePortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                taskStagePortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Compares the baseline and current teacher-stage paired portfolios using the supplied status
     * policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioComparison compareTeacherStagePortfolios(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return comparePairedPortfolios(
                teacherStagePortfolio(baselineReport, shortWindowSize, mediumWindowSize, longWindowSize, policy),
                teacherStagePortfolio(currentReport, shortWindowSize, mediumWindowSize, longWindowSize, policy));
    }

    /**
     * Builds the task portfolio delta dashboard using the default status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDeltaDashboard taskPortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return taskPortfolioDeltaDashboard(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Builds the task portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDeltaDashboard taskPortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDeltaDashboard(compareTaskPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the teacher portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDeltaDashboard teacherPortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDeltaDashboard(compareTeacherPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the stage portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDeltaDashboard stagePortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDeltaDashboard(compareStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task-teacher paired portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDeltaDashboard taskTeacherPortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDeltaDashboard(compareTaskTeacherPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task-stage paired portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDeltaDashboard taskStagePortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDeltaDashboard(compareTaskStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the teacher-stage paired portfolio delta dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDeltaDashboard teacherStagePortfolioDeltaDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDeltaDashboard(compareTeacherStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task portfolio drift dashboard using the default status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDriftDashboard taskPortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return taskPortfolioDriftDashboard(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Builds the task portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDriftDashboard taskPortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDriftDashboard(compareTaskPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the teacher portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDriftDashboard teacherPortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDriftDashboard(compareTeacherPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the stage portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioDriftDashboard stagePortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioDriftDashboard(compareStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task-teacher paired portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDriftDashboard taskTeacherPortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDriftDashboard(compareTaskTeacherPortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task-stage paired portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDriftDashboard taskStagePortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDriftDashboard(compareTaskStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the teacher-stage paired portfolio drift dashboard using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioDriftDashboard teacherStagePortfolioDriftDashboard(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioDriftDashboard(compareTeacherStagePortfolios(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy), limit);
    }

    /**
     * Builds the task portfolio executive summary using the default status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioExecutiveSummary taskPortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return taskPortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Builds the task portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioExecutiveSummary taskPortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioExecutiveSummary(
                taskPortfolioDeltaDashboard(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit,
                        policy),
                taskPortfolioDriftDashboard(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit,
                        policy),
                limit);
    }

    /**
     * Builds the teacher portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioExecutiveSummary teacherPortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioExecutiveSummary(
                teacherPortfolioDeltaDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                teacherPortfolioDriftDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                limit);
    }

    /**
     * Builds the stage portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioExecutiveSummary stagePortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioExecutiveSummary(
                stagePortfolioDeltaDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                stagePortfolioDriftDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                limit);
    }

    /**
     * Builds the task-teacher paired portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioExecutiveSummary taskTeacherPortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioExecutiveSummary(
                taskTeacherPortfolioDeltaDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                taskTeacherPortfolioDriftDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                limit);
    }

    /**
     * Builds the task-stage paired portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioExecutiveSummary taskStagePortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioExecutiveSummary(
                taskStagePortfolioDeltaDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                taskStagePortfolioDriftDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                limit);
    }

    /**
     * Builds the teacher-stage paired portfolio executive summary using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioExecutiveSummary teacherStagePortfolioExecutiveSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioExecutiveSummary(
                teacherStagePortfolioDeltaDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                teacherStagePortfolioDriftDashboard(baselineReport, currentReport, shortWindowSize, mediumWindowSize, longWindowSize, limit, policy),
                limit);
    }

    /**
     * Builds the task portfolio remediation summary using the default status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioRemediationSummary taskPortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return taskPortfolioRemediationSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Builds the task portfolio remediation summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioRemediationSummary taskPortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioRemediationSummary(taskPortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the teacher portfolio remediation summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioRemediationSummary teacherPortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioRemediationSummary(teacherPortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the stage portfolio remediation summary using the supplied status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolioRemediationSummary stagePortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolioRemediationSummary(stagePortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the task-teacher paired portfolio remediation summary using the supplied status
     * policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioRemediationSummary taskTeacherPortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioRemediationSummary(taskTeacherPortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the task-stage paired portfolio remediation summary using the supplied status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioRemediationSummary taskStagePortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioRemediationSummary(taskStagePortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the teacher-stage paired portfolio remediation summary using the supplied status
     * policy.
     */
    public static DiffusionOpdPairedHistoryPortfolioRemediationSummary teacherStagePortfolioRemediationSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolioRemediationSummary(teacherStagePortfolioExecutiveSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit,
                policy));
    }

    /**
     * Builds the task portfolio policy-comparison summary across the built-in default, strict,
     * and lenient status-policy variants.
     */
    public static DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary taskPortfolioPolicyComparisonSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedPolicyComparisonSummary(List.of(
                groupedPolicyProfile(
                        "default",
                        DiffusionOpdRoundHistoryStatusPolicy.defaults(),
                        taskPortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.defaults())),
                groupedPolicyProfile(
                        "strict",
                        DiffusionOpdRoundHistoryStatusPolicy.strict(),
                        taskPortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.strict())),
                groupedPolicyProfile(
                        "lenient",
                        DiffusionOpdRoundHistoryStatusPolicy.lenient(),
                        taskPortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.lenient()))));
    }

    /**
     * Builds the task-stage paired portfolio policy-comparison summary across the built-in
     * default, strict, and lenient status-policy variants.
     */
    public static DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary taskStagePortfolioPolicyComparisonSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedPolicyComparisonSummary(List.of(
                pairedPolicyProfile(
                        "default",
                        DiffusionOpdRoundHistoryStatusPolicy.defaults(),
                        taskStagePortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.defaults())),
                pairedPolicyProfile(
                        "strict",
                        DiffusionOpdRoundHistoryStatusPolicy.strict(),
                        taskStagePortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.strict())),
                pairedPolicyProfile(
                        "lenient",
                        DiffusionOpdRoundHistoryStatusPolicy.lenient(),
                        taskStagePortfolioExecutiveSummary(
                                baselineReport,
                                currentReport,
                                shortWindowSize,
                                mediumWindowSize,
                                longWindowSize,
                                limit,
                                DiffusionOpdRoundHistoryStatusPolicy.lenient()))));
    }

    /**
     * Builds the grouped task-portfolio policy rollout plan from the built-in policy-comparison
     * summary variants.
     */
    public static DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan taskPortfolioPolicyRolloutPlan(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedPolicyRolloutPlan(taskPortfolioPolicyComparisonSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the paired task-stage portfolio policy rollout plan from the built-in
     * policy-comparison summary variants.
     */
    public static DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan taskStagePortfolioPolicyRolloutPlan(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedPolicyRolloutPlan(taskStagePortfolioPolicyComparisonSummary(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the grouped task-portfolio workflow action packet from the grouped policy rollout
     * plan.
     */
    public static DiffusionOpdGroupedHistoryWorkflowActionPacket taskPortfolioWorkflowActionPacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedWorkflowActionPacket(taskPortfolioPolicyRolloutPlan(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the paired task-stage portfolio workflow action packet from the paired policy rollout
     * plan.
     */
    public static DiffusionOpdPairedHistoryWorkflowActionPacket taskStagePortfolioWorkflowActionPacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedWorkflowActionPacket(taskStagePortfolioPolicyRolloutPlan(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the grouped task-portfolio workflow execution plan from the grouped workflow action
     * packet.
     */
    public static DiffusionOpdGroupedHistoryWorkflowExecutionPlan taskPortfolioWorkflowExecutionPlan(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedWorkflowExecutionPlan(taskPortfolioWorkflowActionPacket(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the paired task-stage portfolio workflow execution plan from the paired workflow
     * action packet.
     */
    public static DiffusionOpdPairedHistoryWorkflowExecutionPlan taskStagePortfolioWorkflowExecutionPlan(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedWorkflowExecutionPlan(taskStagePortfolioWorkflowActionPacket(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the grouped task-portfolio workflow dispatch envelope from the grouped workflow
     * execution plan.
     */
    public static DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope taskPortfolioWorkflowDispatchEnvelope(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedWorkflowDispatchEnvelope(taskPortfolioWorkflowExecutionPlan(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the paired task-stage portfolio workflow dispatch envelope from the paired workflow
     * execution plan.
     */
    public static DiffusionOpdPairedHistoryWorkflowDispatchEnvelope taskStagePortfolioWorkflowDispatchEnvelope(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedWorkflowDispatchEnvelope(taskStagePortfolioWorkflowExecutionPlan(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the grouped task-portfolio workflow batch manifest containing the grouped workflow
     * dispatch envelope.
     */
    public static DiffusionOpdGroupedHistoryWorkflowBatchManifest taskPortfolioWorkflowBatchManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return groupedWorkflowBatchManifest(List.of(taskPortfolioWorkflowDispatchEnvelope(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit)));
    }

    /**
     * Builds the paired task-stage portfolio workflow batch manifest containing the paired
     * workflow dispatch envelope.
     */
    public static DiffusionOpdPairedHistoryWorkflowBatchManifest taskStagePortfolioWorkflowBatchManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return pairedWorkflowBatchManifest(List.of(taskStagePortfolioWorkflowDispatchEnvelope(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit)));
    }

    /**
     * Builds the campaign summary from the approval-stage pipeline rooted in the supplied baseline
     * and current reports.
     */
    public static DiffusionOpdCampaignSummary taskCampaignSummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignApprovalPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).campaignSummary();
    }

    /**
     * Builds the campaign rollout pack from the approval-stage pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignRolloutPack taskCampaignRolloutPack(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignApprovalPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).rolloutPack();
    }

    /**
     * Builds the campaign approval packet from the approval-stage pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignApprovalPacket taskCampaignApprovalPacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignApprovalPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).approvalPacket();
    }

    /**
     * Builds the campaign decision log from the incident-handoff pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignDecisionLog taskCampaignDecisionLog(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).decisionLog();
    }

    /**
     * Builds the campaign incident report from the incident-handoff pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignIncidentReport taskCampaignIncidentReport(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).incidentReport();
    }

    /**
     * Builds the campaign resolution packet from the incident-handoff pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignResolutionPacket taskCampaignResolutionPacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).resolutionPacket();
    }

    /**
     * Builds the campaign closure report from the incident-handoff pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignClosureReport taskCampaignClosureReport(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).closureReport();
    }

    /**
     * Builds the campaign handoff bundle from the incident-handoff pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignHandoffBundle taskCampaignHandoffBundle(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).handoffBundle();
    }

    /**
     * Builds the campaign export manifest from the delivery-receipt pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignExportManifest taskCampaignExportManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDeliveryReceiptPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportManifest();
    }

    /**
     * Builds the campaign delivery report from the delivery-receipt pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignDeliveryReport taskCampaignDeliveryReport(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDeliveryReceiptPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).deliveryReport();
    }

    /**
     * Builds the campaign delivery ledger from the delivery-receipt pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignDeliveryLedger taskCampaignDeliveryLedger(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDeliveryReceiptPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).deliveryLedger();
    }

    /**
     * Builds the campaign delivery receipts from the delivery-receipt pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignDeliveryReceipts taskCampaignDeliveryReceipts(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDeliveryReceiptPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).deliveryReceipts();
    }

    /**
     * Builds the campaign receipt acknowledgements from the acknowledgement-planning pipeline
     * rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReceiptAcknowledgements taskCampaignReceiptAcknowledgements(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignAcknowledgementPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).receiptAcknowledgements();
    }

    /**
     * Builds the campaign acknowledgement decisions from the acknowledgement-planning pipeline
     * rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignAcknowledgementDecisions taskCampaignAcknowledgementDecisions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignAcknowledgementPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).acknowledgementDecisions();
    }

    /**
     * Builds the campaign decision resolutions from the dispatch-planning pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignDecisionResolutions taskCampaignDecisionResolutions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDispatchPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).decisionResolutions();
    }

    /**
     * Builds the campaign dispatch eligibility summary from the dispatch-planning pipeline rooted
     * in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignDispatchEligibilitySummary taskCampaignDispatchEligibilitySummary(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDispatchPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).eligibilitySummary();
    }

    /**
     * Builds the campaign dispatch plan from the dispatch-planning pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignDispatchPlan taskCampaignDispatchPlan(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDispatchPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).dispatchPlan();
    }

    /**
     * Builds the campaign execution packet from the dispatch-execution pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignExecutionPacket taskCampaignExecutionPacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDispatchExecutionPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).executionPacket();
    }

    /**
     * Builds the campaign engine handoff envelope from the dispatch-execution pipeline rooted in
     * the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignEngineHandoffEnvelope taskCampaignEngineHandoffEnvelope(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignDispatchExecutionPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).handoffEnvelope();
    }

    /**
     * Builds the campaign execution manifest from the execution-release pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignExecutionManifest taskCampaignExecutionManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignExecutionReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).executionManifest();
    }

    /**
     * Builds the campaign execution review from the execution-release pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignExecutionReview taskCampaignExecutionReview(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignExecutionReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).executionReview();
    }

    /**
     * Builds the campaign release decision from the execution-release pipeline rooted in the
     * supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDecision taskCampaignReleaseDecision(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignExecutionReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseDecision();
    }

    /**
     * Builds the campaign release packet from the release pipeline rooted in the supplied baseline
     * and current reports.
     */
    public static DiffusionOpdCampaignReleasePacket taskCampaignReleasePacket(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releasePacket();
    }

    /**
     * Builds the campaign release ledger from the release pipeline rooted in the supplied baseline
     * and current reports.
     */
    public static DiffusionOpdCampaignReleaseLedger taskCampaignReleaseLedger(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseLedger();
    }

    /**
     * Builds the campaign release receipt from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseReceipt taskCampaignReleaseReceipt(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseReceipt();
    }

    /**
     * Builds the campaign release acknowledgement from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseAcknowledgement taskCampaignReleaseAcknowledgement(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseAcknowledgement();
    }

    /**
     * Builds the campaign release resolution from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseResolution taskCampaignReleaseResolution(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseResolution();
    }

    /**
     * Builds the campaign release closure from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseClosure taskCampaignReleaseClosure(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).releaseClosure();
    }

    /**
     * Builds the campaign release handoff bundle from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseHandoffBundle taskCampaignReleaseHandoffBundle(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).handoffBundle();
    }

    /**
     * Builds the campaign release export manifest from the release pipeline rooted in the supplied
     * baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseExportManifest taskCampaignReleaseExportManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignPipelines.buildCampaignReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportManifest();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery report from the release-delivery
     * adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryReport taskCampaignReleaseDeliveryReport(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).report();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery ledger from the release-delivery
     * adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryLedger taskCampaignReleaseDeliveryLedger(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).ledger();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery receipts from the release-delivery
     * adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryReceipts taskCampaignReleaseDeliveryReceipts(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).receipts();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery acknowledgements from the
     * release-delivery adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryAcknowledgements taskCampaignReleaseDeliveryAcknowledgements(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).acknowledgements();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery decisions from the release-delivery
     * adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryDecisions taskCampaignReleaseDeliveryDecisions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).decisions();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery resolutions from the
     * release-delivery adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryResolutions taskCampaignReleaseDeliveryResolutions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).resolutions();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery closures from the release-delivery
     * adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryClosures taskCampaignReleaseDeliveryClosures(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).closures();
    }

    /**
     * Builds the compatibility-shaped campaign release delivery handoff bundles from the
     * release-delivery adapter pipeline rooted in the supplied baseline and current reports.
     */
    public static List<DiffusionOpdCampaignReleaseDeliveryHandoffBundle> taskCampaignReleaseDeliveryHandoffBundles(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildAdapterPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).handoffBundles();
    }

    /**
     * Builds the canonical campaign release delivery export manifest from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportManifest taskCampaignReleaseDeliveryExportManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportManifest();
    }

    /**
     * Builds the canonical campaign release delivery export report from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportReport taskCampaignReleaseDeliveryExportReport(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportReport();
    }

    /**
     * Builds the canonical campaign release delivery export ledger from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportLedger taskCampaignReleaseDeliveryExportLedger(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportLedger();
    }

    /**
     * Builds the canonical campaign release delivery export receipts from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportReceipts taskCampaignReleaseDeliveryExportReceipts(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportReceipts();
    }

    /**
     * Builds the canonical campaign release delivery export acknowledgements from the
     * release-delivery export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements taskCampaignReleaseDeliveryExportAcknowledgements(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportAcknowledgements();
    }

    /**
     * Builds the canonical campaign release delivery export decisions from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportDecisions taskCampaignReleaseDeliveryExportDecisions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportDecisions();
    }

    /**
     * Builds the canonical campaign release delivery export resolutions from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportResolutions taskCampaignReleaseDeliveryExportResolutions(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportResolutions();
    }

    /**
     * Builds the canonical campaign release delivery export closures from the release-delivery
     * export pipeline rooted in the supplied baseline and current reports.
     */
    public static DiffusionOpdCampaignReleaseDeliveryExportClosures taskCampaignReleaseDeliveryExportClosures(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).exportClosures();
    }

    /**
     * Builds the canonical campaign release delivery export handoff bundles from the
     * release-delivery export pipeline rooted in the supplied baseline and current reports.
     */
    public static List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> taskCampaignReleaseDeliveryExportHandoffBundles(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignReleaseDeliveryFacade.buildExportPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit).handoffBundles();
    }

    /**
     * Builds the grouped task portfolio using the default round-history status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolio taskPortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize) {
        return taskPortfolio(
                report,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Builds the grouped task portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolio taskPortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolio(
                roundHistoryRows(report),
                "taskId",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Builds the grouped teacher portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolio teacherPortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolio(
                roundHistoryRows(report),
                "teacherKey",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Builds the grouped stage portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdGroupedHistoryPortfolio stagePortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return groupedPortfolio(
                roundHistoryRows(report),
                "stageName",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Builds the paired task-teacher portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolio taskTeacherPortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolio(
                roundHistoryRows(report),
                "taskId",
                "teacherKey",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Builds the paired task-stage portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolio taskStagePortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolio(
                roundHistoryRows(report),
                "taskId",
                "stageName",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Builds the paired teacher-stage portfolio using the supplied round-history status policy.
     */
    public static DiffusionOpdPairedHistoryPortfolio teacherStagePortfolio(
            DiffusionOpdReport report,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return pairedPortfolio(
                roundHistoryRows(report),
                "teacherKey",
                "stageName",
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                policy);
    }

    /**
     * Returns the grouped task summaries over the full round-history row set.
     */
    public static List<DiffusionOpdGroupedHistorySummary> taskSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "taskId");
    }

    /**
     * Returns the grouped teacher summaries over the full round-history row set.
     */
    public static List<DiffusionOpdGroupedHistorySummary> teacherSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "teacherKey");
    }

    /**
     * Returns the grouped stage summaries over the full round-history row set.
     */
    public static List<DiffusionOpdGroupedHistorySummary> stageSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "stageName");
    }

    /**
     * Returns the highest-mean-loss task summaries, limited to the requested count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTaskSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummaries(report), limit);
    }

    /**
     * Returns grouped task summaries for the subset of history rows filtered to one stage.
     */
    public static List<DiffusionOpdGroupedHistorySummary> taskSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return groupedSummaries(roundHistoryForStage(report, stageName), "taskId");
    }

    /**
     * Returns grouped task summaries for the subset of history rows filtered to one teacher.
     */
    public static List<DiffusionOpdGroupedHistorySummary> taskSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return groupedSummaries(roundHistoryForTeacher(report, teacherKey), "taskId");
    }

    /**
     * Returns the highest-mean-loss task summaries within one stage, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTaskSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummariesForStage(report, stageName), limit);
    }

    /**
     * Returns the single highest-mean-loss task summary within one stage.
     */
    public static DiffusionOpdGroupedHistorySummary topTaskSummaryForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName) {
        return firstOrNull(topTaskSummariesForStageByMeanLoss(report, stageName, 1));
    }

    /**
     * Returns the highest-mean-loss task summaries within one teacher, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTaskSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummariesForTeacher(report, teacherKey), limit);
    }

    /**
     * Returns the single highest-mean-loss task summary within one teacher.
     */
    public static DiffusionOpdGroupedHistorySummary topTaskSummaryForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey) {
        return firstOrNull(topTaskSummariesForTeacherByMeanLoss(report, teacherKey, 1));
    }

    /**
     * Returns the highest-mean-loss teacher summaries, limited to the requested count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTeacherSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummaries(report), limit);
    }

    /**
     * Returns grouped teacher summaries for the subset of history rows filtered to one task.
     */
    public static List<DiffusionOpdGroupedHistorySummary> teacherSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return groupedSummaries(roundHistoryForTask(report, taskId), "teacherKey");
    }

    /**
     * Returns grouped teacher summaries for the subset of history rows filtered to one stage.
     */
    public static List<DiffusionOpdGroupedHistorySummary> teacherSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return groupedSummaries(roundHistoryForStage(report, stageName), "teacherKey");
    }

    /**
     * Returns the highest-mean-loss teacher summaries within one task, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTeacherSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummariesForTask(report, taskId), limit);
    }

    /**
     * Returns the single highest-mean-loss teacher summary within one task.
     */
    public static DiffusionOpdGroupedHistorySummary topTeacherSummaryForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId) {
        return firstOrNull(topTeacherSummariesForTaskByMeanLoss(report, taskId, 1));
    }

    /**
     * Returns the highest-mean-loss teacher summaries within one stage, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topTeacherSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummariesForStage(report, stageName), limit);
    }

    /**
     * Returns the single highest-mean-loss teacher summary within one stage.
     */
    public static DiffusionOpdGroupedHistorySummary topTeacherSummaryForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName) {
        return firstOrNull(topTeacherSummariesForStageByMeanLoss(report, stageName, 1));
    }

    /**
     * Returns the highest-mean-loss stage summaries, limited to the requested count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummaries(report), limit);
    }

    /**
     * Returns grouped stage summaries for the subset of history rows filtered to one task.
     */
    public static List<DiffusionOpdGroupedHistorySummary> stageSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return groupedSummaries(roundHistoryForTask(report, taskId), "stageName");
    }

    /**
     * Returns grouped stage summaries for the subset of history rows filtered to one teacher.
     */
    public static List<DiffusionOpdGroupedHistorySummary> stageSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return groupedSummaries(roundHistoryForTeacher(report, teacherKey), "stageName");
    }

    /**
     * Returns the highest-mean-loss stage summaries within one task, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topStageSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummariesForTask(report, taskId), limit);
    }

    /**
     * Returns the single highest-mean-loss stage summary within one task.
     */
    public static DiffusionOpdGroupedHistorySummary topStageSummaryForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId) {
        return firstOrNull(topStageSummariesForTaskByMeanLoss(report, taskId, 1));
    }

    /**
     * Returns the highest-mean-loss stage summaries within one teacher, limited to the requested
     * count.
     */
    public static List<DiffusionOpdGroupedHistorySummary> topStageSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummariesForTeacher(report, teacherKey), limit);
    }

    /**
     * Returns the single highest-mean-loss stage summary within one teacher.
     */
    public static DiffusionOpdGroupedHistorySummary topStageSummaryForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey) {
        return firstOrNull(topStageSummariesForTeacherByMeanLoss(report, teacherKey, 1));
    }

    /**
     * Returns the paired task-teacher summaries over the full round-history row set.
     */
    public static List<DiffusionOpdPairedHistorySummary> taskTeacherSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "taskId", "teacherKey");
    }

    /**
     * Returns the paired task-stage summaries over the full round-history row set.
     */
    public static List<DiffusionOpdPairedHistorySummary> taskStageSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "taskId", "stageName");
    }

    /**
     * Returns the paired teacher-stage summaries over the full round-history row set.
     */
    public static List<DiffusionOpdPairedHistorySummary> teacherStageSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "teacherKey", "stageName");
    }

    /**
     * Returns the highest-mean-loss task-teacher paired summaries, limited to the requested
     * count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTaskTeacherSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(taskTeacherSummaries(report), limit);
    }

    /**
     * Returns paired task-teacher summaries for the subset of history rows filtered to one stage.
     */
    public static List<DiffusionOpdPairedHistorySummary> taskTeacherSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return pairedSummaries(roundHistoryForStage(report, stageName), "taskId", "teacherKey");
    }

    /**
     * Returns the highest-mean-loss task-teacher paired summaries within one stage, limited to
     * the requested count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTaskTeacherSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topPairedSummariesByMeanLoss(taskTeacherSummariesForStage(report, stageName), limit);
    }

    /**
     * Returns the single highest-mean-loss task-teacher paired summary within one stage.
     */
    public static DiffusionOpdPairedHistorySummary topTaskTeacherSummaryForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName) {
        return firstOrNull(topTaskTeacherSummariesForStageByMeanLoss(report, stageName, 1));
    }

    /**
     * Returns the highest-mean-loss task-stage paired summaries, limited to the requested count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTaskStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(taskStageSummaries(report), limit);
    }

    /**
     * Returns paired task-stage summaries for the subset of history rows filtered to one teacher.
     */
    public static List<DiffusionOpdPairedHistorySummary> taskStageSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return pairedSummaries(roundHistoryForTeacher(report, teacherKey), "taskId", "stageName");
    }

    /**
     * Returns the highest-mean-loss task-stage paired summaries within one teacher, limited to
     * the requested count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTaskStageSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topPairedSummariesByMeanLoss(taskStageSummariesForTeacher(report, teacherKey), limit);
    }

    /**
     * Returns the single highest-mean-loss task-stage paired summary within one teacher.
     */
    public static DiffusionOpdPairedHistorySummary topTaskStageSummaryForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey) {
        return firstOrNull(topTaskStageSummariesForTeacherByMeanLoss(report, teacherKey, 1));
    }

    /**
     * Returns the highest-mean-loss teacher-stage paired summaries, limited to the requested
     * count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTeacherStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(teacherStageSummaries(report), limit);
    }

    /**
     * Returns paired teacher-stage summaries for the subset of history rows filtered to one task.
     */
    public static List<DiffusionOpdPairedHistorySummary> teacherStageSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return pairedSummaries(roundHistoryForTask(report, taskId), "teacherKey", "stageName");
    }

    /**
     * Returns the highest-mean-loss teacher-stage paired summaries within one task, limited to
     * the requested count.
     */
    public static List<DiffusionOpdPairedHistorySummary> topTeacherStageSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topPairedSummariesByMeanLoss(teacherStageSummariesForTask(report, taskId), limit);
    }

    /**
     * Returns the single highest-mean-loss teacher-stage paired summary within one task.
     */
    public static DiffusionOpdPairedHistorySummary topTeacherStageSummaryForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId) {
        return firstOrNull(topTeacherStageSummariesForTaskByMeanLoss(report, taskId, 1));
    }

    /**
     * Returns the latest row from history filtered to one stage for the paired task-teacher view.
     */
    public static DiffusionOpdRoundHistoryRow latestTaskTeacherRowForStage(
            DiffusionOpdReport report,
            String stageName) {
        return latestRow(roundHistoryForStage(report, stageName));
    }

    /**
     * Returns the latest row from history filtered to one teacher for the paired task-stage view.
     */
    public static DiffusionOpdRoundHistoryRow latestTaskStageRowForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return latestRow(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Returns the latest row from history filtered to one task for the paired teacher-stage view.
     */
    public static DiffusionOpdRoundHistoryRow latestTeacherStageRowForTask(
            DiffusionOpdReport report,
            String taskId) {
        return latestRow(roundHistoryForTask(report, taskId));
    }

    /**
     * Returns the first-to-last delta from history filtered to one stage for the paired
     * task-teacher view.
     */
    public static DiffusionOpdRoundHistoryDelta taskTeacherDeltaForStage(
            DiffusionOpdReport report,
            String stageName) {
        return delta(roundHistoryForStage(report, stageName));
    }

    /**
     * Returns the first-to-last delta from history filtered to one teacher for the paired
     * task-stage view.
     */
    public static DiffusionOpdRoundHistoryDelta taskStageDeltaForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return delta(roundHistoryForTeacher(report, teacherKey));
    }

    /**
     * Returns the first-to-last delta from history filtered to one task for the paired
     * teacher-stage view.
     */
    public static DiffusionOpdRoundHistoryDelta teacherStageDeltaForTask(
            DiffusionOpdReport report,
            String taskId) {
        return delta(roundHistoryForTask(report, taskId));
    }

    /**
     * Computes the trailing-window trend from history filtered to one stage for the paired
     * task-teacher view.
     */
    public static DiffusionOpdRoundHistoryTrend taskTeacherTrendForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return trend(roundHistoryForStage(report, stageName), windowSize);
    }

    /**
     * Computes the trailing-window trend from history filtered to one teacher for the paired
     * task-stage view.
     */
    public static DiffusionOpdRoundHistoryTrend taskStageTrendForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return trend(roundHistoryForTeacher(report, teacherKey), windowSize);
    }

    /**
     * Computes the trailing-window trend from history filtered to one task for the paired
     * teacher-stage view.
     */
    public static DiffusionOpdRoundHistoryTrend teacherStageTrendForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return trend(roundHistoryForTask(report, taskId), windowSize);
    }

    /**
     * Computes the trailing-window stability view from history filtered to one stage for the
     * paired task-teacher view.
     */
    public static DiffusionOpdRoundHistoryStability taskTeacherStabilityForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return stability(roundHistoryForStage(report, stageName), windowSize);
    }

    /**
     * Computes the trailing-window stability view from history filtered to one teacher for the
     * paired task-stage view.
     */
    public static DiffusionOpdRoundHistoryStability taskStageStabilityForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return stability(roundHistoryForTeacher(report, teacherKey), windowSize);
    }

    /**
     * Computes the trailing-window stability view from history filtered to one task for the
     * paired teacher-stage view.
     */
    public static DiffusionOpdRoundHistoryStability teacherStageStabilityForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return stability(roundHistoryForTask(report, taskId), windowSize);
    }

    /**
     * Computes the paired task-teacher status view for one stage using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskTeacherStatusForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return taskTeacherStatusForStage(
                report,
                stageName,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-teacher status view for one stage using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskTeacherStatusForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForStage(report, stageName), windowSize, policy);
    }

    /**
     * Computes the paired task-stage status view for one teacher using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskStageStatusForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return taskStageStatusForTeacher(
                report,
                teacherKey,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-stage status view for one teacher using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryStatus taskStageStatusForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForTeacher(report, teacherKey), windowSize, policy);
    }

    /**
     * Computes the paired teacher-stage status view for one task using the default status policy.
     */
    public static DiffusionOpdRoundHistoryStatus teacherStageStatusForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return teacherStageStatusForTask(
                report,
                taskId,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired teacher-stage status view for one task using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryStatus teacherStageStatusForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return status(roundHistoryForTask(report, taskId), windowSize, policy);
    }

    /**
     * Computes the paired task-teacher recommendation for one stage using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskTeacherRecommendationForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return taskTeacherRecommendationForStage(
                report,
                stageName,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-teacher recommendation for one stage using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskTeacherRecommendationForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(taskTeacherStatusForStage(report, stageName, windowSize, policy));
    }

    /**
     * Computes the paired task-stage recommendation for one teacher using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskStageRecommendationForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return taskStageRecommendationForTeacher(
                report,
                teacherKey,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-stage recommendation for one teacher using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation taskStageRecommendationForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(taskStageStatusForTeacher(report, teacherKey, windowSize, policy));
    }

    /**
     * Computes the paired teacher-stage recommendation for one task using the default status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation teacherStageRecommendationForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return teacherStageRecommendationForTask(
                report,
                taskId,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired teacher-stage recommendation for one task using the supplied status
     * policy.
     */
    public static DiffusionOpdRoundHistoryRecommendation teacherStageRecommendationForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return recommend(teacherStageStatusForTask(report, taskId, windowSize, policy));
    }

    /**
     * Computes the paired task-teacher playbook for one stage using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskTeacherPlaybookForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return taskTeacherPlaybookForStage(
                report,
                stageName,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-teacher playbook for one stage using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskTeacherPlaybookForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(taskTeacherRecommendationForStage(report, stageName, windowSize, policy));
    }

    /**
     * Computes the paired task-stage playbook for one teacher using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskStagePlaybookForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return taskStagePlaybookForTeacher(
                report,
                teacherKey,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-stage playbook for one teacher using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook taskStagePlaybookForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(taskStageRecommendationForTeacher(report, teacherKey, windowSize, policy));
    }

    /**
     * Computes the paired teacher-stage playbook for one task using the default status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook teacherStagePlaybookForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return teacherStagePlaybookForTask(
                report,
                taskId,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired teacher-stage playbook for one task using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryPlaybook teacherStagePlaybookForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return playbook(teacherStageRecommendationForTask(report, taskId, windowSize, policy));
    }

    /**
     * Computes the paired task-teacher snapshot for one stage using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskTeacherSnapshotForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize) {
        return taskTeacherSnapshotForStage(
                report,
                stageName,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-teacher snapshot for one stage using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskTeacherSnapshotForStage(
            DiffusionOpdReport report,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForStage(report, stageName), windowSize, policy);
    }

    /**
     * Computes the paired task-stage snapshot for one teacher using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskStageSnapshotForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize) {
        return taskStageSnapshotForTeacher(
                report,
                teacherKey,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired task-stage snapshot for one teacher using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot taskStageSnapshotForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForTeacher(report, teacherKey), windowSize, policy);
    }

    /**
     * Computes the paired teacher-stage snapshot for one task using the default status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot teacherStageSnapshotForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize) {
        return teacherStageSnapshotForTask(
                report,
                taskId,
                windowSize,
                DiffusionOpdRoundHistoryStatusPolicy.defaults());
    }

    /**
     * Computes the paired teacher-stage snapshot for one task using the supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshot teacherStageSnapshotForTask(
            DiffusionOpdReport report,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return snapshot(roundHistoryForTask(report, taskId), windowSize, policy);
    }

    /**
     * Builds the paired task-teacher timeline for one stage using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline taskTeacherTimelineForStage(
            DiffusionOpdReport report,
            String stageName,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForStage(report, stageName), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Builds the paired task-stage timeline for one teacher using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline taskStageTimelineForTeacher(
            DiffusionOpdReport report,
            String teacherKey,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForTeacher(report, teacherKey), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Builds the paired teacher-stage timeline for one task using the supplied status policy.
     */
    public static DiffusionOpdRoundHistoryTimeline teacherStageTimelineForTask(
            DiffusionOpdReport report,
            String taskId,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return timeline(roundHistoryForTask(report, taskId), shortWindowSize, mediumWindowSize, longWindowSize, policy);
    }

    /**
     * Compares the baseline and current paired task-teacher snapshots for one stage using the
     * supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTaskTeacherSnapshotsForStage(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String stageName,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                taskTeacherSnapshotForStage(baselineReport, stageName, windowSize, policy),
                taskTeacherSnapshotForStage(currentReport, stageName, windowSize, policy));
    }

    /**
     * Compares the baseline and current paired task-stage snapshots for one teacher using the
     * supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTaskStageSnapshotsForTeacher(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String teacherKey,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                taskStageSnapshotForTeacher(baselineReport, teacherKey, windowSize, policy),
                taskStageSnapshotForTeacher(currentReport, teacherKey, windowSize, policy));
    }

    /**
     * Compares the baseline and current paired teacher-stage snapshots for one task using the
     * supplied status policy.
     */
    public static DiffusionOpdRoundHistorySnapshotComparison compareTeacherStageSnapshotsForTask(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            String taskId,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        return compareSnapshots(
                teacherStageSnapshotForTask(baselineReport, taskId, windowSize, policy),
                teacherStageSnapshotForTask(currentReport, taskId, windowSize, policy));
    }

    /**
     * Returns the normalized teachers section as a named section view.
     */
    public static DiffusionOpdSectionView teachers(DiffusionOpdReport report) {
        return section("teachers", report.teachers());
    }

    /**
     * Returns the normalized stages section as a named section view.
     */
    public static DiffusionOpdSectionView stages(DiffusionOpdReport report) {
        return section("stages", report.stages());
    }

    /**
     * Returns the normalized tasks section as a named section view.
     */
    public static DiffusionOpdSectionView tasks(DiffusionOpdReport report) {
        return section("tasks", report.tasks());
    }

    /**
     * Returns the normalized conditioning section as a named section view.
     */
    public static DiffusionOpdSectionView conditioning(DiffusionOpdReport report) {
        return section("conditioning", report.conditioning());
    }

    /**
     * Returns the normalized adaptive section as a named section view.
     */
    public static DiffusionOpdSectionView adaptive(DiffusionOpdReport report) {
        return section("adaptive", report.adaptive());
    }

    /**
     * Returns the normalized bindings section as a named section view.
     */
    public static DiffusionOpdSectionView bindings(DiffusionOpdReport report) {
        return section("bindings", report.bindings());
    }

    /**
     * Wraps a normalized report section in a named section-view record for the public surface.
     */
    private static DiffusionOpdSectionView section(String name, Map<String, Object> values) {
        return new DiffusionOpdSectionView(name, values);
    }

    /**
     * Returns the maximum non-null average loss in the supplied row set.
     */
    private static Double maxLoss(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .max(Double::compareTo)
                .orElse(null);
    }

    /**
     * Returns the highest round index present in the supplied row set.
     */
    private static Integer lastRound(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .map(DiffusionOpdRoundHistoryRow::round)
                .max(Integer::compareTo)
                .orElse(null);
    }

    /**
     * Returns the latest row in the supplied row set by round number.
     */
    private static DiffusionOpdRoundHistoryRow latestRow(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .max(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .orElse(null);
    }

    /**
     * Builds the latest-versus-previous row delta view for the supplied row set.
     */
    private static DiffusionOpdRoundHistoryDelta delta(List<DiffusionOpdRoundHistoryRow> rows) {
        List<DiffusionOpdRoundHistoryRow> ordered = rows.stream()
                .sorted(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round).reversed())
                .toList();
        DiffusionOpdRoundHistoryRow latest = ordered.isEmpty() ? null : ordered.getFirst();
        DiffusionOpdRoundHistoryRow previous = ordered.size() < 2 ? null : ordered.get(1);
        Double lossDelta = latest == null || previous == null || latest.averageLoss() == null || previous.averageLoss() == null
                ? null
                : latest.averageLoss() - previous.averageLoss();
        Integer roundDelta = latest == null || previous == null
                ? null
                : latest.round() - previous.round();
        return new DiffusionOpdRoundHistoryDelta(latest, previous, lossDelta, roundDelta);
    }

    /**
     * Builds the trailing-window trend view for the supplied row set after normalizing the
     * requested window size.
     */
    private static DiffusionOpdRoundHistoryTrend trend(
            List<DiffusionOpdRoundHistoryRow> rows,
            int windowSize) {
        int normalizedWindowSize = normalizedWindowSize(windowSize);
        List<DiffusionOpdRoundHistoryRow> ordered = rows.stream()
                .sorted(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .toList();
        int fromIndex = Math.max(0, ordered.size() - normalizedWindowSize);
        List<DiffusionOpdRoundHistoryRow> window = ordered.subList(fromIndex, ordered.size());
        DiffusionOpdRoundHistoryRow earliest = window.isEmpty() ? null : window.getFirst();
        DiffusionOpdRoundHistoryRow latest = window.isEmpty() ? null : window.getLast();
        Double meanLoss = meanLoss(window);
        Double minLoss = minLoss(window);
        Double maxLoss = maxLoss(window);
        Double lossDelta = latest == null || earliest == null || latest.averageLoss() == null || earliest.averageLoss() == null
                ? null
                : latest.averageLoss() - earliest.averageLoss();
        Integer roundDelta = latest == null || earliest == null
                ? null
                : latest.round() - earliest.round();
        Boolean improving = lossDelta == null ? null : lossDelta < 0.0d;
        Boolean worsening = lossDelta == null ? null : lossDelta > 0.0d;
        return new DiffusionOpdRoundHistoryTrend(
                normalizedWindowSize,
                window.size(),
                window,
                earliest,
                latest,
                meanLoss,
                minLoss,
                maxLoss,
                lossDelta,
                roundDelta,
                improving,
                worsening);
    }

    /**
     * Builds the trailing-window stability view on top of the computed trend for the supplied row
     * set.
     */
    private static DiffusionOpdRoundHistoryStability stability(
            List<DiffusionOpdRoundHistoryRow> rows,
            int windowSize) {
        DiffusionOpdRoundHistoryTrend trend = trend(rows, windowSize);
        Double lossRange = trend.maxLoss() == null || trend.minLoss() == null
                ? null
                : trend.maxLoss() - trend.minLoss();
        List<Double> stepDeltas = stepDeltas(trend.rows());
        Double averageSignedStepDelta = average(stepDeltas);
        Double averageAbsoluteStepDelta = averageAbsolute(stepDeltas);
        Double latestAbsoluteStepDelta = stepDeltas.isEmpty() ? null : Math.abs(stepDeltas.getLast());
        Boolean volatileWindow = averageAbsoluteStepDelta == null || latestAbsoluteStepDelta == null
                ? Boolean.FALSE
                : latestAbsoluteStepDelta > averageAbsoluteStepDelta
                        && lossRange != null
                        && lossRange > averageAbsoluteStepDelta;
        Boolean stabilizing = trend.improving() == null || averageAbsoluteStepDelta == null || latestAbsoluteStepDelta == null
                ? null
                : trend.improving() && latestAbsoluteStepDelta <= averageAbsoluteStepDelta;
        return new DiffusionOpdRoundHistoryStability(
                trend,
                lossRange,
                averageSignedStepDelta,
                averageAbsoluteStepDelta,
                latestAbsoluteStepDelta,
                volatileWindow,
                stabilizing);
    }

    /**
     * Classifies the supplied row set into a status view using the requested window size and
     * policy thresholds.
     */
    private static DiffusionOpdRoundHistoryStatus status(
            List<DiffusionOpdRoundHistoryRow> rows,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        DiffusionOpdRoundHistoryStability stability = stability(rows, windowSize);
        int rowCount = stability.trend().rowCount();
        boolean unstable = isUnstable(stability, policy);
        boolean regressing = Boolean.TRUE.equals(stability.trend().worsening());
        boolean stabilizing = isHealthy(stability, policy);

        if (rowCount < policy.minRows()) {
            return new DiffusionOpdRoundHistoryStatus(
                    stability,
                    "insufficient_data",
                    "info",
                    "trend.insufficient_data",
                    "Not enough recent rounds to evaluate training status.",
                    Boolean.FALSE,
                    Boolean.FALSE,
                    Boolean.FALSE);
        }
        if (regressing && unstable) {
            return new DiffusionOpdRoundHistoryStatus(
                    stability,
                    "regressing",
                    "critical",
                    "trend.regressing.unstable",
                    "Recent loss is worsening and the slice looks unstable.",
                    Boolean.FALSE,
                    Boolean.TRUE,
                    Boolean.TRUE);
        }
        if (regressing) {
            return new DiffusionOpdRoundHistoryStatus(
                    stability,
                    "regressing",
                    "warning",
                    "trend.regressing",
                    "Recent loss is worsening over the selected window.",
                    Boolean.FALSE,
                    Boolean.TRUE,
                    Boolean.FALSE);
        }
        if (unstable) {
            return new DiffusionOpdRoundHistoryStatus(
                    stability,
                    "unstable",
                    "warning",
                    "trend.unstable",
                    "Recent loss is moving in an unstable pattern.",
                    Boolean.FALSE,
                    Boolean.FALSE,
                    Boolean.TRUE);
        }
        if (stabilizing) {
            return new DiffusionOpdRoundHistoryStatus(
                    stability,
                    "healthy",
                    "info",
                    "trend.healthy." + policy.name(),
                    "Recent loss is improving and stabilizing under the " + policy.name() + " policy.",
                    Boolean.TRUE,
                    Boolean.FALSE,
                    Boolean.FALSE);
        }
        return new DiffusionOpdRoundHistoryStatus(
                stability,
                "monitoring",
                "info",
                "trend.monitoring",
                "Recent loss is moving without a strong stability signal yet.",
                Boolean.TRUE,
                Boolean.FALSE,
                Boolean.FALSE);
    }

    private static boolean isUnstable(
            DiffusionOpdRoundHistoryStability stability,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        Double latestAbsoluteStepDelta = stability.latestAbsoluteStepDelta();
        Double averageAbsoluteStepDelta = stability.averageAbsoluteStepDelta();
        Double lossRange = stability.lossRange();
        if (latestAbsoluteStepDelta == null || averageAbsoluteStepDelta == null || lossRange == null) {
            return false;
        }
        return latestAbsoluteStepDelta > averageAbsoluteStepDelta * policy.unstableLatestStepMultiplier()
                && lossRange > averageAbsoluteStepDelta * policy.unstableLossRangeMultiplier();
    }

    private static boolean isHealthy(
            DiffusionOpdRoundHistoryStability stability,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        Double latestAbsoluteStepDelta = stability.latestAbsoluteStepDelta();
        Double averageAbsoluteStepDelta = stability.averageAbsoluteStepDelta();
        if (latestAbsoluteStepDelta == null || averageAbsoluteStepDelta == null) {
            return false;
        }
        if (policy.requireImprovingForHealthy() && !Boolean.TRUE.equals(stability.trend().improving())) {
            return false;
        }
        return latestAbsoluteStepDelta <= averageAbsoluteStepDelta * policy.healthyLatestStepMultiplier();
    }

    /**
     * Maps a status view into the next recommended training action and escalation posture.
     */
    private static DiffusionOpdRoundHistoryRecommendation recommend(
            DiffusionOpdRoundHistoryStatus status) {
        return switch (status.status()) {
            case "insufficient_data" -> new DiffusionOpdRoundHistoryRecommendation(
                    status,
                    "monitor",
                    "low",
                    "action.monitor.insufficient_data",
                    "Collect more rounds before making a stronger training decision.",
                    Boolean.FALSE);
            case "regressing" -> new DiffusionOpdRoundHistoryRecommendation(
                    status,
                    Boolean.TRUE.equals(status.unstable()) ? "early_stop_review" : "reduce_lr",
                    Boolean.TRUE.equals(status.unstable()) ? "high" : "medium",
                    Boolean.TRUE.equals(status.unstable())
                            ? "action.early_stop_review.regressing_unstable"
                            : "action.reduce_lr.regressing",
                    Boolean.TRUE.equals(status.unstable())
                            ? "The slice is worsening and unstable, so it should be reviewed before continuing."
                            : "The slice is worsening, so reducing learning rate is a safer next move.",
                    Boolean.TRUE.equals(status.unstable()) ? Boolean.FALSE : Boolean.TRUE);
            case "unstable" -> new DiffusionOpdRoundHistoryRecommendation(
                    status,
                    "monitor",
                    "medium",
                    "action.monitor.unstable",
                    "The slice is unstable, so continued monitoring is safer than aggressive intervention.",
                    Boolean.TRUE);
            case "healthy" -> new DiffusionOpdRoundHistoryRecommendation(
                    status,
                    "continue",
                    "low",
                    "action.continue.healthy",
                    "The slice is improving and stable enough to continue.",
                    Boolean.TRUE);
            default -> new DiffusionOpdRoundHistoryRecommendation(
                    status,
                    "monitor",
                    "low",
                    "action.monitor.default",
                    "The slice should keep running while staying under observation.",
                    Boolean.TRUE);
        };
    }

    /**
     * Maps a recommendation into an operator-oriented playbook with urgency and concrete next
     * steps.
     */
    private static DiffusionOpdRoundHistoryPlaybook playbook(
            DiffusionOpdRoundHistoryRecommendation recommendation) {
        return switch (recommendation.action()) {
            case "continue" -> new DiffusionOpdRoundHistoryPlaybook(
                    recommendation,
                    Boolean.FALSE,
                    0,
                    Math.max(2, recommendation.status().stability().trend().requestedWindowSize()),
                    "none",
                    List.of(
                            "Keep current training settings.",
                            "Re-check after the next normal monitoring window."));
            case "monitor" -> new DiffusionOpdRoundHistoryPlaybook(
                    recommendation,
                    Boolean.FALSE,
                    1,
                    Math.max(2, recommendation.status().stability().trend().requestedWindowSize()),
                    "training-monitor",
                    List.of(
                            "Watch the next short loss window closely.",
                            "Compare the next status result before changing optimizer settings."));
            case "reduce_lr" -> new DiffusionOpdRoundHistoryPlaybook(
                    recommendation,
                    Boolean.FALSE,
                    1,
                    Math.max(2, recommendation.status().stability().trend().requestedWindowSize()),
                    "optimizer-control",
                    List.of(
                            "Reduce learning rate conservatively.",
                            "Watch whether instability drops in the next evaluation window.",
                            "Escalate if regression continues after the cooldown window."));
            case "early_stop_review" -> new DiffusionOpdRoundHistoryPlaybook(
                    recommendation,
                    Boolean.TRUE,
                    2,
                    Math.max(2, recommendation.status().stability().trend().requestedWindowSize()),
                    "human-review",
                    List.of(
                            "Pause aggressive automatic changes.",
                            "Review recent loss and stability metrics.",
                            "Decide whether to stop, retune, or resume with safer settings."));
            default -> new DiffusionOpdRoundHistoryPlaybook(
                    recommendation,
                    Boolean.FALSE,
                    1,
                    Math.max(2, recommendation.status().stability().trend().requestedWindowSize()),
                    "training-monitor",
                    new ArrayList<>());
        };
    }

    /**
     * Builds the composite snapshot view for one row slice by combining trend, stability, status,
     * recommendation, and playbook.
     */
    private static DiffusionOpdRoundHistorySnapshot snapshot(
            List<DiffusionOpdRoundHistoryRow> rows,
            int windowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        DiffusionOpdRoundHistoryTrend trend = trend(rows, windowSize);
        DiffusionOpdRoundHistoryStability stability = stability(rows, windowSize);
        DiffusionOpdRoundHistoryStatus status = status(rows, windowSize, policy);
        DiffusionOpdRoundHistoryRecommendation recommendation = recommend(status);
        DiffusionOpdRoundHistoryPlaybook playbook = playbook(recommendation);
        return new DiffusionOpdRoundHistorySnapshot(
                trend,
                stability,
                status,
                recommendation,
                playbook);
    }

    /**
     * Builds the short-, medium-, and long-window timeline view for one row slice under a single
     * policy.
     */
    private static DiffusionOpdRoundHistoryTimeline timeline(
            List<DiffusionOpdRoundHistoryRow> rows,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        DiffusionOpdRoundHistorySnapshot shortSnapshot = snapshot(rows, shortWindowSize, policy);
        DiffusionOpdRoundHistorySnapshot mediumSnapshot = snapshot(rows, mediumWindowSize, policy);
        DiffusionOpdRoundHistorySnapshot longSnapshot = snapshot(rows, longWindowSize, policy);
        return new DiffusionOpdRoundHistoryTimeline(
                shortWindowSize,
                shortSnapshot,
                mediumWindowSize,
                mediumSnapshot,
                longWindowSize,
                longSnapshot,
                improvingAcrossWindows(shortSnapshot, mediumSnapshot, longSnapshot),
                stableAcrossWindows(shortSnapshot, mediumSnapshot, longSnapshot),
                consistentRecommendation(shortSnapshot, mediumSnapshot, longSnapshot),
                requiresEscalation(shortSnapshot, mediumSnapshot, longSnapshot));
    }

    /**
     * Compares baseline and current snapshots to produce loss deltas plus status and
     * recommendation change flags.
     */
    private static DiffusionOpdRoundHistorySnapshotComparison compareSnapshots(
            DiffusionOpdRoundHistorySnapshot baseline,
            DiffusionOpdRoundHistorySnapshot current) {
        Double meanLossDelta = subtract(
                current.trend().meanLoss(),
                baseline.trend().meanLoss());
        Double latestLossDelta = subtract(
                current.trend().latest() == null ? null : current.trend().latest().averageLoss(),
                baseline.trend().latest() == null ? null : baseline.trend().latest().averageLoss());
        Boolean improved = meanLossDelta != null
                ? meanLossDelta < 0.0d
                : latestLossDelta != null ? latestLossDelta < 0.0d : null;
        return new DiffusionOpdRoundHistorySnapshotComparison(
                baseline,
                current,
                meanLossDelta,
                latestLossDelta,
                improved,
                !Objects.equals(baseline.status().status(), current.status().status()),
                !Objects.equals(baseline.recommendation().action(), current.recommendation().action()));
    }

    /**
     * Returns the current-minus-baseline delta when both values are present.
     */
    private static Double subtract(Double current, Double baseline) {
        return current == null || baseline == null ? null : current - baseline;
    }

    /**
     * Aligns grouped portfolio entries by value and builds the baseline-vs-current grouped
     * comparison view.
     */
    private static DiffusionOpdGroupedHistoryPortfolioComparison compareGroupedPortfolios(
            DiffusionOpdGroupedHistoryPortfolio baseline,
            DiffusionOpdGroupedHistoryPortfolio current) {
        Map<String, DiffusionOpdGroupedHistoryDashboardEntry> baselineEntries = new LinkedHashMap<>();
        for (DiffusionOpdGroupedHistoryDashboardEntry entry : baseline.entries()) {
            baselineEntries.put(entry.value(), entry);
        }

        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> entries = current.entries().stream()
                .filter(entry -> baselineEntries.containsKey(entry.value()))
                .map(entry -> toGroupedPortfolioComparison(baseline.field(), baselineEntries.get(entry.value()), entry))
                .sorted(Comparator.comparing(
                        DiffusionOpdGroupedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.naturalOrder())))
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolioComparison(current.field(), entries);
    }

    /**
     * Aligns paired portfolio entries by pair key and builds the baseline-vs-current paired
     * comparison view.
     */
    private static DiffusionOpdPairedHistoryPortfolioComparison comparePairedPortfolios(
            DiffusionOpdPairedHistoryPortfolio baseline,
            DiffusionOpdPairedHistoryPortfolio current) {
        Map<String, DiffusionOpdPairedHistoryDashboardEntry> baselineEntries = new LinkedHashMap<>();
        for (DiffusionOpdPairedHistoryDashboardEntry entry : baseline.entries()) {
            baselineEntries.put(entry.pair(), entry);
        }

        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> entries = current.entries().stream()
                .filter(entry -> baselineEntries.containsKey(entry.pair()))
                .map(entry -> toPairedPortfolioComparison(
                        baseline.firstField(),
                        baseline.secondField(),
                        baselineEntries.get(entry.pair()),
                        entry))
                .sorted(Comparator.comparing(
                        DiffusionOpdPairedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.naturalOrder())))
                .toList();
        return new DiffusionOpdPairedHistoryPortfolioComparison(
                current.firstField(),
                current.secondField(),
                entries);
    }

    /**
     * Builds one grouped portfolio comparison entry from aligned baseline and current dashboard
     * entries.
     */
    private static DiffusionOpdGroupedHistoryPortfolioComparisonEntry toGroupedPortfolioComparison(
            String field,
            DiffusionOpdGroupedHistoryDashboardEntry baselineEntry,
            DiffusionOpdGroupedHistoryDashboardEntry currentEntry) {
        Double meanLossDelta = subtract(
                currentEntry.summary().meanLoss(),
                baselineEntry.summary().meanLoss());
        Boolean improved = meanLossDelta == null ? null : meanLossDelta < 0.0d;
        return new DiffusionOpdGroupedHistoryPortfolioComparisonEntry(
                field,
                currentEntry.value(),
                baselineEntry,
                currentEntry,
                meanLossDelta,
                improved,
                !Objects.equals(
                        baselineEntry.snapshot().status().status(),
                        currentEntry.snapshot().status().status()),
                !Objects.equals(
                        baselineEntry.snapshot().recommendation().action(),
                        currentEntry.snapshot().recommendation().action()));
    }

    /**
     * Builds the grouped delta dashboard by selecting top improvements, regressions, and
     * recommendation changes from a grouped comparison.
     */
    private static DiffusionOpdGroupedHistoryPortfolioDeltaDashboard groupedPortfolioDeltaDashboard(
            DiffusionOpdGroupedHistoryPortfolioComparison comparison,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> improvements = comparison.entries().stream()
                .filter(entry -> Boolean.TRUE.equals(entry.improved()))
                .sorted(Comparator.comparing(
                        DiffusionOpdGroupedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.naturalOrder())))
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> regressions = comparison.entries().stream()
                .filter(entry -> Boolean.FALSE.equals(entry.improved()))
                .filter(entry -> entry.meanLossDelta() != null && entry.meanLossDelta() > 0.0d)
                .sorted(Comparator.comparing(
                        DiffusionOpdGroupedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> recommendationChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::recommendationChanged)
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolioDeltaDashboard(
                comparison,
                comparison.entries().size(),
                (int) comparison.entries().stream().filter(entry -> Boolean.TRUE.equals(entry.improved())).count(),
                (int) comparison.entries().stream()
                        .filter(entry -> entry.meanLossDelta() != null && entry.meanLossDelta() > 0.0d)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::recommendationChanged)
                        .count(),
                improvements,
                regressions,
                recommendationChangedEntries);
    }

    /**
     * Builds the grouped drift dashboard by selecting status, recommendation, and escalation
     * changes from a grouped comparison.
     */
    private static DiffusionOpdGroupedHistoryPortfolioDriftDashboard groupedPortfolioDriftDashboard(
            DiffusionOpdGroupedHistoryPortfolioComparison comparison,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> statusChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::statusChanged)
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> recommendationChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::recommendationChanged)
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> escalatedEntries = comparison.entries().stream()
                .filter(DiffusionOpdReportQueries::isEscalated)
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolioDriftDashboard(
                comparison,
                comparison.entries().size(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::statusChanged)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::recommendationChanged)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isEscalated)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isHealthyToUnhealthy)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isUnhealthyToHealthy)
                        .count(),
                statusChangedEntries,
                recommendationChangedEntries,
                escalatedEntries);
    }

    /**
     * Builds one paired portfolio comparison entry from aligned baseline and current dashboard
     * entries.
     */
    private static DiffusionOpdPairedHistoryPortfolioComparisonEntry toPairedPortfolioComparison(
            String firstField,
            String secondField,
            DiffusionOpdPairedHistoryDashboardEntry baselineEntry,
            DiffusionOpdPairedHistoryDashboardEntry currentEntry) {
        Double meanLossDelta = subtract(
                currentEntry.summary().meanLoss(),
                baselineEntry.summary().meanLoss());
        Boolean improved = meanLossDelta == null ? null : meanLossDelta < 0.0d;
        return new DiffusionOpdPairedHistoryPortfolioComparisonEntry(
                firstField,
                currentEntry.firstValue(),
                secondField,
                currentEntry.secondValue(),
                currentEntry.pair(),
                baselineEntry,
                currentEntry,
                meanLossDelta,
                improved,
                !Objects.equals(
                        baselineEntry.snapshot().status().status(),
                        currentEntry.snapshot().status().status()),
                !Objects.equals(
                        baselineEntry.snapshot().recommendation().action(),
                        currentEntry.snapshot().recommendation().action()));
    }

    /**
     * Builds the paired delta dashboard by selecting top improvements, regressions, and
     * recommendation changes from a paired comparison.
     */
    private static DiffusionOpdPairedHistoryPortfolioDeltaDashboard pairedPortfolioDeltaDashboard(
            DiffusionOpdPairedHistoryPortfolioComparison comparison,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> improvements = comparison.entries().stream()
                .filter(entry -> Boolean.TRUE.equals(entry.improved()))
                .sorted(Comparator.comparing(
                        DiffusionOpdPairedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.naturalOrder())))
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> regressions = comparison.entries().stream()
                .filter(entry -> Boolean.FALSE.equals(entry.improved()))
                .filter(entry -> entry.meanLossDelta() != null && entry.meanLossDelta() > 0.0d)
                .sorted(Comparator.comparing(
                        DiffusionOpdPairedHistoryPortfolioComparisonEntry::meanLossDelta,
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> recommendationChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::recommendationChanged)
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdPairedHistoryPortfolioDeltaDashboard(
                comparison,
                comparison.entries().size(),
                (int) comparison.entries().stream().filter(entry -> Boolean.TRUE.equals(entry.improved())).count(),
                (int) comparison.entries().stream()
                        .filter(entry -> entry.meanLossDelta() != null && entry.meanLossDelta() > 0.0d)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::recommendationChanged)
                        .count(),
                improvements,
                regressions,
                recommendationChangedEntries);
    }

    /**
     * Builds the paired drift dashboard by selecting status, recommendation, and escalation
     * changes from a paired comparison.
     */
    private static DiffusionOpdPairedHistoryPortfolioDriftDashboard pairedPortfolioDriftDashboard(
            DiffusionOpdPairedHistoryPortfolioComparison comparison,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> statusChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::statusChanged)
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> recommendationChangedEntries = comparison.entries().stream()
                .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::recommendationChanged)
                .limit(normalizedLimit)
                .toList();
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> escalatedEntries = comparison.entries().stream()
                .filter(DiffusionOpdReportQueries::isEscalated)
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdPairedHistoryPortfolioDriftDashboard(
                comparison,
                comparison.entries().size(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::statusChanged)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdPairedHistoryPortfolioComparisonEntry::recommendationChanged)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isEscalated)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isHealthyToUnhealthy)
                        .count(),
                (int) comparison.entries().stream()
                        .filter(DiffusionOpdReportQueries::isUnhealthyToHealthy)
                        .count(),
                statusChangedEntries,
                recommendationChangedEntries,
                escalatedEntries);
    }

    /**
     * Builds the grouped executive summary by combining the grouped delta and drift dashboards
     * with review-hotspot extraction.
     */
    private static DiffusionOpdGroupedHistoryPortfolioExecutiveSummary groupedPortfolioExecutiveSummary(
            DiffusionOpdGroupedHistoryPortfolioDeltaDashboard deltaDashboard,
            DiffusionOpdGroupedHistoryPortfolioDriftDashboard driftDashboard,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> reviewHotspots = deltaDashboard.comparison().entries().stream()
                .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().playbook().reviewRequired())
                        || !"continue".equals(entry.current().snapshot().recommendation().action()))
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolioExecutiveSummary(
                deltaDashboard,
                driftDashboard,
                firstOrNull(deltaDashboard.biggestImprovements()),
                firstOrNull(deltaDashboard.biggestRegressions()),
                firstOrNull(driftDashboard.escalatedEntries()),
                reviewHotspots.size(),
                reviewHotspots,
                buildGroupedExecutiveSummaryMessage(deltaDashboard, driftDashboard, reviewHotspots.size()));
    }

    /**
     * Builds the paired executive summary by combining the paired delta and drift dashboards with
     * review-hotspot extraction.
     */
    private static DiffusionOpdPairedHistoryPortfolioExecutiveSummary pairedPortfolioExecutiveSummary(
            DiffusionOpdPairedHistoryPortfolioDeltaDashboard deltaDashboard,
            DiffusionOpdPairedHistoryPortfolioDriftDashboard driftDashboard,
            int limit) {
        int normalizedLimit = Math.max(0, limit);
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> reviewHotspots = deltaDashboard.comparison().entries().stream()
                .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().playbook().reviewRequired())
                        || !"continue".equals(entry.current().snapshot().recommendation().action()))
                .limit(normalizedLimit)
                .toList();
        return new DiffusionOpdPairedHistoryPortfolioExecutiveSummary(
                deltaDashboard,
                driftDashboard,
                firstOrNull(deltaDashboard.biggestImprovements()),
                firstOrNull(deltaDashboard.biggestRegressions()),
                firstOrNull(driftDashboard.escalatedEntries()),
                reviewHotspots.size(),
                reviewHotspots,
                buildPairedExecutiveSummaryMessage(deltaDashboard, driftDashboard, reviewHotspots.size()));
    }

    /**
     * Builds the grouped executive-summary message from grouped dashboard counts and hotspot
     * totals.
     */
    private static String buildGroupedExecutiveSummaryMessage(
            DiffusionOpdGroupedHistoryPortfolioDeltaDashboard deltaDashboard,
            DiffusionOpdGroupedHistoryPortfolioDriftDashboard driftDashboard,
            int reviewHotspotCount) {
        return "Grouped portfolio changed across "
                + deltaDashboard.totalEntries()
                + " slices with "
                + deltaDashboard.improvedCount()
                + " improvements, "
                + deltaDashboard.regressedCount()
                + " regressions, "
                + driftDashboard.statusChangedCount()
                + " status shifts, and "
                + reviewHotspotCount
                + " review hotspots.";
    }

    /**
     * Builds the paired executive-summary message from paired dashboard counts and hotspot totals.
     */
    private static String buildPairedExecutiveSummaryMessage(
            DiffusionOpdPairedHistoryPortfolioDeltaDashboard deltaDashboard,
            DiffusionOpdPairedHistoryPortfolioDriftDashboard driftDashboard,
            int reviewHotspotCount) {
        return "Paired portfolio changed across "
                + deltaDashboard.totalEntries()
                + " slices with "
                + deltaDashboard.improvedCount()
                + " improvements, "
                + deltaDashboard.regressedCount()
                + " regressions, "
                + driftDashboard.statusChangedCount()
                + " status shifts, and "
                + reviewHotspotCount
                + " review hotspots.";
    }

    /**
     * Builds the grouped remediation summary by bucketing grouped comparison entries by their
     * recommended action.
     */
    private static DiffusionOpdGroupedHistoryPortfolioRemediationSummary groupedPortfolioRemediationSummary(
            DiffusionOpdGroupedHistoryPortfolioExecutiveSummary executiveSummary) {
        Map<String, List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry>> grouped = new LinkedHashMap<>();
        for (DiffusionOpdGroupedHistoryPortfolioComparisonEntry entry :
                executiveSummary.deltaDashboard().comparison().entries()) {
            grouped.computeIfAbsent(
                            entry.current().snapshot().recommendation().action(),
                            ignored -> new ArrayList<>())
                    .add(entry);
        }
        List<DiffusionOpdGroupedHistoryPortfolioRemediationBucket> actionBuckets = grouped.entrySet().stream()
                .map(entry -> new DiffusionOpdGroupedHistoryPortfolioRemediationBucket(
                        entry.getKey(),
                        entry.getValue().size(),
                        entry.getValue().stream()
                                .anyMatch(item -> Boolean.TRUE.equals(item.current().snapshot().playbook().reviewRequired())),
                        entry.getValue().stream()
                                .allMatch(item -> Boolean.TRUE.equals(item.current().snapshot().recommendation().automationFriendly())),
                        entry.getValue()))
                .sorted(Comparator.comparing(
                                DiffusionOpdGroupedHistoryPortfolioRemediationBucket::count,
                                Comparator.reverseOrder())
                        .thenComparing(bucket -> Boolean.TRUE.equals(bucket.reviewRequired()) ? 0 : 1)
                        .thenComparing(DiffusionOpdGroupedHistoryPortfolioRemediationBucket::action))
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolioRemediationSummary(
                executiveSummary,
                executiveSummary.deltaDashboard().totalEntries(),
                (int) executiveSummary.deltaDashboard().comparison().entries().stream()
                        .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().playbook().reviewRequired()))
                        .count(),
                (int) executiveSummary.deltaDashboard().comparison().entries().stream()
                        .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().recommendation().automationFriendly()))
                        .count(),
                firstOrNull(actionBuckets) == null ? null : firstOrNull(actionBuckets).action(),
                actionBuckets,
                buildGroupedRemediationSummaryMessage(executiveSummary, actionBuckets));
    }

    /**
     * Builds the paired remediation summary by bucketing paired comparison entries by their
     * recommended action.
     */
    private static DiffusionOpdPairedHistoryPortfolioRemediationSummary pairedPortfolioRemediationSummary(
            DiffusionOpdPairedHistoryPortfolioExecutiveSummary executiveSummary) {
        Map<String, List<DiffusionOpdPairedHistoryPortfolioComparisonEntry>> grouped = new LinkedHashMap<>();
        for (DiffusionOpdPairedHistoryPortfolioComparisonEntry entry :
                executiveSummary.deltaDashboard().comparison().entries()) {
            grouped.computeIfAbsent(
                            entry.current().snapshot().recommendation().action(),
                            ignored -> new ArrayList<>())
                    .add(entry);
        }
        List<DiffusionOpdPairedHistoryPortfolioRemediationBucket> actionBuckets = grouped.entrySet().stream()
                .map(entry -> new DiffusionOpdPairedHistoryPortfolioRemediationBucket(
                        entry.getKey(),
                        entry.getValue().size(),
                        entry.getValue().stream()
                                .anyMatch(item -> Boolean.TRUE.equals(item.current().snapshot().playbook().reviewRequired())),
                        entry.getValue().stream()
                                .allMatch(item -> Boolean.TRUE.equals(item.current().snapshot().recommendation().automationFriendly())),
                        entry.getValue()))
                .sorted(Comparator.comparing(
                                DiffusionOpdPairedHistoryPortfolioRemediationBucket::count,
                                Comparator.reverseOrder())
                        .thenComparing(bucket -> Boolean.TRUE.equals(bucket.reviewRequired()) ? 0 : 1)
                        .thenComparing(DiffusionOpdPairedHistoryPortfolioRemediationBucket::action))
                .toList();
        return new DiffusionOpdPairedHistoryPortfolioRemediationSummary(
                executiveSummary,
                executiveSummary.deltaDashboard().totalEntries(),
                (int) executiveSummary.deltaDashboard().comparison().entries().stream()
                        .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().playbook().reviewRequired()))
                        .count(),
                (int) executiveSummary.deltaDashboard().comparison().entries().stream()
                        .filter(entry -> Boolean.TRUE.equals(entry.current().snapshot().recommendation().automationFriendly()))
                        .count(),
                firstOrNull(actionBuckets) == null ? null : firstOrNull(actionBuckets).action(),
                actionBuckets,
                buildPairedRemediationSummaryMessage(executiveSummary, actionBuckets));
    }

    /**
     * Builds the grouped remediation-summary message from grouped hotspots and action buckets.
     */
    private static String buildGroupedRemediationSummaryMessage(
            DiffusionOpdGroupedHistoryPortfolioExecutiveSummary executiveSummary,
            List<DiffusionOpdGroupedHistoryPortfolioRemediationBucket> actionBuckets) {
        String primaryAction = firstOrNull(actionBuckets) == null ? "none" : firstOrNull(actionBuckets).action();
        return "Grouped remediation plan highlights "
                + executiveSummary.reviewHotspotCount()
                + " review hotspots with primary action "
                + primaryAction
                + ".";
    }

    /**
     * Builds the paired remediation-summary message from paired hotspots and action buckets.
     */
    private static String buildPairedRemediationSummaryMessage(
            DiffusionOpdPairedHistoryPortfolioExecutiveSummary executiveSummary,
            List<DiffusionOpdPairedHistoryPortfolioRemediationBucket> actionBuckets) {
        String primaryAction = firstOrNull(actionBuckets) == null ? "none" : firstOrNull(actionBuckets).action();
        return "Paired remediation plan highlights "
                + executiveSummary.reviewHotspotCount()
                + " review hotspots with primary action "
                + primaryAction
                + ".";
    }

    /**
     * Wraps one grouped executive summary as a named grouped policy profile plus its derived
     * remediation summary.
     */
    private static DiffusionOpdGroupedHistoryPortfolioPolicyProfile groupedPolicyProfile(
            String policyName,
            DiffusionOpdRoundHistoryStatusPolicy policy,
            DiffusionOpdGroupedHistoryPortfolioExecutiveSummary executiveSummary) {
        return new DiffusionOpdGroupedHistoryPortfolioPolicyProfile(
                policyName,
                policy,
                executiveSummary,
                groupedPortfolioRemediationSummary(executiveSummary));
    }

    /**
     * Wraps one paired executive summary as a named paired policy profile plus its derived
     * remediation summary.
     */
    private static DiffusionOpdPairedHistoryPortfolioPolicyProfile pairedPolicyProfile(
            String policyName,
            DiffusionOpdRoundHistoryStatusPolicy policy,
            DiffusionOpdPairedHistoryPortfolioExecutiveSummary executiveSummary) {
        return new DiffusionOpdPairedHistoryPortfolioPolicyProfile(
                policyName,
                policy,
                executiveSummary,
                pairedPortfolioRemediationSummary(executiveSummary));
    }

    /**
     * Builds the grouped policy-comparison summary by comparing grouped policy profiles across the
     * built-in policy variants.
     */
    private static DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary groupedPolicyComparisonSummary(
            List<DiffusionOpdGroupedHistoryPortfolioPolicyProfile> profiles) {
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile strictProfile = profiles.stream()
                .filter(profile -> "strict".equals(profile.policyName()))
                .findFirst()
                .orElse(firstOrNull(profiles));
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile lenientProfile = profiles.stream()
                .filter(profile -> "lenient".equals(profile.policyName()))
                .findFirst()
                .orElse(firstOrNull(profiles));
        int maxReviewHotspotCount = profiles.stream()
                .mapToInt(profile -> profile.executiveSummary().reviewHotspotCount())
                .max()
                .orElse(0);
        int maxManualReviewCount = profiles.stream()
                .mapToInt(profile -> profile.remediationSummary().manualReviewCount())
                .max()
                .orElse(0);
        String strictestPrimaryAction = strictProfile == null ? null : strictProfile.remediationSummary().primaryAction();
        String loosestPrimaryAction = lenientProfile == null ? null : lenientProfile.remediationSummary().primaryAction();
        Boolean changed = !Objects.equals(strictestPrimaryAction, loosestPrimaryAction);
        return new DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary(
                profiles,
                strictestPrimaryAction,
                loosestPrimaryAction,
                changed,
                maxReviewHotspotCount,
                maxManualReviewCount,
                "Grouped policy comparison spans "
                        + profiles.size()
                        + " profiles with "
                        + maxReviewHotspotCount
                        + " maximum review hotspots.");
    }

    /**
     * Builds the paired policy-comparison summary by comparing paired policy profiles across the
     * built-in policy variants.
     */
    private static DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary pairedPolicyComparisonSummary(
            List<DiffusionOpdPairedHistoryPortfolioPolicyProfile> profiles) {
        DiffusionOpdPairedHistoryPortfolioPolicyProfile strictProfile = profiles.stream()
                .filter(profile -> "strict".equals(profile.policyName()))
                .findFirst()
                .orElse(firstOrNull(profiles));
        DiffusionOpdPairedHistoryPortfolioPolicyProfile lenientProfile = profiles.stream()
                .filter(profile -> "lenient".equals(profile.policyName()))
                .findFirst()
                .orElse(firstOrNull(profiles));
        int maxReviewHotspotCount = profiles.stream()
                .mapToInt(profile -> profile.executiveSummary().reviewHotspotCount())
                .max()
                .orElse(0);
        int maxManualReviewCount = profiles.stream()
                .mapToInt(profile -> profile.remediationSummary().manualReviewCount())
                .max()
                .orElse(0);
        String strictestPrimaryAction = strictProfile == null ? null : strictProfile.remediationSummary().primaryAction();
        String loosestPrimaryAction = lenientProfile == null ? null : lenientProfile.remediationSummary().primaryAction();
        Boolean changed = !Objects.equals(strictestPrimaryAction, loosestPrimaryAction);
        return new DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary(
                profiles,
                strictestPrimaryAction,
                loosestPrimaryAction,
                changed,
                maxReviewHotspotCount,
                maxManualReviewCount,
                "Paired policy comparison spans "
                        + profiles.size()
                        + " profiles with "
                        + maxReviewHotspotCount
                        + " maximum review hotspots.");
    }

    /**
     * Selects grouped recommended and fallback policy profiles and wraps them as a grouped policy
     * rollout plan.
     */
    private static DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan groupedPolicyRolloutPlan(
            DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary comparisonSummary) {
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile recommendedProfile = comparisonSummary.profiles().stream()
                .sorted(Comparator.comparing(
                                (DiffusionOpdGroupedHistoryPortfolioPolicyProfile profile) ->
                                        profile.remediationSummary().manualReviewCount())
                        .thenComparing(profile -> actionSeverityRank(profile.remediationSummary().primaryAction()))
                        .thenComparing(profile -> profile.executiveSummary().reviewHotspotCount())
                        .thenComparing(profile -> policyPreferenceRank(profile.policyName())))
                .findFirst()
                .orElse(null);
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile fallbackProfile = comparisonSummary.profiles().stream()
                .filter(profile -> recommendedProfile == null || !Objects.equals(profile.policyName(), recommendedProfile.policyName()))
                .sorted(Comparator.comparing(
                                (DiffusionOpdGroupedHistoryPortfolioPolicyProfile profile) ->
                                        profile.remediationSummary().manualReviewCount())
                        .thenComparing(profile -> actionSeverityRank(profile.remediationSummary().primaryAction()))
                        .thenComparing(profile -> profile.executiveSummary().reviewHotspotCount())
                        .thenComparing(profile -> policyPreferenceRank(profile.policyName())))
                .findFirst()
                .orElse(null);
        return new DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan(
                comparisonSummary,
                recommendedProfile,
                fallbackProfile,
                buildGroupedRolloutRationale(recommendedProfile, fallbackProfile),
                comparisonSummary.primaryActionChangedAcrossPolicies());
    }

    /**
     * Selects paired recommended and fallback policy profiles and wraps them as a paired policy
     * rollout plan.
     */
    private static DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan pairedPolicyRolloutPlan(
            DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary comparisonSummary) {
        DiffusionOpdPairedHistoryPortfolioPolicyProfile recommendedProfile = comparisonSummary.profiles().stream()
                .sorted(Comparator.comparing(
                                (DiffusionOpdPairedHistoryPortfolioPolicyProfile profile) ->
                                        profile.remediationSummary().manualReviewCount())
                        .thenComparing(profile -> actionSeverityRank(profile.remediationSummary().primaryAction()))
                        .thenComparing(profile -> profile.executiveSummary().reviewHotspotCount())
                        .thenComparing(profile -> policyPreferenceRank(profile.policyName())))
                .findFirst()
                .orElse(null);
        DiffusionOpdPairedHistoryPortfolioPolicyProfile fallbackProfile = comparisonSummary.profiles().stream()
                .filter(profile -> recommendedProfile == null || !Objects.equals(profile.policyName(), recommendedProfile.policyName()))
                .sorted(Comparator.comparing(
                                (DiffusionOpdPairedHistoryPortfolioPolicyProfile profile) ->
                                        profile.remediationSummary().manualReviewCount())
                        .thenComparing(profile -> actionSeverityRank(profile.remediationSummary().primaryAction()))
                        .thenComparing(profile -> profile.executiveSummary().reviewHotspotCount())
                        .thenComparing(profile -> policyPreferenceRank(profile.policyName())))
                .findFirst()
                .orElse(null);
        return new DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan(
                comparisonSummary,
                recommendedProfile,
                fallbackProfile,
                buildPairedRolloutRationale(recommendedProfile, fallbackProfile),
                comparisonSummary.primaryActionChangedAcrossPolicies());
    }

    /**
     * Explains why the grouped rollout plan chose its recommended profile over the fallback.
     */
    private static String buildGroupedRolloutRationale(
            DiffusionOpdGroupedHistoryPortfolioPolicyProfile recommendedProfile,
            DiffusionOpdGroupedHistoryPortfolioPolicyProfile fallbackProfile) {
        if (recommendedProfile == null) {
            return "No grouped policy profile recommendation is available.";
        }
        if (fallbackProfile == null) {
            return "Recommended grouped policy "
                    + recommendedProfile.policyName()
                    + " because it is the only available profile.";
        }
        return "Recommended grouped policy "
                + recommendedProfile.policyName()
                + " over "
                + fallbackProfile.policyName()
                + " because it minimizes manual review load and action severity.";
    }

    /**
     * Explains why the paired rollout plan chose its recommended profile over the fallback.
     */
    private static String buildPairedRolloutRationale(
            DiffusionOpdPairedHistoryPortfolioPolicyProfile recommendedProfile,
            DiffusionOpdPairedHistoryPortfolioPolicyProfile fallbackProfile) {
        if (recommendedProfile == null) {
            return "No paired policy profile recommendation is available.";
        }
        if (fallbackProfile == null) {
            return "Recommended paired policy "
                    + recommendedProfile.policyName()
                    + " because it is the only available profile.";
        }
        return "Recommended paired policy "
                + recommendedProfile.policyName()
                + " over "
                + fallbackProfile.policyName()
                + " because it minimizes manual review load and action severity.";
    }

    /**
     * Builds the grouped workflow action packet from the selected grouped rollout profile and its
     * primary remediation bucket.
     */
    private static DiffusionOpdGroupedHistoryWorkflowActionPacket groupedWorkflowActionPacket(
            DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan rolloutPlan) {
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile selectedProfile = rolloutPlan.recommendedProfile();
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile fallbackProfile = rolloutPlan.fallbackProfile();
        String primaryAction = selectedProfile == null ? null : selectedProfile.remediationSummary().primaryAction();
        DiffusionOpdGroupedHistoryPortfolioRemediationBucket primaryBucket =
                groupedPrimaryBucket(selectedProfile, primaryAction);
        return new DiffusionOpdGroupedHistoryWorkflowActionPacket(
                rolloutPlan,
                selectedProfile == null ? null : selectedProfile.policyName(),
                fallbackProfile == null ? null : fallbackProfile.policyName(),
                primaryAction,
                fallbackProfile == null ? null : fallbackProfile.remediationSummary().primaryAction(),
                primaryBucket == null ? null : primaryBucket.automationFriendly(),
                primaryBucket == null ? null : primaryBucket.reviewRequired(),
                selectedProfile == null ? 0 : selectedProfile.executiveSummary().reviewHotspotCount(),
                selectedProfile == null ? 0 : selectedProfile.remediationSummary().manualReviewCount(),
                buildGroupedWorkflowPacketMessage(rolloutPlan, primaryAction));
    }

    /**
     * Builds the paired workflow action packet from the selected paired rollout profile and its
     * primary remediation bucket.
     */
    private static DiffusionOpdPairedHistoryWorkflowActionPacket pairedWorkflowActionPacket(
            DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan rolloutPlan) {
        DiffusionOpdPairedHistoryPortfolioPolicyProfile selectedProfile = rolloutPlan.recommendedProfile();
        DiffusionOpdPairedHistoryPortfolioPolicyProfile fallbackProfile = rolloutPlan.fallbackProfile();
        String primaryAction = selectedProfile == null ? null : selectedProfile.remediationSummary().primaryAction();
        DiffusionOpdPairedHistoryPortfolioRemediationBucket primaryBucket =
                pairedPrimaryBucket(selectedProfile, primaryAction);
        return new DiffusionOpdPairedHistoryWorkflowActionPacket(
                rolloutPlan,
                selectedProfile == null ? null : selectedProfile.policyName(),
                fallbackProfile == null ? null : fallbackProfile.policyName(),
                primaryAction,
                fallbackProfile == null ? null : fallbackProfile.remediationSummary().primaryAction(),
                primaryBucket == null ? null : primaryBucket.automationFriendly(),
                primaryBucket == null ? null : primaryBucket.reviewRequired(),
                selectedProfile == null ? 0 : selectedProfile.executiveSummary().reviewHotspotCount(),
                selectedProfile == null ? 0 : selectedProfile.remediationSummary().manualReviewCount(),
                buildPairedWorkflowPacketMessage(rolloutPlan, primaryAction));
    }

    /**
     * Builds the grouped workflow execution plan from the grouped action packet and the playbooks
     * in its selected remediation bucket.
     */
    private static DiffusionOpdGroupedHistoryWorkflowExecutionPlan groupedWorkflowExecutionPlan(
            DiffusionOpdGroupedHistoryWorkflowActionPacket actionPacket) {
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile selectedProfile =
                actionPacket.rolloutPlan().recommendedProfile();
        DiffusionOpdGroupedHistoryPortfolioRemediationBucket primaryBucket =
                groupedPrimaryBucket(selectedProfile, actionPacket.primaryAction());
        List<DiffusionOpdRoundHistoryPlaybook> playbooks = groupedBucketPlaybooks(primaryBucket);
        String escalationTarget = firstEscalationTarget(playbooks);
        DiffusionOpdWorkflowApprovalGate approvalGate = workflowApprovalGate(
                actionPacket.primaryAction(),
                actionPacket.automationFriendly(),
                actionPacket.reviewRequired(),
                actionPacket.manualReviewCount(),
                actionPacket.reviewHotspotCount(),
                escalationTarget);
        String executionMode = workflowExecutionMode(
                actionPacket.primaryAction(),
                actionPacket.automationFriendly(),
                actionPacket.reviewRequired());
        return new DiffusionOpdGroupedHistoryWorkflowExecutionPlan(
                actionPacket,
                approvalGate,
                executionMode,
                maxCooldownRounds(playbooks),
                minNextCheckWindowSize(playbooks),
                escalationTarget,
                mergedChecklist(playbooks),
                buildGroupedWorkflowExecutionMessage(actionPacket, executionMode, approvalGate));
    }

    /**
     * Builds the paired workflow execution plan from the paired action packet and the playbooks in
     * its selected remediation bucket.
     */
    private static DiffusionOpdPairedHistoryWorkflowExecutionPlan pairedWorkflowExecutionPlan(
            DiffusionOpdPairedHistoryWorkflowActionPacket actionPacket) {
        DiffusionOpdPairedHistoryPortfolioPolicyProfile selectedProfile =
                actionPacket.rolloutPlan().recommendedProfile();
        DiffusionOpdPairedHistoryPortfolioRemediationBucket primaryBucket =
                pairedPrimaryBucket(selectedProfile, actionPacket.primaryAction());
        List<DiffusionOpdRoundHistoryPlaybook> playbooks = pairedBucketPlaybooks(primaryBucket);
        String escalationTarget = firstEscalationTarget(playbooks);
        DiffusionOpdWorkflowApprovalGate approvalGate = workflowApprovalGate(
                actionPacket.primaryAction(),
                actionPacket.automationFriendly(),
                actionPacket.reviewRequired(),
                actionPacket.manualReviewCount(),
                actionPacket.reviewHotspotCount(),
                escalationTarget);
        String executionMode = workflowExecutionMode(
                actionPacket.primaryAction(),
                actionPacket.automationFriendly(),
                actionPacket.reviewRequired());
        return new DiffusionOpdPairedHistoryWorkflowExecutionPlan(
                actionPacket,
                approvalGate,
                executionMode,
                maxCooldownRounds(playbooks),
                minNextCheckWindowSize(playbooks),
                escalationTarget,
                mergedChecklist(playbooks),
                buildPairedWorkflowExecutionMessage(actionPacket, executionMode, approvalGate));
    }

    /**
     * Builds the grouped dispatch envelope from the grouped execution plan and the selected target
     * slices in its remediation bucket.
     */
    private static DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope groupedWorkflowDispatchEnvelope(
            DiffusionOpdGroupedHistoryWorkflowExecutionPlan executionPlan) {
        DiffusionOpdGroupedHistoryWorkflowActionPacket actionPacket = executionPlan.actionPacket();
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile selectedProfile =
                actionPacket.rolloutPlan().recommendedProfile();
        DiffusionOpdGroupedHistoryPortfolioRemediationBucket primaryBucket =
                groupedPrimaryBucket(selectedProfile, actionPacket.primaryAction());
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> entries =
                primaryBucket == null ? List.of() : primaryBucket.entries();
        String targetField = entries.isEmpty() ? null : entries.getFirst().field();
        List<String> targetValues = entries.stream()
                .map(DiffusionOpdGroupedHistoryPortfolioComparisonEntry::value)
                .distinct()
                .toList();
        String dispatchPriority = workflowDispatchPriority(
                actionPacket.primaryAction(),
                executionPlan.approvalGate().approvalRequired(),
                actionPacket.reviewHotspotCount());
        String retryPolicy = workflowRetryPolicy(
                executionPlan.executionMode(),
                actionPacket.primaryAction(),
                executionPlan.cooldownRounds());
        String dispatchKey = workflowDispatchKey(
                "grouped",
                actionPacket.selectedPolicyName(),
                actionPacket.primaryAction(),
                targetField,
                targetValues);
        return new DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope(
                executionPlan,
                targetField,
                targetValues,
                dispatchPriority,
                retryPolicy,
                executionPlan.cooldownRounds(),
                executionPlan.nextCheckWindowSize(),
                dispatchKey,
                buildGroupedWorkflowDispatchMessage(executionPlan, dispatchPriority, targetValues.size()));
    }

    /**
     * Builds the paired dispatch envelope from the paired execution plan and the selected target
     * pairs in its remediation bucket.
     */
    private static DiffusionOpdPairedHistoryWorkflowDispatchEnvelope pairedWorkflowDispatchEnvelope(
            DiffusionOpdPairedHistoryWorkflowExecutionPlan executionPlan) {
        DiffusionOpdPairedHistoryWorkflowActionPacket actionPacket = executionPlan.actionPacket();
        DiffusionOpdPairedHistoryPortfolioPolicyProfile selectedProfile =
                actionPacket.rolloutPlan().recommendedProfile();
        DiffusionOpdPairedHistoryPortfolioRemediationBucket primaryBucket =
                pairedPrimaryBucket(selectedProfile, actionPacket.primaryAction());
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> entries =
                primaryBucket == null ? List.of() : primaryBucket.entries();
        String firstField = entries.isEmpty() ? null : entries.getFirst().firstField();
        String secondField = entries.isEmpty() ? null : entries.getFirst().secondField();
        List<String> targetPairs = entries.stream()
                .map(DiffusionOpdPairedHistoryPortfolioComparisonEntry::pair)
                .distinct()
                .toList();
        String dispatchPriority = workflowDispatchPriority(
                actionPacket.primaryAction(),
                executionPlan.approvalGate().approvalRequired(),
                actionPacket.reviewHotspotCount());
        String retryPolicy = workflowRetryPolicy(
                executionPlan.executionMode(),
                actionPacket.primaryAction(),
                executionPlan.cooldownRounds());
        String dispatchKey = workflowDispatchKey(
                "paired",
                actionPacket.selectedPolicyName(),
                actionPacket.primaryAction(),
                firstField + "+" + secondField,
                targetPairs);
        return new DiffusionOpdPairedHistoryWorkflowDispatchEnvelope(
                executionPlan,
                firstField,
                secondField,
                targetPairs,
                dispatchPriority,
                retryPolicy,
                executionPlan.cooldownRounds(),
                executionPlan.nextCheckWindowSize(),
                dispatchKey,
                buildPairedWorkflowDispatchMessage(executionPlan, dispatchPriority, targetPairs.size()));
    }

    /**
     * Aggregates grouped dispatch envelopes into one grouped workflow batch manifest with batch
     * totals and recheck metadata.
     */
    private static DiffusionOpdGroupedHistoryWorkflowBatchManifest groupedWorkflowBatchManifest(
            List<DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope> envelopes) {
        int targetSliceCount = envelopes.stream()
                .mapToInt(envelope -> envelope.targetValues() == null ? 0 : envelope.targetValues().size())
                .sum();
        int approvalRequiredCount = (int) envelopes.stream()
                .filter(envelope -> Boolean.TRUE.equals(envelope.executionPlan().approvalGate().approvalRequired()))
                .count();
        int manualReviewCount = envelopes.stream()
                .mapToInt(envelope -> envelope.executionPlan().approvalGate().manualReviewCount())
                .sum();
        String highestDispatchPriority = envelopes.stream()
                .map(DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope::dispatchPriority)
                .min(Comparator.comparingInt(DiffusionOpdReportQueries::dispatchPriorityRank))
                .orElse("none");
        Integer earliestRecheckAfterRounds = envelopes.stream()
                .map(DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope::recheckAfterRounds)
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        Integer earliestDeferredRecheckWindowSize = envelopes.stream()
                .map(DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope::deferredRecheckWindowSize)
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        return new DiffusionOpdGroupedHistoryWorkflowBatchManifest(
                envelopes,
                envelopes.size(),
                targetSliceCount,
                approvalRequiredCount,
                manualReviewCount,
                highestDispatchPriority,
                earliestRecheckAfterRounds,
                earliestDeferredRecheckWindowSize,
                workflowBatchKey("grouped", envelopes.size(), highestDispatchPriority, targetSliceCount),
                buildGroupedWorkflowBatchMessage(envelopes.size(), targetSliceCount, highestDispatchPriority));
    }

    /**
     * Aggregates paired dispatch envelopes into one paired workflow batch manifest with batch
     * totals and recheck metadata.
     */
    private static DiffusionOpdPairedHistoryWorkflowBatchManifest pairedWorkflowBatchManifest(
            List<DiffusionOpdPairedHistoryWorkflowDispatchEnvelope> envelopes) {
        int targetSliceCount = envelopes.stream()
                .mapToInt(envelope -> envelope.targetPairs() == null ? 0 : envelope.targetPairs().size())
                .sum();
        int approvalRequiredCount = (int) envelopes.stream()
                .filter(envelope -> Boolean.TRUE.equals(envelope.executionPlan().approvalGate().approvalRequired()))
                .count();
        int manualReviewCount = envelopes.stream()
                .mapToInt(envelope -> envelope.executionPlan().approvalGate().manualReviewCount())
                .sum();
        String highestDispatchPriority = envelopes.stream()
                .map(DiffusionOpdPairedHistoryWorkflowDispatchEnvelope::dispatchPriority)
                .min(Comparator.comparingInt(DiffusionOpdReportQueries::dispatchPriorityRank))
                .orElse("none");
        Integer earliestRecheckAfterRounds = envelopes.stream()
                .map(DiffusionOpdPairedHistoryWorkflowDispatchEnvelope::recheckAfterRounds)
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        Integer earliestDeferredRecheckWindowSize = envelopes.stream()
                .map(DiffusionOpdPairedHistoryWorkflowDispatchEnvelope::deferredRecheckWindowSize)
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        return new DiffusionOpdPairedHistoryWorkflowBatchManifest(
                envelopes,
                envelopes.size(),
                targetSliceCount,
                approvalRequiredCount,
                manualReviewCount,
                highestDispatchPriority,
                earliestRecheckAfterRounds,
                earliestDeferredRecheckWindowSize,
                workflowBatchKey("paired", envelopes.size(), highestDispatchPriority, targetSliceCount),
                buildPairedWorkflowBatchMessage(envelopes.size(), targetSliceCount, highestDispatchPriority));
    }

    /**
     * Combines the grouped and paired workflow batch manifests into the top-level campaign
     * summary.
     */
    static DiffusionOpdCampaignSummary campaignSummary(
            DiffusionOpdGroupedHistoryWorkflowBatchManifest groupedBatchManifest,
            DiffusionOpdPairedHistoryWorkflowBatchManifest pairedBatchManifest) {
        int batchCount = 2;
        int envelopeCount = groupedBatchManifest.envelopeCount() + pairedBatchManifest.envelopeCount();
        int targetSliceCount = groupedBatchManifest.targetSliceCount() + pairedBatchManifest.targetSliceCount();
        int approvalRequiredCount = groupedBatchManifest.approvalRequiredCount()
                + pairedBatchManifest.approvalRequiredCount();
        int manualReviewCount = groupedBatchManifest.manualReviewCount() + pairedBatchManifest.manualReviewCount();
        String highestDispatchPriority = List.of(
                        groupedBatchManifest.highestDispatchPriority(),
                        pairedBatchManifest.highestDispatchPriority())
                .stream()
                .min(Comparator.comparingInt(DiffusionOpdReportQueries::dispatchPriorityRank))
                .orElse("none");
        Integer earliestRecheckAfterRounds = List.of(
                        groupedBatchManifest.earliestRecheckAfterRounds(),
                        pairedBatchManifest.earliestRecheckAfterRounds())
                .stream()
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        Integer earliestDeferredRecheckWindowSize = List.of(
                        groupedBatchManifest.earliestDeferredRecheckWindowSize(),
                        pairedBatchManifest.earliestDeferredRecheckWindowSize())
                .stream()
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
        return new DiffusionOpdCampaignSummary(
                groupedBatchManifest,
                pairedBatchManifest,
                batchCount,
                envelopeCount,
                targetSliceCount,
                approvalRequiredCount,
                manualReviewCount,
                highestDispatchPriority,
                earliestRecheckAfterRounds,
                earliestDeferredRecheckWindowSize,
                campaignKey(highestDispatchPriority, envelopeCount, targetSliceCount),
                buildCampaignMessage(batchCount, envelopeCount, targetSliceCount, highestDispatchPriority));
    }

    /**
     * Builds the campaign rollout pack by ordering grouped and paired batch execution steps and
     * selecting the recommended execution mode.
     */
    static DiffusionOpdCampaignRolloutPack campaignRolloutPack(
            DiffusionOpdCampaignSummary campaignSummary) {
        List<DiffusionOpdCampaignExecutionStep> batchExecutionSteps = List.of(
                        new DiffusionOpdCampaignExecutionStep(
                                "grouped",
                                campaignSummary.groupedBatchManifest().batchKey(),
                                campaignSummary.groupedBatchManifest().highestDispatchPriority(),
                                campaignSummary.groupedBatchManifest().targetSliceCount(),
                                campaignSummary.groupedBatchManifest().approvalRequiredCount(),
                                campaignSummary.groupedBatchManifest().earliestRecheckAfterRounds()),
                        new DiffusionOpdCampaignExecutionStep(
                                "paired",
                                campaignSummary.pairedBatchManifest().batchKey(),
                                campaignSummary.pairedBatchManifest().highestDispatchPriority(),
                                campaignSummary.pairedBatchManifest().targetSliceCount(),
                                campaignSummary.pairedBatchManifest().approvalRequiredCount(),
                                campaignSummary.pairedBatchManifest().earliestRecheckAfterRounds()))
                .stream()
                .sorted(Comparator.comparing(
                                DiffusionOpdCampaignExecutionStep::dispatchPriority,
                                Comparator.comparingInt(DiffusionOpdReportQueries::dispatchPriorityRank))
                        .thenComparing(DiffusionOpdCampaignExecutionStep::approvalRequiredCount, Comparator.reverseOrder())
                        .thenComparing(DiffusionOpdCampaignExecutionStep::targetSliceCount, Comparator.reverseOrder())
                        .thenComparing(DiffusionOpdCampaignExecutionStep::batchScope))
                .toList();
        String recommendedExecutionMode;
        if (campaignSummary.manualReviewCount() > 0) {
            recommendedExecutionMode = "manual";
        } else if (campaignSummary.approvalRequiredCount() > 0) {
            recommendedExecutionMode = "assisted";
        } else {
            recommendedExecutionMode = "automatic";
        }
        String fallbackExecutionMode = "assisted".equals(recommendedExecutionMode) ? "manual" : "assisted";
        String primaryStrategy = campaignSummary.approvalRequiredCount() > 0
                ? "staged_review"
                : "direct_dispatch";
        return new DiffusionOpdCampaignRolloutPack(
                campaignSummary,
                recommendedExecutionMode,
                fallbackExecutionMode,
                primaryStrategy,
                campaignSummary.approvalRequiredCount() > 0,
                batchExecutionSteps,
                buildCampaignRolloutMessage(
                        recommendedExecutionMode,
                        primaryStrategy,
                        batchExecutionSteps.size(),
                        campaignSummary.highestDispatchPriority()));
    }

    /**
     * Builds the campaign approval packet from the rollout pack by extracting approval gates and
     * sign-off reasons.
     */
    static DiffusionOpdCampaignApprovalPacket campaignApprovalPacket(
            DiffusionOpdCampaignRolloutPack rolloutPack) {
        String primaryApproverTarget = rolloutPack.campaignSummary().manualReviewCount() > 0
                ? "human-review"
                : rolloutPack.approvalRequired() ? "training-monitor" : "none";
        List<String> approvalReasonCodes = new ArrayList<>();
        if (rolloutPack.approvalRequired()) {
            approvalReasonCodes.add("approval.gated_batches");
        }
        if (rolloutPack.campaignSummary().manualReviewCount() > 0) {
            approvalReasonCodes.add("approval.manual_review_required");
        }
        if ("high".equals(rolloutPack.campaignSummary().highestDispatchPriority())
                || "urgent".equals(rolloutPack.campaignSummary().highestDispatchPriority())) {
            approvalReasonCodes.add("approval.priority_escalated");
        }
        List<DiffusionOpdCampaignApprovalStep> gatedBatchSteps = rolloutPack.batchExecutionSteps().stream()
                .filter(step -> step.approvalRequiredCount() > 0)
                .map(step -> new DiffusionOpdCampaignApprovalStep(
                        step.batchScope(),
                        step.batchKey(),
                        Boolean.TRUE,
                        "approval.batch_gate",
                        step.targetSliceCount(),
                        primaryApproverTarget))
                .toList();
        return new DiffusionOpdCampaignApprovalPacket(
                rolloutPack,
                rolloutPack.approvalRequired(),
                primaryApproverTarget,
                List.copyOf(approvalReasonCodes),
                gatedBatchSteps,
                gatedBatchSteps.size(),
                rolloutPack.fallbackExecutionMode(),
                buildCampaignApprovalMessage(
                        rolloutPack.recommendedExecutionMode(),
                        primaryApproverTarget,
                        gatedBatchSteps.size()));
    }

    /**
     * Builds the campaign decision log from the approval packet by recording rollout-, approval-,
     * and gated-batch decisions.
     */
    static DiffusionOpdCampaignDecisionLog campaignDecisionLog(
            DiffusionOpdCampaignApprovalPacket approvalPacket) {
        List<DiffusionOpdCampaignDecisionLogEntry> entries = new ArrayList<>();
        entries.add(new DiffusionOpdCampaignDecisionLogEntry(
                "rollout",
                "decision.rollout.execution_mode",
                "campaign",
                "Selected "
                        + approvalPacket.rolloutPack().recommendedExecutionMode()
                        + " mode with "
                        + approvalPacket.rolloutPack().primaryStrategy()
                        + " strategy.",
                approvalPacket.signOffRequired() ? "warning" : "info"));
        entries.add(new DiffusionOpdCampaignDecisionLogEntry(
                "approval",
                "decision.approval.primary_gate",
                "campaign",
                "Routed approval through "
                        + approvalPacket.primaryApproverTarget()
                        + " with "
                        + approvalPacket.blockedBatchCount()
                        + " gated batches.",
                approvalPacket.signOffRequired() ? "warning" : "info"));
        approvalPacket.gatedBatchSteps().forEach(step -> entries.add(new DiffusionOpdCampaignDecisionLogEntry(
                "batch_gate",
                step.reasonCode(),
                step.batchScope(),
                "Batch "
                        + step.batchKey()
                        + " requires sign-off for "
                        + step.targetSliceCount()
                        + " target slices.",
                "warning")));
        String highestSeverity = entries.stream()
                .map(DiffusionOpdCampaignDecisionLogEntry::severity)
                .min(Comparator.comparingInt(DiffusionOpdReportQueries::decisionSeverityRank))
                .orElse("info");
        int gatedDecisionCount = (int) entries.stream()
                .filter(entry -> "batch_gate".equals(entry.decisionType()))
                .count();
        return new DiffusionOpdCampaignDecisionLog(
                approvalPacket,
                List.copyOf(entries),
                entries.size(),
                gatedDecisionCount,
                highestSeverity,
                buildCampaignDecisionLogMessage(entries.size(), gatedDecisionCount, highestSeverity));
    }

    /**
     * Builds the campaign incident report from the decision log by surfacing blocked scopes,
     * severity, and operator-routing details.
     */
    static DiffusionOpdCampaignIncidentReport campaignIncidentReport(
            DiffusionOpdCampaignDecisionLog decisionLog) {
        DiffusionOpdCampaignApprovalPacket approvalPacket = decisionLog.approvalPacket();
        List<String> blockedBatchScopes = approvalPacket.gatedBatchSteps().stream()
                .map(DiffusionOpdCampaignApprovalStep::batchScope)
                .distinct()
                .toList();
        String recommendedOperatorAction = approvalPacket.signOffRequired()
                ? "review_and_sign_off"
                : "monitor_only";
        return new DiffusionOpdCampaignIncidentReport(
                decisionLog,
                approvalPacket.signOffRequired(),
                decisionLog.highestSeverity(),
                approvalPacket.blockedBatchCount(),
                blockedBatchScopes,
                approvalPacket.approvalReasonCodes(),
                approvalPacket.primaryApproverTarget(),
                recommendedOperatorAction,
                buildCampaignIncidentReportMessage(
                        approvalPacket.blockedBatchCount(),
                        decisionLog.highestSeverity(),
                        recommendedOperatorAction));
    }

    /**
     * Builds the campaign resolution packet from the incident report by proposing operator-facing
     * outcomes for each gated batch.
     */
    static DiffusionOpdCampaignResolutionPacket campaignResolutionPacket(
            DiffusionOpdCampaignIncidentReport incidentReport) {
        String suggestedOutcome = incidentReport.incidentOpen() ? "deferred" : "approved";
        String fallbackOutcome = incidentReport.incidentOpen() ? "rejected" : "deferred";
        List<DiffusionOpdCampaignResolutionStep> batchResolutions = incidentReport.decisionLog()
                .approvalPacket()
                .gatedBatchSteps()
                .stream()
                .map(step -> new DiffusionOpdCampaignResolutionStep(
                        step.batchScope(),
                        step.batchKey(),
                        suggestedOutcome,
                        Boolean.TRUE,
                        "resolution.await_operator_response"))
                .toList();
        return new DiffusionOpdCampaignResolutionPacket(
                incidentReport,
                suggestedOutcome,
                fallbackOutcome,
                incidentReport.incidentOpen(),
                batchResolutions,
                buildCampaignResolutionMessage(
                        suggestedOutcome,
                        fallbackOutcome,
                        batchResolutions.size()));
    }

    /**
     * Builds the campaign closure report from the resolution packet by determining whether the
     * campaign is closed or awaiting operator confirmation.
     */
    static DiffusionOpdCampaignClosureReport campaignClosureReport(
            DiffusionOpdCampaignResolutionPacket resolutionPacket) {
        boolean closed = !Boolean.TRUE.equals(resolutionPacket.operatorResponseRequired());
        String finalOutcome = closed ? resolutionPacket.suggestedOutcome() : "pending_operator_response";
        int unresolvedOverrideCount = resolutionPacket.batchResolutions().size();
        String followUpAction = closed ? "archive_campaign" : "await_operator_confirmation";
        return new DiffusionOpdCampaignClosureReport(
                resolutionPacket,
                finalOutcome,
                closed,
                unresolvedOverrideCount,
                followUpAction,
                buildCampaignClosureMessage(
                        finalOutcome,
                        unresolvedOverrideCount,
                        followUpAction));
    }

    /**
     * Builds the campaign handoff bundle from the decision, incident, resolution, and closure
     * artifacts.
     */
    static DiffusionOpdCampaignHandoffBundle campaignHandoffBundle(
            DiffusionOpdCampaignDecisionLog decisionLog,
            DiffusionOpdCampaignIncidentReport incidentReport,
            DiffusionOpdCampaignResolutionPacket resolutionPacket,
            DiffusionOpdCampaignClosureReport closureReport) {
        String handoffTarget = incidentReport.incidentOpen()
                ? incidentReport.primaryOperatorTarget()
                : "workflow-engine";
        String exportStatus = closureReport.closed() ? "finalized" : "pending_review";
        return new DiffusionOpdCampaignHandoffBundle(
                decisionLog,
                incidentReport,
                resolutionPacket,
                closureReport,
                handoffTarget,
                exportStatus,
                buildCampaignHandoffMessage(handoffTarget, exportStatus));
    }

    /**
     * Builds the campaign export manifest from the final handoff bundle by projecting the exported
     * campaign artifacts into flat export records.
     */
    static DiffusionOpdCampaignExportManifest campaignExportManifest(
            DiffusionOpdCampaignHandoffBundle handoffBundle) {
        List<DiffusionOpdCampaignExportRecord> records = List.of(
                new DiffusionOpdCampaignExportRecord(
                        "decision_log",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.decisionLog().approvalPacket().rolloutPack().campaignSummary().campaignKey()),
                new DiffusionOpdCampaignExportRecord(
                        "incident_report",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.incidentReport().decisionLog().approvalPacket().rolloutPack().campaignSummary().campaignKey()),
                new DiffusionOpdCampaignExportRecord(
                        "resolution_packet",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.resolutionPacket().incidentReport().decisionLog().approvalPacket().rolloutPack().campaignSummary().campaignKey()),
                new DiffusionOpdCampaignExportRecord(
                        "closure_report",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.closureReport().resolutionPacket().incidentReport().decisionLog().approvalPacket().rolloutPack().campaignSummary().campaignKey()));
        return new DiffusionOpdCampaignExportManifest(
                handoffBundle,
                records,
                records.size(),
                handoffBundle.handoffTarget(),
                handoffBundle.exportStatus(),
                buildCampaignExportManifestMessage(records.size(), handoffBundle.exportStatus()));
    }

    /**
     * Delegates campaign delivery-report construction to the extracted delivery-planning helper.
     */
    static DiffusionOpdCampaignDeliveryReport campaignDeliveryReport(
            DiffusionOpdCampaignExportManifest exportManifest) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDeliveryReport(exportManifest);
    }

    /**
     * Delegates campaign delivery-ledger construction to the extracted delivery-planning helper.
     */
    static DiffusionOpdCampaignDeliveryLedger campaignDeliveryLedger(
            DiffusionOpdCampaignDeliveryReport deliveryReport) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDeliveryLedger(deliveryReport);
    }

    /**
     * Delegates campaign delivery-receipts construction to the extracted delivery-planning
     * helper.
     */
    static DiffusionOpdCampaignDeliveryReceipts campaignDeliveryReceipts(
            DiffusionOpdCampaignDeliveryLedger deliveryLedger) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDeliveryReceipts(deliveryLedger);
    }

    /**
     * Delegates campaign receipt-acknowledgement construction to the extracted delivery-planning
     * helper.
     */
    static DiffusionOpdCampaignReceiptAcknowledgements campaignReceiptAcknowledgements(
            DiffusionOpdCampaignDeliveryReceipts deliveryReceipts) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignReceiptAcknowledgements(deliveryReceipts);
    }

    /**
     * Delegates campaign acknowledgement-decision construction to the extracted delivery-planning
     * helper.
     */
    static DiffusionOpdCampaignAcknowledgementDecisions campaignAcknowledgementDecisions(
            DiffusionOpdCampaignReceiptAcknowledgements receiptAcknowledgements) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignAcknowledgementDecisions(receiptAcknowledgements);
    }

    /**
     * Delegates campaign decision-resolution construction to the extracted delivery-planning
     * helper.
     */
    static DiffusionOpdCampaignDecisionResolutions campaignDecisionResolutions(
            DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDecisionResolutions(acknowledgementDecisions);
    }

    /**
     * Delegates campaign dispatch-eligibility construction to the extracted delivery-planning
     * helper.
     */
    static DiffusionOpdCampaignDispatchEligibilitySummary campaignDispatchEligibilitySummary(
            DiffusionOpdCampaignDecisionResolutions decisionResolutions) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDispatchEligibilitySummary(decisionResolutions);
    }

    /**
     * Delegates campaign dispatch-plan construction to the extracted delivery-planning helper.
     */
    static DiffusionOpdCampaignDispatchPlan campaignDispatchPlan(
            DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary) {
        return DiffusionOpdCampaignDeliveryPlanning.campaignDispatchPlan(eligibilitySummary);
    }

    /**
     * Delegates campaign execution-packet construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignExecutionPacket campaignExecutionPacket(
            DiffusionOpdCampaignDispatchPlan dispatchPlan) {
        return DiffusionOpdCampaignExecutionReleases.campaignExecutionPacket(dispatchPlan);
    }

    /**
     * Delegates campaign engine-handoff construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignEngineHandoffEnvelope campaignEngineHandoffEnvelope(
            DiffusionOpdCampaignExecutionPacket executionPacket) {
        return DiffusionOpdCampaignExecutionReleases.campaignEngineHandoffEnvelope(executionPacket);
    }

    /**
     * Delegates campaign execution-manifest construction to the extracted execution/release
     * helper.
     */
    static DiffusionOpdCampaignExecutionManifest campaignExecutionManifest(
            DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope) {
        return DiffusionOpdCampaignExecutionReleases.campaignExecutionManifest(handoffEnvelope);
    }

    /**
     * Delegates campaign execution-review construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignExecutionReview campaignExecutionReview(
            DiffusionOpdCampaignExecutionManifest executionManifest) {
        return DiffusionOpdCampaignExecutionReleases.campaignExecutionReview(executionManifest);
    }

    /**
     * Delegates campaign release-decision construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleaseDecision campaignReleaseDecision(
            DiffusionOpdCampaignExecutionReview executionReview) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseDecision(executionReview);
    }

    /**
     * Delegates campaign release-packet construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleasePacket campaignReleasePacket(
            DiffusionOpdCampaignReleaseDecision releaseDecision) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleasePacket(releaseDecision);
    }

    /**
     * Delegates campaign release-ledger construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleaseLedger campaignReleaseLedger(
            DiffusionOpdCampaignReleasePacket releasePacket) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseLedger(releasePacket);
    }

    /**
     * Delegates campaign release-receipt construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleaseReceipt campaignReleaseReceipt(
            DiffusionOpdCampaignReleaseLedger releaseLedger) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseReceipt(releaseLedger);
    }

    /**
     * Delegates campaign release-acknowledgement construction to the extracted execution/release
     * helper.
     */
    static DiffusionOpdCampaignReleaseAcknowledgement campaignReleaseAcknowledgement(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseAcknowledgement(releaseReceipt);
    }

    /**
     * Delegates campaign release-resolution construction to the extracted execution/release
     * helper.
     */
    static DiffusionOpdCampaignReleaseResolution campaignReleaseResolution(
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseResolution(releaseAcknowledgement);
    }

    /**
     * Delegates campaign release-closure construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleaseClosure campaignReleaseClosure(
            DiffusionOpdCampaignReleaseResolution releaseResolution) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseClosure(releaseResolution);
    }

    /**
     * Delegates campaign release-handoff construction to the extracted execution/release helper.
     */
    static DiffusionOpdCampaignReleaseHandoffBundle campaignReleaseHandoffBundle(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt,
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
            DiffusionOpdCampaignReleaseResolution releaseResolution,
            DiffusionOpdCampaignReleaseClosure releaseClosure) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseHandoffBundle(
                releaseReceipt,
                releaseAcknowledgement,
                releaseResolution,
                releaseClosure);
    }

    /**
     * Delegates campaign release-export-manifest construction to the extracted execution/release
     * helper.
     */
    static DiffusionOpdCampaignReleaseExportManifest campaignReleaseExportManifest(
            DiffusionOpdCampaignReleaseHandoffBundle handoffBundle) {
        return DiffusionOpdCampaignExecutionReleases.campaignReleaseExportManifest(handoffBundle);
    }

    /**
     * Delegates conversion from release-export records to release-delivery export records to the
     * extracted release-delivery export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportRecord toCampaignReleaseDeliveryExportRecord(
            DiffusionOpdCampaignReleaseExportRecord record) {
        return DiffusionOpdCampaignReleaseDeliveryExports.toCampaignReleaseDeliveryExportRecord(record);
    }

    /**
     * Delegates seeded release-delivery export-manifest construction to the extracted
     * release-delivery export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportManifest campaignReleaseDeliveryExportSeedManifest(
            List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records,
            String primaryDestination,
            String overallDeliveryStatus) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportSeedManifest(
                records,
                primaryDestination,
                overallDeliveryStatus);
    }

    /**
     * Delegates release-delivery export-manifest construction from handoff bundles to the
     * extracted release-delivery export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportManifest campaignReleaseDeliveryExportManifest(
            List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> handoffBundles) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportManifest(handoffBundles);
    }

    /**
     * Delegates release-delivery export-report construction to the extracted release-delivery
     * export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportReport campaignReleaseDeliveryExportReport(
            DiffusionOpdCampaignReleaseDeliveryExportManifest exportManifest) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportReport(exportManifest);
    }

    /**
     * Delegates release-delivery export-ledger construction to the extracted release-delivery
     * export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportLedger campaignReleaseDeliveryExportLedger(
            DiffusionOpdCampaignReleaseDeliveryExportReport exportReport) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportLedger(exportReport);
    }

    /**
     * Delegates release-delivery export-receipts construction to the extracted release-delivery
     * export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportReceipts campaignReleaseDeliveryExportReceipts(
            DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportReceipts(exportLedger);
    }

    /**
     * Delegates release-delivery export-acknowledgements construction to the extracted
     * release-delivery export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements campaignReleaseDeliveryExportAcknowledgements(
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportAcknowledgements(exportReceipts);
    }

    /**
     * Delegates release-delivery export-decisions construction to the extracted release-delivery
     * export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportDecisions campaignReleaseDeliveryExportDecisions(
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportDecisions(exportAcknowledgements);
    }

    /**
     * Delegates release-delivery export-resolutions construction to the extracted
     * release-delivery export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportResolutions campaignReleaseDeliveryExportResolutions(
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportResolutions(exportDecisions);
    }

    /**
     * Delegates release-delivery export-closures construction to the extracted release-delivery
     * export helper.
     */
    static DiffusionOpdCampaignReleaseDeliveryExportClosures campaignReleaseDeliveryExportClosures(
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions) {
        return DiffusionOpdCampaignReleaseDeliveryExports.campaignReleaseDeliveryExportClosures(exportResolutions);
    }

    /**
     * Delegates release-delivery export-handoff-bundle construction to the extracted
     * release-delivery export helper.
     */
    static List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> buildCampaignReleaseDeliveryExportHandoffBundles(
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts,
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements,
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions,
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions,
            DiffusionOpdCampaignReleaseDeliveryExportClosures exportClosures) {
        return DiffusionOpdCampaignReleaseDeliveryExports.buildCampaignReleaseDeliveryExportHandoffBundles(
                exportReceipts,
                exportAcknowledgements,
                exportDecisions,
                exportResolutions,
                exportClosures);
    }

    private static String buildGroupedWorkflowPacketMessage(
            DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan rolloutPlan,
            String primaryAction) {
        return "Execute grouped workflow with policy "
                + (rolloutPlan.recommendedProfile() == null ? "none" : rolloutPlan.recommendedProfile().policyName())
                + " and primary action "
                + primaryAction
                + ".";
    }

    private static String buildPairedWorkflowPacketMessage(
            DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan rolloutPlan,
            String primaryAction) {
        return "Execute paired workflow with policy "
                + (rolloutPlan.recommendedProfile() == null ? "none" : rolloutPlan.recommendedProfile().policyName())
                + " and primary action "
                + primaryAction
                + ".";
    }

    private static String buildGroupedWorkflowExecutionMessage(
            DiffusionOpdGroupedHistoryWorkflowActionPacket actionPacket,
            String executionMode,
            DiffusionOpdWorkflowApprovalGate approvalGate) {
        return "Execute grouped workflow in "
                + executionMode
                + " mode with action "
                + actionPacket.primaryAction()
                + " under policy "
                + actionPacket.selectedPolicyName()
                + " using gate "
                + approvalGate.gateCode()
                + ".";
    }

    private static String buildPairedWorkflowExecutionMessage(
            DiffusionOpdPairedHistoryWorkflowActionPacket actionPacket,
            String executionMode,
            DiffusionOpdWorkflowApprovalGate approvalGate) {
        return "Execute paired workflow in "
                + executionMode
                + " mode with action "
                + actionPacket.primaryAction()
                + " under policy "
                + actionPacket.selectedPolicyName()
                + " using gate "
                + approvalGate.gateCode()
                + ".";
    }

    private static String buildGroupedWorkflowDispatchMessage(
            DiffusionOpdGroupedHistoryWorkflowExecutionPlan executionPlan,
            String dispatchPriority,
            int targetCount) {
        return "Dispatch grouped workflow with priority "
                + dispatchPriority
                + " for "
                + targetCount
                + " target slices under action "
                + executionPlan.actionPacket().primaryAction()
                + ".";
    }

    private static String buildPairedWorkflowDispatchMessage(
            DiffusionOpdPairedHistoryWorkflowExecutionPlan executionPlan,
            String dispatchPriority,
            int targetCount) {
        return "Dispatch paired workflow with priority "
                + dispatchPriority
                + " for "
                + targetCount
                + " target slices under action "
                + executionPlan.actionPacket().primaryAction()
                + ".";
    }

    private static String buildGroupedWorkflowBatchMessage(
            int envelopeCount,
            int targetSliceCount,
            String highestDispatchPriority) {
        return "Prepared grouped workflow batch with "
                + envelopeCount
                + " envelopes across "
                + targetSliceCount
                + " target slices at "
                + highestDispatchPriority
                + " priority.";
    }

    private static String buildPairedWorkflowBatchMessage(
            int envelopeCount,
            int targetSliceCount,
            String highestDispatchPriority) {
        return "Prepared paired workflow batch with "
                + envelopeCount
                + " envelopes across "
                + targetSliceCount
                + " target slices at "
                + highestDispatchPriority
                + " priority.";
    }

    private static String buildCampaignMessage(
            int batchCount,
            int envelopeCount,
            int targetSliceCount,
            String highestDispatchPriority) {
        return "Prepared workflow campaign with "
                + batchCount
                + " batches, "
                + envelopeCount
                + " envelopes, and "
                + targetSliceCount
                + " target slices at "
                + highestDispatchPriority
                + " priority.";
    }

    private static String buildCampaignRolloutMessage(
            String recommendedExecutionMode,
            String primaryStrategy,
            int batchCount,
            String highestDispatchPriority) {
        return "Roll out workflow campaign in "
                + recommendedExecutionMode
                + " mode using "
                + primaryStrategy
                + " across "
                + batchCount
                + " ordered batches at "
                + highestDispatchPriority
                + " priority.";
    }

    private static String buildCampaignApprovalMessage(
            String recommendedExecutionMode,
            String primaryApproverTarget,
            int gatedBatchCount) {
        return "Approve workflow campaign for "
                + recommendedExecutionMode
                + " execution through "
                + primaryApproverTarget
                + " with "
                + gatedBatchCount
                + " gated batches.";
    }

    private static String buildCampaignDecisionLogMessage(
            int decisionCount,
            int gatedDecisionCount,
            String highestSeverity) {
        return "Recorded workflow campaign decision log with "
                + decisionCount
                + " decisions, including "
                + gatedDecisionCount
                + " gated batch decisions at "
                + highestSeverity
                + " severity.";
    }

    private static String buildCampaignIncidentReportMessage(
            int blockedBatchCount,
            String highestSeverity,
            String recommendedOperatorAction) {
        return "Workflow campaign incident report shows "
                + blockedBatchCount
                + " blocked batches at "
                + highestSeverity
                + " severity with operator action "
                + recommendedOperatorAction
                + ".";
    }

    private static String buildCampaignResolutionMessage(
            String suggestedOutcome,
            String fallbackOutcome,
            int batchResolutionCount) {
        return "Prepared workflow campaign resolution packet with suggested outcome "
                + suggestedOutcome
                + ", fallback "
                + fallbackOutcome
                + ", and "
                + batchResolutionCount
                + " batch resolutions.";
    }

    private static String buildCampaignClosureMessage(
            String finalOutcome,
            int unresolvedOverrideCount,
            String followUpAction) {
        return "Workflow campaign closure report records final outcome "
                + finalOutcome
                + " with "
                + unresolvedOverrideCount
                + " unresolved overrides and follow-up action "
                + followUpAction
                + ".";
    }

    private static String buildCampaignHandoffMessage(
            String handoffTarget,
            String exportStatus) {
        return "Prepared workflow campaign handoff bundle for "
                + handoffTarget
                + " with export status "
                + exportStatus
                + ".";
    }

    private static String buildCampaignExportManifestMessage(
            int recordCount,
            String overallDeliveryStatus) {
        return "Prepared workflow campaign export manifest with "
                + recordCount
                + " records at "
                + overallDeliveryStatus
                + " status.";
    }

    static String buildCampaignDeliveryDestinationMessage(
            String destination,
            int artifactCount,
            int pendingCount,
            int blockedCount) {
        return "Prepared workflow campaign delivery summary for "
                + destination
                + " with "
                + artifactCount
                + " artifacts, "
                + pendingCount
                + " pending deliveries, and "
                + blockedCount
                + " blocked deliveries.";
    }

    static String buildCampaignDeliveryReportMessage(
            int totalArtifacts,
            int pendingArtifacts,
            int blockedArtifacts) {
        return "Prepared workflow campaign delivery report with "
                + totalArtifacts
                + " artifacts, "
                + pendingArtifacts
                + " pending deliveries, and "
                + blockedArtifacts
                + " blocked deliveries.";
    }

    static String buildCampaignDeliveryLedgerEntryMessage(
            String artifactType,
            String destination,
            String acknowledgementStatus,
            String retryPolicy) {
        return "Recorded delivery ledger entry for "
                + artifactType
                + " to "
                + destination
                + " with acknowledgement "
                + acknowledgementStatus
                + " and retry policy "
                + retryPolicy
                + ".";
    }

    static String buildCampaignDeliveryLedgerMessage(
            int entryCount,
            int pendingAcknowledgementCount,
            int retryRequiredCount) {
        return "Prepared workflow campaign delivery ledger with "
                + entryCount
                + " entries, "
                + pendingAcknowledgementCount
                + " pending acknowledgements, and "
                + retryRequiredCount
                + " retry-required entries.";
    }

    static String buildCampaignDeliveryReceiptMessage(
            String artifactType,
            String destination,
            String receiptStatus,
            String operatorAcknowledgement) {
        return "Prepared workflow campaign delivery receipt for "
                + artifactType
                + " to "
                + destination
                + " with receipt status "
                + receiptStatus
                + " and operator acknowledgement "
                + operatorAcknowledgement
                + ".";
    }

    static String buildCampaignDeliveryReceiptsMessage(
            int receiptCount,
            int pendingCount,
            int operatorActionRequiredCount) {
        return "Prepared workflow campaign delivery receipts with "
                + receiptCount
                + " receipts, "
                + pendingCount
                + " pending confirmations, and "
                + operatorActionRequiredCount
                + " operator-action-required receipts.";
    }

    static String buildCampaignReceiptAcknowledgementMessage(
            String receiptId,
            String acknowledgementOutcome,
            String reviewer) {
        return "Prepared workflow campaign receipt acknowledgement for "
                + receiptId
                + " with outcome "
                + acknowledgementOutcome
                + " and reviewer "
                + reviewer
                + ".";
    }

    static String buildCampaignReceiptAcknowledgementsMessage(
            int acknowledgementCount,
            int pendingCount,
            int rejectedCount) {
        return "Prepared workflow campaign receipt acknowledgements with "
                + acknowledgementCount
                + " entries, "
                + pendingCount
                + " pending reviews, and "
                + rejectedCount
                + " rejected acknowledgements.";
    }

    static String buildCampaignAcknowledgementDecisionMessage(
            String receiptId,
            String decision,
            String resolutionRoute) {
        return "Prepared workflow campaign acknowledgement decision for "
                + receiptId
                + " with decision "
                + decision
                + " routed to "
                + resolutionRoute
                + ".";
    }

    static String buildCampaignAcknowledgementDecisionsMessage(
            int decisionCount,
            int deferredCount,
            int rejectedCount) {
        return "Prepared workflow campaign acknowledgement decisions with "
                + decisionCount
                + " entries, "
                + deferredCount
                + " deferred decisions, and "
                + rejectedCount
                + " rejected decisions.";
    }

    static String buildCampaignDecisionResolutionMessage(
            String receiptId,
            String resolution,
            String executionReadiness) {
        return "Prepared campaign decision resolution for "
                + receiptId
                + " with resolution "
                + resolution
                + " and execution readiness "
                + executionReadiness
                + ".";
    }

    static String buildCampaignDecisionResolutionsMessage(
            int resolutionCount,
            int blockedCount,
            int escalationRequiredCount) {
        return "Prepared campaign decision resolutions with "
                + resolutionCount
                + " entries, "
                + blockedCount
                + " blocked resolutions, and "
                + escalationRequiredCount
                + " escalation-required resolutions.";
    }

    static String buildCampaignDispatchEligibilityEntryMessage(
            String receiptId,
            String eligibility,
            String executionReadiness) {
        return "Prepared campaign dispatch eligibility entry for "
                + receiptId
                + " with eligibility "
                + eligibility
                + " and execution readiness "
                + executionReadiness
                + ".";
    }

    static String buildCampaignDispatchEligibilitySummaryMessage(
            int entryCount,
            int dispatchableCount,
            int escalationRequiredCount) {
        return "Prepared campaign dispatch eligibility summary with "
                + entryCount
                + " entries, "
                + dispatchableCount
                + " dispatchable entries, and "
                + escalationRequiredCount
                + " escalation-required entries.";
    }

    static String buildCampaignDispatchPlanStepMessage(
            String receiptId,
            String executionMode,
            String dispatchTarget) {
        return "Prepared campaign dispatch plan step for "
                + receiptId
                + " using "
                + executionMode
                + " mode toward "
                + dispatchTarget
                + ".";
    }

    static String buildCampaignDispatchPlanMessage(
            int stepCount,
            int blockedCount,
            int escalationRequiredCount) {
        return "Prepared campaign dispatch plan with "
                + stepCount
                + " executable steps, "
                + blockedCount
                + " blocked entries, and "
                + escalationRequiredCount
                + " escalation-required entries.";
    }

    static String buildCampaignExecutionPacketMessage(
            int runnableStepCount,
            int blockedEntryCount,
            int escalationRequiredCount) {
        return "Prepared campaign execution packet with "
                + runnableStepCount
                + " runnable steps, "
                + blockedEntryCount
                + " blocked entries, and "
                + escalationRequiredCount
                + " escalation-required entries.";
    }

    static String buildCampaignEngineHandoffEnvelopeMessage(
            String engineId,
            String submissionMode,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign engine handoff envelope for "
                + engineId
                + " in "
                + submissionMode
                + " mode with "
                + runnableStepCount
                + " runnable steps and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignExecutionManifestMessage(
            String queueName,
            String batchMode,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign execution manifest for queue "
                + queueName
                + " in "
                + batchMode
                + " mode with "
                + runnableStepCount
                + " runnable steps and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignExecutionReviewMessage(
            String reviewerTarget,
            String reviewOutcome,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign execution review for "
                + reviewerTarget
                + " with outcome "
                + reviewOutcome
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseDecisionMessage(
            String releaseOutcome,
            String fallbackOutcome,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release decision with outcome "
                + releaseOutcome
                + ", fallback "
                + fallbackOutcome
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleasePacketMessage(
            String dispatchTarget,
            String packetOutcome,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release packet for "
                + dispatchTarget
                + " with outcome "
                + packetOutcome
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseLedgerMessage(
            String releaseStatus,
            String acknowledgementStatus,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release ledger with status "
                + releaseStatus
                + ", acknowledgement "
                + acknowledgementStatus
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseReceiptMessage(
            String receiptStatus,
            String finalAcknowledgement,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release receipt with status "
                + receiptStatus
                + ", acknowledgement "
                + finalAcknowledgement
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseAcknowledgementMessage(
            String acknowledgementOutcome,
            String reviewer,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release acknowledgement with outcome "
                + acknowledgementOutcome
                + " by "
                + reviewer
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseResolutionMessage(
            String resolution,
            String escalationTarget,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release resolution "
                + resolution
                + " with escalation target "
                + escalationTarget
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseClosureMessage(
            String finalOutcome,
            String followUpAction,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release closure with outcome "
                + finalOutcome
                + ", follow-up "
                + followUpAction
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseHandoffBundleMessage(
            String handoffTarget,
            String exportStatus,
            int runnableStepCount,
            int blockedEntryCount) {
        return "Prepared campaign release handoff bundle for "
                + handoffTarget
                + " with export status "
                + exportStatus
                + ", "
                + runnableStepCount
                + " runnable steps, and "
                + blockedEntryCount
                + " blocked entries.";
    }

    static String buildCampaignReleaseExportManifestMessage(
            int recordCount,
            String exportStatus) {
        return "Prepared campaign release export manifest with "
                + recordCount
                + " records and status "
                + exportStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryDestinationMessage(
            String destination,
            int artifactCount,
            int blockedCount) {
        return "Prepared release delivery destination "
                + destination
                + " with "
                + artifactCount
                + " artifacts and "
                + blockedCount
                + " blocked artifacts.";
    }

    private static String buildCampaignReleaseDeliveryLedgerEntryMessage(
            String artifactType,
            String destination,
            String deliveryStatus) {
        return "Prepared release delivery ledger entry for "
                + artifactType
                + " to "
                + destination
                + " with status "
                + deliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryReceiptMessage(
            String artifactType,
            String destination,
            String deliveryStatus) {
        return "Prepared release delivery receipt for "
                + artifactType
                + " to "
                + destination
                + " with delivery status "
                + deliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryAcknowledgementMessage(
            String artifactType,
            String destination,
            String receiptStatus) {
        return "Prepared release delivery acknowledgement for "
                + artifactType
                + " to "
                + destination
                + " with receipt status "
                + receiptStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryDecisionMessage(
            String artifactType,
            String destination,
            String decision) {
        return "Prepared release delivery decision for "
                + artifactType
                + " to "
                + destination
                + " with decision "
                + decision
                + ".";
    }

    private static String buildCampaignReleaseDeliveryResolutionMessage(
            String artifactType,
            String destination,
            String resolution) {
        return "Prepared release delivery resolution for "
                + artifactType
                + " to "
                + destination
                + " with resolution "
                + resolution
                + ".";
    }

    private static String buildCampaignReleaseDeliveryClosureMessage(
            String artifactType,
            String destination,
            String finalOutcome) {
        return "Prepared release delivery closure for "
                + artifactType
                + " to "
                + destination
                + " with final outcome "
                + finalOutcome
                + ".";
    }

    private static String buildCampaignReleaseDeliveryHandoffBundleMessage(
            String artifactType,
            String handoffTarget,
            String exportStatus) {
        return "Prepared release delivery handoff bundle for "
                + artifactType
                + " to "
                + handoffTarget
                + " with export status "
                + exportStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportManifestMessage(
            int recordCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export manifest with "
                + recordCount
                + " artifacts and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportReportMessage(
            int totalArtifacts,
            int pendingArtifacts,
            int blockedArtifacts,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export report with "
                + totalArtifacts
                + " artifacts, "
                + pendingArtifacts
                + " pending, "
                + blockedArtifacts
                + " blocked, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportLedgerMessage(
            int entryCount,
            int pendingAcknowledgementCount,
            int retryRequiredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export ledger with "
                + entryCount
                + " entries, "
                + pendingAcknowledgementCount
                + " pending acknowledgements, "
                + retryRequiredCount
                + " retry-required entries, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportLedgerEntryMessage(
            String artifactType,
            String destination,
            String deliveryStatus) {
        return "Prepared release delivery export ledger entry for "
                + artifactType
                + " to "
                + destination
                + " with delivery status "
                + deliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportReceiptsMessage(
            int receiptCount,
            int pendingCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export receipts with "
                + receiptCount
                + " receipts, "
                + pendingCount
                + " pending confirmations, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportReceiptMessage(
            String artifactType,
            String destination,
            String deliveryStatus) {
        return "Prepared release delivery export receipt for "
                + artifactType
                + " to "
                + destination
                + " with delivery status "
                + deliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportAcknowledgementsMessage(
            int acknowledgementCount,
            int pendingCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export acknowledgements with "
                + acknowledgementCount
                + " acknowledgements, "
                + pendingCount
                + " pending reviews, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportAcknowledgementMessage(
            String artifactType,
            String destination,
            String receiptStatus) {
        return "Prepared release delivery export acknowledgement for "
                + artifactType
                + " to "
                + destination
                + " with receipt status "
                + receiptStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportDecisionsMessage(
            int decisionCount,
            int deferredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export decisions with "
                + decisionCount
                + " decisions, "
                + deferredCount
                + " deferred, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportDecisionMessage(
            String artifactType,
            String destination,
            String decision) {
        return "Prepared release delivery export decision for "
                + artifactType
                + " to "
                + destination
                + " with decision "
                + decision
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportResolutionsMessage(
            int resolutionCount,
            int deferredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export resolutions with "
                + resolutionCount
                + " resolutions, "
                + deferredCount
                + " deferred, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportResolutionMessage(
            String artifactType,
            String destination,
            String resolution) {
        return "Prepared release delivery export resolution for "
                + artifactType
                + " to "
                + destination
                + " with resolution "
                + resolution
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportClosuresMessage(
            int closureCount,
            int openCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery export closures with "
                + closureCount
                + " closures, "
                + openCount
                + " open, and overall delivery status "
                + overallDeliveryStatus
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportClosureMessage(
            String artifactType,
            String destination,
            String finalOutcome) {
        return "Prepared release delivery export closure for "
                + artifactType
                + " to "
                + destination
                + " with final outcome "
                + finalOutcome
                + ".";
    }

    static String buildCampaignReleaseDeliveryExportHandoffBundleMessage(
            String artifactType,
            String handoffTarget,
            String exportStatus) {
        return "Prepared release delivery export handoff bundle for "
                + artifactType
                + " to "
                + handoffTarget
                + " with export status "
                + exportStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryReportMessage(
            int totalArtifacts,
            int pendingArtifacts,
            int blockedArtifacts,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery report with "
                + totalArtifacts
                + " artifacts, "
                + pendingArtifacts
                + " pending artifacts, "
                + blockedArtifacts
                + " blocked artifacts, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryLedgerMessage(
            int entryCount,
            int pendingAcknowledgementCount,
            int retryRequiredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery ledger with "
                + entryCount
                + " entries, "
                + pendingAcknowledgementCount
                + " pending acknowledgements, "
                + retryRequiredCount
                + " retry-required entries, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryReceiptsMessage(
            int receiptCount,
            int pendingCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery receipts with "
                + receiptCount
                + " receipts, "
                + pendingCount
                + " pending confirmations, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryAcknowledgementsMessage(
            int acknowledgementCount,
            int pendingCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery acknowledgements with "
                + acknowledgementCount
                + " acknowledgements, "
                + pendingCount
                + " pending reviews, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryDecisionsMessage(
            int decisionCount,
            int deferredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery decisions with "
                + decisionCount
                + " decisions, "
                + deferredCount
                + " deferred decisions, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryResolutionsMessage(
            int resolutionCount,
            int deferredCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery resolutions with "
                + resolutionCount
                + " resolutions, "
                + deferredCount
                + " deferred resolutions, and status "
                + overallDeliveryStatus
                + ".";
    }

    private static String buildCampaignReleaseDeliveryClosuresMessage(
            int closureCount,
            int openCount,
            String overallDeliveryStatus) {
        return "Prepared campaign release delivery closures with "
                + closureCount
                + " closures, "
                + openCount
                + " open closures, and status "
                + overallDeliveryStatus
                + ".";
    }

    static boolean isDeliveredDeliveryStatus(String deliveryStatus) {
        return "finalized".equals(deliveryStatus)
                || "delivered".equals(deliveryStatus)
                || "completed".equals(deliveryStatus);
    }

    static boolean isPendingDeliveryStatus(String deliveryStatus) {
        return "pending".equals(deliveryStatus)
                || "pending_review".equals(deliveryStatus)
                || "queued".equals(deliveryStatus);
    }

    static boolean isBlockedDeliveryStatus(String deliveryStatus) {
        return "blocked".equals(deliveryStatus)
                || "failed".equals(deliveryStatus)
                || "rejected".equals(deliveryStatus);
    }

    static String acknowledgementStatus(String deliveryStatus) {
        if (isDeliveredDeliveryStatus(deliveryStatus)) {
            return "acknowledged";
        }
        if (isBlockedDeliveryStatus(deliveryStatus)) {
            return "rejected";
        }
        return "pending_ack";
    }

    static String deliveryRetryPolicy(String deliveryStatus) {
        if (isBlockedDeliveryStatus(deliveryStatus)) {
            return "retry_after_review";
        }
        if (isPendingDeliveryStatus(deliveryStatus)) {
            return "await_review";
        }
        return "no_retry";
    }

    static boolean isAcknowledgedDeliveryStatus(String acknowledgementStatus) {
        return "acknowledged".equals(acknowledgementStatus);
    }

    static boolean isPendingAcknowledgementStatus(String acknowledgementStatus) {
        return "pending_ack".equals(acknowledgementStatus);
    }

    static boolean requiresRetry(String retryPolicy) {
        return "retry_after_review".equals(retryPolicy);
    }

    static String receiptStatus(String acknowledgementStatus) {
        if (isAcknowledgedDeliveryStatus(acknowledgementStatus)) {
            return "confirmed";
        }
        if ("rejected".equals(acknowledgementStatus)) {
            return "rejected";
        }
        return "pending_confirmation";
    }

    static String operatorAcknowledgement(
            String acknowledgementStatus,
            String retryPolicy) {
        if (requiresRetry(retryPolicy) || "rejected".equals(acknowledgementStatus)) {
            return "operator_review_required";
        }
        if (isPendingAcknowledgementStatus(acknowledgementStatus)) {
            return "awaiting_operator_confirmation";
        }
        return "not_required";
    }

    static String deliveryReceiptId(String referenceKey, String artifactType) {
        return referenceKey + ":" + artifactType + ":receipt";
    }

    static String deliveryTimestamp(String deliveryStatus) {
        return isDeliveredDeliveryStatus(deliveryStatus)
                ? "delivered_at_dispatch"
                : "awaiting_delivery_confirmation";
    }

    static boolean isConfirmedReceiptStatus(String receiptStatus) {
        return "confirmed".equals(receiptStatus);
    }

    static boolean isPendingReceiptStatus(String receiptStatus) {
        return "pending_confirmation".equals(receiptStatus);
    }

    static String receiptAcknowledgementOutcome(
            String receiptStatus,
            String operatorAcknowledgement) {
        if ("confirmed".equals(receiptStatus)) {
            return "accepted";
        }
        if ("operator_review_required".equals(operatorAcknowledgement)) {
            return "rejected";
        }
        return "pending_review";
    }

    static String receiptReviewer(String operatorAcknowledgement) {
        return "operator_review_required".equals(operatorAcknowledgement)
                ? "training-monitor"
                : "delivery-bot";
    }

    static String receiptReviewerRole(String operatorAcknowledgement) {
        return "operator_review_required".equals(operatorAcknowledgement)
                ? "human_reviewer"
                : "automated_dispatch";
    }

    static boolean isFinalAcknowledgementOutcome(String acknowledgementOutcome) {
        return "accepted".equals(acknowledgementOutcome)
                || "rejected".equals(acknowledgementOutcome);
    }

    static String acknowledgementDecision(String acknowledgementOutcome) {
        if ("accepted".equals(acknowledgementOutcome)) {
            return "approved";
        }
        if ("rejected".equals(acknowledgementOutcome)) {
            return "rejected";
        }
        return "deferred";
    }

    static String acknowledgementDecisionNote(String decision) {
        return switch (decision) {
            case "approved" -> "Receipt acknowledged and ready for final handoff.";
            case "rejected" -> "Receipt requires rejection handling and escalation.";
            default -> "Receipt requires follow-up review before final approval.";
        };
    }

    static String acknowledgementResolutionRoute(String decision) {
        return switch (decision) {
            case "approved" -> "scheduler_dispatch";
            case "rejected" -> "manual_escalation";
            default -> "review_queue";
        };
    }

    static boolean isFinalAcknowledgementDecision(String decision) {
        return "approved".equals(decision) || "rejected".equals(decision);
    }

    static String campaignDecisionResolutionValue(String decision) {
        return switch (decision) {
            case "approved" -> "dispatch_ready";
            case "rejected" -> "escalated";
            default -> "awaiting_review";
        };
    }

    static String campaignEscalationOwner(String resolution) {
        return switch (resolution) {
            case "escalated" -> "training-monitor";
            case "awaiting_review" -> "review-queue";
            default -> "none";
        };
    }

    static String campaignExecutionReadiness(String resolution) {
        return "dispatch_ready".equals(resolution) ? "ready" : "blocked";
    }

    static String campaignDownstreamEligibility(String resolution) {
        return "dispatch_ready".equals(resolution) ? "eligible" : "hold";
    }

    static boolean isFinalCampaignDecisionResolution(String resolution) {
        return "dispatch_ready".equals(resolution) || "escalated".equals(resolution);
    }

    private static DiffusionOpdGroupedHistoryPortfolioRemediationBucket groupedPrimaryBucket(
            DiffusionOpdGroupedHistoryPortfolioPolicyProfile selectedProfile,
            String primaryAction) {
        return selectedProfile == null
                ? null
                : selectedProfile.remediationSummary().actionBuckets().stream()
                        .filter(bucket -> Objects.equals(bucket.action(), primaryAction))
                        .findFirst()
                        .orElse(null);
    }

    private static DiffusionOpdPairedHistoryPortfolioRemediationBucket pairedPrimaryBucket(
            DiffusionOpdPairedHistoryPortfolioPolicyProfile selectedProfile,
            String primaryAction) {
        return selectedProfile == null
                ? null
                : selectedProfile.remediationSummary().actionBuckets().stream()
                        .filter(bucket -> Objects.equals(bucket.action(), primaryAction))
                        .findFirst()
                        .orElse(null);
    }

    private static List<DiffusionOpdRoundHistoryPlaybook> groupedBucketPlaybooks(
            DiffusionOpdGroupedHistoryPortfolioRemediationBucket bucket) {
        return bucket == null
                ? List.of()
                : bucket.entries().stream()
                        .map(entry -> entry.current().snapshot().playbook())
                        .filter(Objects::nonNull)
                        .toList();
    }

    private static List<DiffusionOpdRoundHistoryPlaybook> pairedBucketPlaybooks(
            DiffusionOpdPairedHistoryPortfolioRemediationBucket bucket) {
        return bucket == null
                ? List.of()
                : bucket.entries().stream()
                        .map(entry -> entry.current().snapshot().playbook())
                        .filter(Objects::nonNull)
                        .toList();
    }

    private static DiffusionOpdWorkflowApprovalGate workflowApprovalGate(
            String primaryAction,
            Boolean automationFriendly,
            Boolean reviewRequired,
            int manualReviewCount,
            int reviewHotspotCount,
            String escalationTarget) {
        boolean reviewGate = Boolean.TRUE.equals(reviewRequired);
        boolean approvalRequired = reviewGate
                || manualReviewCount > 0
                || actionSeverityRank(primaryAction) >= 2
                || !Boolean.TRUE.equals(automationFriendly);
        String gateCode;
        if (reviewGate) {
            gateCode = "approval.human_review_required";
        } else if (manualReviewCount > 0) {
            gateCode = "approval.manual_review_load";
        } else if (actionSeverityRank(primaryAction) >= 2) {
            gateCode = "approval.action_escalated";
        } else if (!Boolean.TRUE.equals(automationFriendly)) {
            gateCode = "approval.assisted_execution";
        } else {
            gateCode = "approval.none";
        }
        return new DiffusionOpdWorkflowApprovalGate(
                approvalRequired,
                gateCode,
                escalationTarget,
                reviewRequired,
                manualReviewCount,
                reviewHotspotCount);
    }

    private static String workflowExecutionMode(
            String primaryAction,
            Boolean automationFriendly,
            Boolean reviewRequired) {
        if (Boolean.TRUE.equals(reviewRequired)) {
            return "manual";
        }
        if (Boolean.TRUE.equals(automationFriendly) && actionSeverityRank(primaryAction) <= 1) {
            return "automatic";
        }
        return "assisted";
    }

    private static String workflowDispatchPriority(
            String primaryAction,
            Boolean approvalRequired,
            int reviewHotspotCount) {
        if (actionSeverityRank(primaryAction) >= 3) {
            return "urgent";
        }
        if (actionSeverityRank(primaryAction) >= 2 || reviewHotspotCount >= 3) {
            return "high";
        }
        if (Boolean.TRUE.equals(approvalRequired)) {
            return "normal";
        }
        return "low";
    }

    private static String workflowRetryPolicy(
            String executionMode,
            String primaryAction,
            Integer cooldownRounds) {
        if ("manual".equals(executionMode)) {
            return "manual_only";
        }
        if (actionSeverityRank(primaryAction) >= 2) {
            return "after_cooldown";
        }
        if (cooldownRounds != null && cooldownRounds > 0) {
            return "deferred_recheck";
        }
        return "immediate";
    }

    private static int dispatchPriorityRank(String priority) {
        return switch (priority) {
            case "urgent" -> 0;
            case "high" -> 1;
            case "normal" -> 2;
            case "low" -> 3;
            default -> 4;
        };
    }

    private static int decisionSeverityRank(String severity) {
        return switch (severity) {
            case "critical" -> 0;
            case "warning" -> 1;
            case "info" -> 2;
            default -> 3;
        };
    }

    private static Integer maxCooldownRounds(List<DiffusionOpdRoundHistoryPlaybook> playbooks) {
        return playbooks.stream()
                .map(DiffusionOpdRoundHistoryPlaybook::cooldownRounds)
                .filter(Objects::nonNull)
                .max(Integer::compareTo)
                .orElse(null);
    }

    private static Integer minNextCheckWindowSize(List<DiffusionOpdRoundHistoryPlaybook> playbooks) {
        return playbooks.stream()
                .map(DiffusionOpdRoundHistoryPlaybook::nextCheckWindowSize)
                .filter(Objects::nonNull)
                .min(Integer::compareTo)
                .orElse(null);
    }

    private static String firstEscalationTarget(List<DiffusionOpdRoundHistoryPlaybook> playbooks) {
        return playbooks.stream()
                .map(DiffusionOpdRoundHistoryPlaybook::escalationTarget)
                .filter(Objects::nonNull)
                .findFirst()
                .orElse("none");
    }

    private static List<String> mergedChecklist(List<DiffusionOpdRoundHistoryPlaybook> playbooks) {
        LinkedHashSet<String> checklist = new LinkedHashSet<>();
        for (DiffusionOpdRoundHistoryPlaybook playbook : playbooks) {
            if (playbook.checklist() != null) {
                checklist.addAll(playbook.checklist());
            }
        }
        return List.copyOf(checklist);
    }

    private static String workflowDispatchKey(
            String scope,
            String policyName,
            String primaryAction,
            String targetDimension,
            List<String> targets) {
        return scope
                + ":"
                + Objects.toString(policyName, "none")
                + ":"
                + Objects.toString(primaryAction, "none")
                + ":"
                + Objects.toString(targetDimension, "none")
                + ":"
                + targets.size();
    }

    private static String workflowBatchKey(
            String scope,
            int envelopeCount,
            String highestDispatchPriority,
            int targetSliceCount) {
        return scope
                + ":batch:"
                + highestDispatchPriority
                + ":"
                + envelopeCount
                + ":"
                + targetSliceCount;
    }

    private static String campaignKey(
            String highestDispatchPriority,
            int envelopeCount,
            int targetSliceCount) {
        return "campaign:"
                + highestDispatchPriority
                + ":"
                + envelopeCount
                + ":"
                + targetSliceCount;
    }

    private static int actionSeverityRank(String action) {
        return switch (action) {
            case "continue" -> 0;
            case "monitor" -> 1;
            case "reduce_lr" -> 2;
            case "early_stop_review" -> 3;
            default -> 4;
        };
    }

    private static int policyPreferenceRank(String policyName) {
        return switch (policyName) {
            case "default" -> 0;
            case "lenient" -> 1;
            case "strict" -> 2;
            default -> 3;
        };
    }

    private static boolean isEscalated(DiffusionOpdGroupedHistoryPortfolioComparisonEntry entry) {
        return alertLevelRank(entry.current().snapshot().status().alertLevel())
                > alertLevelRank(entry.baseline().snapshot().status().alertLevel());
    }

    private static boolean isEscalated(DiffusionOpdPairedHistoryPortfolioComparisonEntry entry) {
        return alertLevelRank(entry.current().snapshot().status().alertLevel())
                > alertLevelRank(entry.baseline().snapshot().status().alertLevel());
    }

    private static boolean isHealthyToUnhealthy(DiffusionOpdGroupedHistoryPortfolioComparisonEntry entry) {
        return Boolean.TRUE.equals(entry.baseline().snapshot().status().healthy())
                && !Boolean.TRUE.equals(entry.current().snapshot().status().healthy());
    }

    private static boolean isHealthyToUnhealthy(DiffusionOpdPairedHistoryPortfolioComparisonEntry entry) {
        return Boolean.TRUE.equals(entry.baseline().snapshot().status().healthy())
                && !Boolean.TRUE.equals(entry.current().snapshot().status().healthy());
    }

    private static boolean isUnhealthyToHealthy(DiffusionOpdGroupedHistoryPortfolioComparisonEntry entry) {
        return !Boolean.TRUE.equals(entry.baseline().snapshot().status().healthy())
                && Boolean.TRUE.equals(entry.current().snapshot().status().healthy());
    }

    private static boolean isUnhealthyToHealthy(DiffusionOpdPairedHistoryPortfolioComparisonEntry entry) {
        return !Boolean.TRUE.equals(entry.baseline().snapshot().status().healthy())
                && Boolean.TRUE.equals(entry.current().snapshot().status().healthy());
    }

    private static int alertLevelRank(String alertLevel) {
        return switch (alertLevel) {
            case "critical" -> 3;
            case "warning" -> 2;
            case "info" -> 1;
            default -> 0;
        };
    }

    private static DiffusionOpdGroupedHistoryPortfolio groupedPortfolio(
            List<DiffusionOpdRoundHistoryRow> rows,
            String field,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        List<DiffusionOpdGroupedHistoryDashboardEntry> entries = groupedSummaries(rows, field).stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .map(summary -> {
                    List<DiffusionOpdRoundHistoryRow> filteredRows = rowsForFieldValue(rows, field, summary.value());
                    return new DiffusionOpdGroupedHistoryDashboardEntry(
                            summary.field(),
                            summary.value(),
                            summary.summary(),
                            snapshot(filteredRows, mediumWindowSize, policy),
                            timeline(filteredRows, shortWindowSize, mediumWindowSize, longWindowSize, policy));
                })
                .toList();
        return new DiffusionOpdGroupedHistoryPortfolio(field, entries);
    }

    private static DiffusionOpdPairedHistoryPortfolio pairedPortfolio(
            List<DiffusionOpdRoundHistoryRow> rows,
            String firstField,
            String secondField,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            DiffusionOpdRoundHistoryStatusPolicy policy) {
        List<DiffusionOpdPairedHistoryDashboardEntry> entries = pairedSummaries(rows, firstField, secondField).stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .map(summary -> {
                    List<DiffusionOpdRoundHistoryRow> filteredRows =
                            rowsForPairValues(rows, firstField, summary.firstValue(), secondField, summary.secondValue());
                    return new DiffusionOpdPairedHistoryDashboardEntry(
                            summary.firstField(),
                            summary.firstValue(),
                            summary.secondField(),
                            summary.secondValue(),
                            summary.pair(),
                            summary.summary(),
                            snapshot(filteredRows, mediumWindowSize, policy),
                            timeline(filteredRows, shortWindowSize, mediumWindowSize, longWindowSize, policy));
                })
                .toList();
        return new DiffusionOpdPairedHistoryPortfolio(firstField, secondField, entries);
    }

    private static List<DiffusionOpdRoundHistoryRow> rowsForFieldValue(
            List<DiffusionOpdRoundHistoryRow> rows,
            String field,
            String value) {
        return rows.stream()
                .filter(row -> Objects.equals(fieldValue(row, field), value))
                .toList();
    }

    private static List<DiffusionOpdRoundHistoryRow> rowsForPairValues(
            List<DiffusionOpdRoundHistoryRow> rows,
            String firstField,
            String firstValue,
            String secondField,
            String secondValue) {
        return rows.stream()
                .filter(row -> Objects.equals(fieldValue(row, firstField), firstValue))
                .filter(row -> Objects.equals(fieldValue(row, secondField), secondValue))
                .toList();
    }

    private static String fieldValue(DiffusionOpdRoundHistoryRow row, String field) {
        return switch (field) {
            case "taskId" -> row.taskId();
            case "teacherKey" -> row.teacherKey();
            case "stageName" -> row.stageName();
            default -> throw new IllegalArgumentException("Unsupported field: " + field);
        };
    }

    private static Boolean improvingAcrossWindows(
            DiffusionOpdRoundHistorySnapshot shortSnapshot,
            DiffusionOpdRoundHistorySnapshot mediumSnapshot,
            DiffusionOpdRoundHistorySnapshot longSnapshot) {
        Double shortMean = shortSnapshot.trend().meanLoss();
        Double mediumMean = mediumSnapshot.trend().meanLoss();
        Double longMean = longSnapshot.trend().meanLoss();
        if (shortMean == null || mediumMean == null || longMean == null) {
            return null;
        }
        return shortMean <= mediumMean && mediumMean <= longMean;
    }

    private static Boolean stableAcrossWindows(
            DiffusionOpdRoundHistorySnapshot shortSnapshot,
            DiffusionOpdRoundHistorySnapshot mediumSnapshot,
            DiffusionOpdRoundHistorySnapshot longSnapshot) {
        return !Boolean.TRUE.equals(shortSnapshot.status().unstable())
                && !Boolean.TRUE.equals(mediumSnapshot.status().unstable())
                && !Boolean.TRUE.equals(longSnapshot.status().unstable())
                && !Boolean.TRUE.equals(shortSnapshot.status().regressing())
                && !Boolean.TRUE.equals(mediumSnapshot.status().regressing())
                && !Boolean.TRUE.equals(longSnapshot.status().regressing());
    }

    private static Boolean consistentRecommendation(
            DiffusionOpdRoundHistorySnapshot shortSnapshot,
            DiffusionOpdRoundHistorySnapshot mediumSnapshot,
            DiffusionOpdRoundHistorySnapshot longSnapshot) {
        return Objects.equals(shortSnapshot.recommendation().action(), mediumSnapshot.recommendation().action())
                && Objects.equals(mediumSnapshot.recommendation().action(), longSnapshot.recommendation().action());
    }

    private static Boolean requiresEscalation(
            DiffusionOpdRoundHistorySnapshot shortSnapshot,
            DiffusionOpdRoundHistorySnapshot mediumSnapshot,
            DiffusionOpdRoundHistorySnapshot longSnapshot) {
        return Boolean.TRUE.equals(shortSnapshot.playbook().reviewRequired())
                || Boolean.TRUE.equals(mediumSnapshot.playbook().reviewRequired())
                || Boolean.TRUE.equals(longSnapshot.playbook().reviewRequired());
    }

    private static Double meanLoss(List<DiffusionOpdRoundHistoryRow> rows) {
        OptionalDouble mean = rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .average();
        return mean.isPresent() ? mean.getAsDouble() : null;
    }

    private static Double minLoss(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .min(Double::compareTo)
                .orElse(null);
    }

    private static List<Double> stepDeltas(List<DiffusionOpdRoundHistoryRow> rows) {
        ArrayList<Double> deltas = new ArrayList<>();
        for (int i = 1; i < rows.size(); i++) {
            Double previous = rows.get(i - 1).averageLoss();
            Double current = rows.get(i).averageLoss();
            if (previous != null && current != null) {
                deltas.add(current - previous);
            }
        }
        return deltas;
    }

    private static Double average(List<Double> values) {
        OptionalDouble mean = values.stream()
                .mapToDouble(Double::doubleValue)
                .average();
        return mean.isPresent() ? mean.getAsDouble() : null;
    }

    private static Double averageAbsolute(List<Double> values) {
        OptionalDouble mean = values.stream()
                .mapToDouble(Math::abs)
                .average();
        return mean.isPresent() ? mean.getAsDouble() : null;
    }

    private static List<DiffusionOpdGroupedHistorySummary> topGroupedSummariesByMeanLoss(
            List<DiffusionOpdGroupedHistorySummary> summaries,
            int limit) {
        return summaries.stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit(limit))
                .toList();
    }

    private static List<DiffusionOpdPairedHistorySummary> topPairedSummariesByMeanLoss(
            List<DiffusionOpdPairedHistorySummary> summaries,
            int limit) {
        return summaries.stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit(limit))
                .toList();
    }

    private static int normalizedLimit(int limit) {
        if (limit < 0) {
            throw new IllegalArgumentException("limit must be >= 0 but was " + limit);
        }
        return limit;
    }

    private static int normalizedWindowSize(int windowSize) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException("windowSize must be > 0 but was " + windowSize);
        }
        return windowSize;
    }

    private static <T> T firstOrNull(List<T> values) {
        return values.isEmpty() ? null : values.getFirst();
    }

    private static List<DiffusionOpdGroupedHistorySummary> groupedSummaries(
            List<DiffusionOpdRoundHistoryRow> rows,
            String field) {
        LinkedHashMap<String, List<DiffusionOpdRoundHistoryRow>> grouped = new LinkedHashMap<>();
        for (DiffusionOpdRoundHistoryRow row : rows) {
            String value = switch (field) {
                case "taskId" -> row.taskId();
                case "teacherKey" -> row.teacherKey();
                case "stageName" -> row.stageName();
                default -> throw new IllegalArgumentException("Unsupported grouped field '" + field + "'.");
            };
            if (value == null || value.isBlank()) {
                continue;
            }
            grouped.computeIfAbsent(value, ignored -> new ArrayList<>()).add(row);
        }
        return grouped.entrySet().stream()
                .map(entry -> new DiffusionOpdGroupedHistorySummary(
                        field,
                        entry.getKey(),
                        summarizeRows(entry.getValue())))
                .toList();
    }

    private static List<DiffusionOpdPairedHistorySummary> pairedSummaries(
            List<DiffusionOpdRoundHistoryRow> rows,
            String firstField,
            String secondField) {
        LinkedHashMap<String, List<DiffusionOpdRoundHistoryRow>> grouped = new LinkedHashMap<>();
        Map<String, String[]> values = new LinkedHashMap<>();
        for (DiffusionOpdRoundHistoryRow row : rows) {
            String firstValue = fieldValue(row, firstField);
            String secondValue = fieldValue(row, secondField);
            if (firstValue == null || firstValue.isBlank() || secondValue == null || secondValue.isBlank()) {
                continue;
            }
            String pair = firstValue + "," + secondValue;
            grouped.computeIfAbsent(pair, ignored -> new ArrayList<>()).add(row);
            values.putIfAbsent(pair, new String[] {firstValue, secondValue});
        }
        return grouped.entrySet().stream()
                .map(entry -> {
                    String pair = entry.getKey();
                    String[] pairValues = values.get(pair);
                    return new DiffusionOpdPairedHistorySummary(
                            firstField,
                            pairValues[0],
                            secondField,
                            pairValues[1],
                            pair,
                            summarizeRows(entry.getValue()));
                })
                .toList();
    }

    private static DiffusionOpdRoundHistorySummary summarizeRows(
            List<DiffusionOpdRoundHistoryRow> rows) {
        OptionalDouble meanLoss = rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .average();
        DiffusionOpdRoundHistoryRow last = rows.stream()
                .max(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .orElse(null);
        List<DiffusionOpdRoundHistoryRow> topLosses = rows.stream()
                .sorted(Comparator.comparing(
                        DiffusionOpdRoundHistoryRow::averageLoss,
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(3)
                .toList();
        List<DiffusionOpdRoundHistoryRow> firstRounds = rows.stream()
                .sorted(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .limit(3)
                .toList();
        return new DiffusionOpdRoundHistorySummary(
                rows.size(),
                meanLoss.isPresent() ? meanLoss.getAsDouble() : null,
                last,
                topLosses,
                firstRounds);
    }
}
