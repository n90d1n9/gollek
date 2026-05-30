package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryDashboardEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolio;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioComparison;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioDeltaDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioDriftDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioExecutiveSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryPortfolioRemediationSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowActionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowBatchManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistoryWorkflowExecutionPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryDashboardEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolio;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioComparison;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioDeltaDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioDriftDashboard;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioExecutiveSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryPortfolioRemediationSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowActionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowBatchManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowDispatchEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistoryWorkflowExecutionPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryDelta;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryPlaybook;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSectionView;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignApprovalPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionLog;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilitySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionReview;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlan;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReport;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignIncidentReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignClosureReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignRolloutPack;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignResolutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignSummary;

class DiffusionOpdReportQueriesTest {

    @Test
    void exposesTypedRoundHistoryRows() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdRoundHistoryRow> rows = DiffusionOpdReportQueries.roundHistoryRows(report);

        assertEquals(3, rows.size());
        assertEquals(1, rows.getFirst().round());
        assertEquals("ocr", rows.getFirst().taskId());
        assertEquals("caption-main", rows.getLast().teacherKey());
    }

    @Test
    void summarizesTaskHistoryThroughTypedApi() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistorySummary summary =
                DiffusionOpdReportQueries.summarizeTaskHistory(report, "ocr");

        assertEquals(2, summary.count());
        assertEquals(0.55d, summary.meanLoss());
        assertEquals(2, summary.last().round());
        assertEquals("ocr-early", summary.last().teacherKey());
        assertEquals(0.60d, summary.topLosses().getFirst().averageLoss());
        assertEquals(1, summary.firstRounds().getFirst().round());
    }

    @Test
    void exposesTypedReportSections() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdSectionView tasks = DiffusionOpdReportQueries.tasks(report);
        DiffusionOpdSectionView conditioning = DiffusionOpdReportQueries.conditioning(report);

        assertEquals("tasks", tasks.name());
        assertEquals("teacher-specialized", tasks.string("mode"));
        assertEquals("conditioning", conditioning.name());
        assertEquals(2.0d, conditioning.number("laneCount"));
        assertEquals(Map.of("ocr", "clip-ocr"), conditioning.objectMap("fixtures"));
    }

    @Test
    void exposesTypedGroupedRollups() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedHistorySummary> tasks =
                DiffusionOpdReportQueries.taskSummaries(report);

        assertEquals(2, tasks.size());
        assertEquals("taskId", tasks.getFirst().field());
        assertEquals("ocr", tasks.getFirst().value());
        assertEquals(2, tasks.getFirst().summary().count());
        assertEquals(0.55d, tasks.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedPairedRollups() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedHistorySummary> pairs =
                DiffusionOpdReportQueries.taskTeacherSummaries(report);

        assertEquals(3, pairs.size());
        assertEquals("taskId", pairs.getFirst().firstField());
        assertEquals("teacherKey", pairs.getFirst().secondField());
        assertEquals("ocr", pairs.getFirst().firstValue());
        assertEquals("ocr-base", pairs.getFirst().secondValue());
        assertEquals("ocr,ocr-base", pairs.getFirst().pair());
        assertEquals(1, pairs.getFirst().summary().count());
    }

    @Test
    void exposesTypedGroupedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTaskSummariesByMeanLoss(report, 1);

        assertEquals(1, leaderboard.size());
        assertEquals("caption", leaderboard.getFirst().value());
        assertEquals(0.90d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedPairedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTaskTeacherSummariesByMeanLoss(report, 2);

        assertEquals(2, leaderboard.size());
        assertEquals("caption,caption-main", leaderboard.getFirst().pair());
        assertEquals(0.90d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedScalarAggregates() {
        DiffusionOpdReport report = sampleReport();

        assertEquals(0.55d, DiffusionOpdReportQueries.meanLossForTask(report, "ocr"));
        assertEquals(0.90d, DiffusionOpdReportQueries.maxLossForTeacher(report, "caption-main"));
        assertEquals(2, DiffusionOpdReportQueries.lastRoundForStage(report, "early"));
    }

    @Test
    void exposesTypedLatestRowHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryRow latestTask = DiffusionOpdReportQueries.latestTaskRow(report, "ocr");
        DiffusionOpdRoundHistoryRow latestTeacherStageForTask =
                DiffusionOpdReportQueries.latestTeacherStageRowForTask(report, "ocr");

        assertEquals(2, latestTask.round());
        assertEquals("ocr-early", latestTask.teacherKey());
        assertEquals(2, latestTeacherStageForTask.round());
        assertEquals("early", latestTeacherStageForTask.stageName());
    }

    @Test
    void exposesTypedDeltaHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryDelta taskDelta = DiffusionOpdReportQueries.taskDelta(report, "ocr");
        DiffusionOpdRoundHistoryDelta stageDelta = DiffusionOpdReportQueries.stageDelta(report, "early");

        assertEquals(2, taskDelta.latest().round());
        assertEquals(1, taskDelta.previous().round());
        assertEquals(-0.10d, taskDelta.lossDelta(), 1e-9);
        assertEquals(1, taskDelta.roundDelta());

        assertEquals(2, stageDelta.latest().round());
        assertEquals(1, stageDelta.previous().round());
        assertEquals(-0.10d, stageDelta.lossDelta(), 1e-9);
    }

    @Test
    void exposesTypedRecentTrendHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryTrend taskTrend =
                DiffusionOpdReportQueries.taskTrend(report, "ocr", 2);
        DiffusionOpdRoundHistoryTrend taskTeacherTrend =
                DiffusionOpdReportQueries.taskTeacherTrendForStage(report, "early", 2);

        assertEquals(2, taskTrend.requestedWindowSize());
        assertEquals(2, taskTrend.rowCount());
        assertEquals(1, taskTrend.earliest().round());
        assertEquals(2, taskTrend.latest().round());
        assertEquals(0.55d, taskTrend.meanLoss(), 1e-9);
        assertEquals(0.50d, taskTrend.minLoss(), 1e-9);
        assertEquals(0.60d, taskTrend.maxLoss(), 1e-9);
        assertEquals(-0.10d, taskTrend.lossDelta(), 1e-9);
        assertEquals(1, taskTrend.roundDelta());
        assertEquals(Boolean.TRUE, taskTrend.improving());
        assertEquals(Boolean.FALSE, taskTrend.worsening());

        assertEquals(2, taskTeacherTrend.rowCount());
        assertEquals(2, taskTeacherTrend.latest().round());
        assertEquals(-0.10d, taskTeacherTrend.lossDelta(), 1e-9);
    }

    @Test
    void exposesTypedStabilityHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryStability taskStability =
                DiffusionOpdReportQueries.taskStability(report, "ocr", 2);
        DiffusionOpdRoundHistoryStability taskTeacherStability =
                DiffusionOpdReportQueries.taskTeacherStabilityForStage(report, "early", 2);

        assertEquals(0.10d, taskStability.lossRange(), 1e-9);
        assertEquals(-0.10d, taskStability.averageSignedStepDelta(), 1e-9);
        assertEquals(0.10d, taskStability.averageAbsoluteStepDelta(), 1e-9);
        assertEquals(0.10d, taskStability.latestAbsoluteStepDelta(), 1e-9);
        assertEquals(Boolean.FALSE, taskStability.volatileWindow());
        assertEquals(Boolean.TRUE, taskStability.stabilizing());

        assertEquals(2, taskTeacherStability.trend().rowCount());
        assertEquals(Boolean.TRUE, taskTeacherStability.stabilizing());
    }

    @Test
    void exposesTypedStatusHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryStatus taskStatus =
                DiffusionOpdReportQueries.taskStatus(report, "ocr", 2);
        DiffusionOpdRoundHistoryStatus taskTeacherStatus =
                DiffusionOpdReportQueries.taskTeacherStatusForStage(report, "early", 2);

        assertEquals("healthy", taskStatus.status());
        assertEquals("info", taskStatus.alertLevel());
        assertEquals("trend.healthy.default", taskStatus.issueCode());
        assertEquals(Boolean.TRUE, taskStatus.healthy());
        assertEquals(Boolean.FALSE, taskStatus.regressing());
        assertEquals(Boolean.FALSE, taskStatus.unstable());

        assertEquals("healthy", taskTeacherStatus.status());
        assertEquals(Boolean.TRUE, taskTeacherStatus.healthy());
    }

    @Test
    void exposesTypedPolicyDrivenStatusHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryStatus strictStatus =
                DiffusionOpdReportQueries.taskStatus(
                        report,
                        "ocr",
                        2,
                        DiffusionOpdRoundHistoryStatusPolicy.strict());

        assertEquals("unstable", strictStatus.status());
        assertEquals("warning", strictStatus.alertLevel());
        assertEquals("trend.unstable", strictStatus.issueCode());
        assertEquals(Boolean.FALSE, strictStatus.healthy());
        assertEquals(Boolean.FALSE, strictStatus.regressing());
        assertEquals(Boolean.TRUE, strictStatus.unstable());
    }

    @Test
    void exposesTypedRecommendationHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryRecommendation taskRecommendation =
                DiffusionOpdReportQueries.taskRecommendation(report, "ocr", 2);
        DiffusionOpdRoundHistoryRecommendation strictRecommendation =
                DiffusionOpdReportQueries.taskRecommendation(
                        report,
                        "ocr",
                        2,
                        DiffusionOpdRoundHistoryStatusPolicy.strict());

        assertEquals("continue", taskRecommendation.action());
        assertEquals("low", taskRecommendation.priority());
        assertEquals("action.continue.healthy", taskRecommendation.reasonCode());
        assertEquals(Boolean.TRUE, taskRecommendation.automationFriendly());

        assertEquals("monitor", strictRecommendation.action());
        assertEquals("medium", strictRecommendation.priority());
        assertEquals("action.monitor.unstable", strictRecommendation.reasonCode());
        assertEquals(Boolean.TRUE, strictRecommendation.automationFriendly());
    }

    @Test
    void exposesTypedPlaybookHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryPlaybook taskPlaybook =
                DiffusionOpdReportQueries.taskPlaybook(report, "ocr", 2);
        DiffusionOpdRoundHistoryPlaybook strictPlaybook =
                DiffusionOpdReportQueries.taskPlaybook(
                        report,
                        "ocr",
                        2,
                        DiffusionOpdRoundHistoryStatusPolicy.strict());

        assertEquals("continue", taskPlaybook.recommendation().action());
        assertEquals(Boolean.FALSE, taskPlaybook.reviewRequired());
        assertEquals(0, taskPlaybook.cooldownRounds());
        assertEquals("none", taskPlaybook.escalationTarget());

        assertEquals("monitor", strictPlaybook.recommendation().action());
        assertEquals(Boolean.FALSE, strictPlaybook.reviewRequired());
        assertEquals(1, strictPlaybook.cooldownRounds());
        assertEquals("training-monitor", strictPlaybook.escalationTarget());
    }

    @Test
    void exposesTypedSnapshotHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistorySnapshot taskSnapshot =
                DiffusionOpdReportQueries.taskSnapshot(report, "ocr", 2);
        DiffusionOpdRoundHistorySnapshot strictSnapshot =
                DiffusionOpdReportQueries.taskSnapshot(
                        report,
                        "ocr",
                        2,
                        DiffusionOpdRoundHistoryStatusPolicy.strict());

        assertEquals(2, taskSnapshot.trend().rowCount());
        assertEquals("healthy", taskSnapshot.status().status());
        assertEquals("continue", taskSnapshot.recommendation().action());
        assertEquals("none", taskSnapshot.playbook().escalationTarget());

        assertEquals("unstable", strictSnapshot.status().status());
        assertEquals("monitor", strictSnapshot.recommendation().action());
        assertEquals("training-monitor", strictSnapshot.playbook().escalationTarget());
    }

    @Test
    void exposesTypedTimelineHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdRoundHistoryTimeline taskTimeline =
                DiffusionOpdReportQueries.taskTimeline(report, "ocr", 1, 2, 3);
        DiffusionOpdRoundHistoryTimeline strictTimeline =
                DiffusionOpdReportQueries.taskTimeline(
                        report,
                        "ocr",
                        1,
                        2,
                        3,
                        DiffusionOpdRoundHistoryStatusPolicy.strict());

        assertEquals(1, taskTimeline.shortWindow().trend().rowCount());
        assertEquals(2, taskTimeline.mediumWindow().trend().rowCount());
        assertEquals(2, taskTimeline.longWindow().trend().rowCount());
        assertEquals(Boolean.TRUE, taskTimeline.improvingAcrossWindows());
        assertEquals(Boolean.TRUE, taskTimeline.stableAcrossWindows());
        assertEquals(Boolean.FALSE, taskTimeline.consistentRecommendation());
        assertEquals(Boolean.FALSE, taskTimeline.requiresEscalation());

        assertEquals("insufficient_data", strictTimeline.shortWindow().status().status());
        assertEquals("unstable", strictTimeline.mediumWindow().status().status());
        assertEquals(Boolean.FALSE, strictTimeline.stableAcrossWindows());
    }

    @Test
    void exposesTypedGroupedPortfolioHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdGroupedHistoryPortfolio portfolio =
                DiffusionOpdReportQueries.taskPortfolio(report, 1, 2, 3);

        assertEquals("taskId", portfolio.field());
        assertEquals(2, portfolio.entries().size());

        DiffusionOpdGroupedHistoryDashboardEntry first = portfolio.entries().getFirst();
        assertEquals("caption", first.value());
        assertEquals(0.90d, first.summary().meanLoss());
        assertEquals(1, first.snapshot().trend().rowCount());
        assertEquals("monitor", first.timeline().mediumWindow().recommendation().action());
    }

    @Test
    void exposesTypedPairedPortfolioHelpers() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdPairedHistoryPortfolio portfolio =
                DiffusionOpdReportQueries.taskTeacherPortfolio(
                        report,
                        1,
                        2,
                        3,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals("taskId", portfolio.firstField());
        assertEquals("teacherKey", portfolio.secondField());
        assertEquals(3, portfolio.entries().size());

        DiffusionOpdPairedHistoryDashboardEntry first = portfolio.entries().getFirst();
        assertEquals("caption,caption-main", first.pair());
        assertEquals(0.90d, first.summary().meanLoss());
        assertEquals(1, first.snapshot().trend().rowCount());
        assertEquals(Boolean.TRUE, first.timeline().consistentRecommendation());
    }

    @Test
    void exposesTypedCrossRunSnapshotComparisons() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport improved = improvedReport();

        DiffusionOpdRoundHistorySnapshotComparison comparison =
                DiffusionOpdReportQueries.compareTaskSnapshots(
                        baseline,
                        improved,
                        "ocr",
                        2);

        assertEquals(-0.075d, comparison.meanLossDelta(), 1e-9);
        assertEquals(-0.10d, comparison.latestLossDelta(), 1e-9);
        assertEquals(Boolean.TRUE, comparison.improved());
        assertEquals(Boolean.FALSE, comparison.statusChanged());
        assertEquals(Boolean.FALSE, comparison.recommendationChanged());
    }

    @Test
    void exposesTypedCrossRunGroupedPortfolioComparisons() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport improved = improvedReport();

        DiffusionOpdGroupedHistoryPortfolioComparison comparison =
                DiffusionOpdReportQueries.compareTaskPortfolios(
                        baseline,
                        improved,
                        1,
                        2,
                        3);

        assertEquals("taskId", comparison.field());
        assertEquals(2, comparison.entries().size());
        assertEquals("caption", comparison.entries().getFirst().value());
        assertEquals(-0.10d, comparison.entries().getFirst().meanLossDelta(), 1e-9);
        assertEquals(Boolean.TRUE, comparison.entries().getFirst().improved());
        assertEquals(Boolean.FALSE, comparison.entries().getFirst().statusChanged());
    }

    @Test
    void exposesTypedCrossRunPairedPortfolioComparisons() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport improved = improvedReport();

        DiffusionOpdPairedHistoryPortfolioComparison comparison =
                DiffusionOpdReportQueries.compareTaskTeacherPortfolios(
                        baseline,
                        improved,
                        1,
                        2,
                        3,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals("taskId", comparison.firstField());
        assertEquals("teacherKey", comparison.secondField());
        assertEquals(3, comparison.entries().size());

        var earlyPair = comparison.entries().stream()
                .filter(entry -> "ocr,ocr-early".equals(entry.pair()))
                .findFirst()
                .orElseThrow();
        assertEquals(-0.10d, earlyPair.meanLossDelta(), 1e-9);
        assertEquals(Boolean.TRUE, earlyPair.improved());
        assertEquals(Boolean.FALSE, earlyPair.statusChanged());
    }

    @Test
    void exposesTypedGroupedPortfolioDeltaDashboards() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport improved = improvedReport();

        DiffusionOpdGroupedHistoryPortfolioDeltaDashboard dashboard =
                DiffusionOpdReportQueries.taskPortfolioDeltaDashboard(
                        baseline,
                        improved,
                        1,
                        2,
                        3,
                        2);

        assertEquals(2, dashboard.totalEntries());
        assertEquals(2, dashboard.improvedCount());
        assertEquals(0, dashboard.regressedCount());
        assertEquals(0, dashboard.recommendationChangedCount());
        assertEquals(2, dashboard.biggestImprovements().size());
        assertEquals("caption", dashboard.biggestImprovements().getFirst().value());
        assertEquals(-0.10d, dashboard.biggestImprovements().getFirst().meanLossDelta(), 1e-9);
        assertEquals(0, dashboard.biggestRegressions().size());
    }

    @Test
    void exposesTypedPairedPortfolioDeltaDashboards() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport improved = improvedReport();

        DiffusionOpdPairedHistoryPortfolioDeltaDashboard dashboard =
                DiffusionOpdReportQueries.taskTeacherPortfolioDeltaDashboard(
                        baseline,
                        improved,
                        1,
                        2,
                        3,
                        2,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals(3, dashboard.totalEntries());
        assertEquals(3, dashboard.improvedCount());
        assertEquals(0, dashboard.regressedCount());
        assertEquals(0, dashboard.recommendationChangedCount());
        assertEquals(2, dashboard.biggestImprovements().size());
        assertEquals(-0.10d, dashboard.biggestImprovements().getFirst().meanLossDelta(), 1e-9);
        assertEquals(0, dashboard.biggestRegressions().size());
    }

    @Test
    void exposesTypedGroupedPortfolioDriftDashboards() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryPortfolioDriftDashboard dashboard =
                DiffusionOpdReportQueries.taskPortfolioDriftDashboard(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(2, dashboard.totalEntries());
        assertEquals(1, dashboard.statusChangedCount());
        assertEquals(1, dashboard.recommendationChangedCount());
        assertEquals(1, dashboard.escalatedCount());
        assertEquals(1, dashboard.healthyToUnhealthyCount());
        assertEquals(0, dashboard.unhealthyToHealthyCount());
        assertEquals(1, dashboard.statusChangedEntries().size());
        assertEquals("ocr", dashboard.statusChangedEntries().getFirst().value());
        assertEquals(1, dashboard.recommendationChangedEntries().size());
        assertEquals(1, dashboard.escalatedEntries().size());
    }

    @Test
    void exposesTypedPairedPortfolioDriftDashboards() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryPortfolioDriftDashboard dashboard =
                DiffusionOpdReportQueries.taskStagePortfolioDriftDashboard(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals(2, dashboard.totalEntries());
        assertEquals(1, dashboard.statusChangedCount());
        assertEquals(1, dashboard.recommendationChangedCount());
        assertEquals(1, dashboard.escalatedCount());
        assertEquals(1, dashboard.healthyToUnhealthyCount());
        assertEquals(0, dashboard.unhealthyToHealthyCount());
        assertEquals(1, dashboard.statusChangedEntries().size());
        assertEquals("ocr,early", dashboard.statusChangedEntries().getFirst().pair());
    }

    @Test
    void exposesTypedGroupedPortfolioExecutiveSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryPortfolioExecutiveSummary summary =
                DiffusionOpdReportQueries.taskPortfolioExecutiveSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(2, summary.deltaDashboard().totalEntries());
        assertEquals(1, summary.driftDashboard().statusChangedCount());
        assertEquals("ocr", summary.primaryRegression().value());
        assertEquals("ocr", summary.primaryEscalation().value());
        assertEquals(2, summary.reviewHotspotCount());
        assertEquals(
                true,
                summary.reviewHotspots().stream().anyMatch(entry -> "ocr".equals(entry.value())));
    }

    @Test
    void exposesTypedPairedPortfolioExecutiveSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryPortfolioExecutiveSummary summary =
                DiffusionOpdReportQueries.taskStagePortfolioExecutiveSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals(2, summary.deltaDashboard().totalEntries());
        assertEquals(1, summary.driftDashboard().statusChangedCount());
        assertEquals("ocr,early", summary.primaryRegression().pair());
        assertEquals("ocr,early", summary.primaryEscalation().pair());
        assertEquals(2, summary.reviewHotspotCount());
        assertEquals(
                true,
                summary.reviewHotspots().stream().anyMatch(entry -> "ocr,early".equals(entry.pair())));
    }

    @Test
    void exposesTypedGroupedPortfolioRemediationSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryPortfolioRemediationSummary summary =
                DiffusionOpdReportQueries.taskPortfolioRemediationSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(2, summary.totalEntries());
        assertEquals(0, summary.manualReviewCount());
        assertEquals(1, summary.automationFriendlyCount());
        assertEquals("monitor", summary.primaryAction());
        assertEquals(2, summary.actionBuckets().size());
        assertEquals(
                true,
                summary.actionBuckets().stream().anyMatch(bucket -> "monitor".equals(bucket.action())));
        assertEquals(
                true,
                summary.actionBuckets().stream().anyMatch(bucket -> "reduce_lr".equals(bucket.action())));
    }

    @Test
    void exposesTypedPairedPortfolioRemediationSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryPortfolioRemediationSummary summary =
                DiffusionOpdReportQueries.taskStagePortfolioRemediationSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5,
                        DiffusionOpdRoundHistoryStatusPolicy.defaults());

        assertEquals(2, summary.totalEntries());
        assertEquals(0, summary.manualReviewCount());
        assertEquals(1, summary.automationFriendlyCount());
        assertEquals("monitor", summary.primaryAction());
        assertEquals(2, summary.actionBuckets().size());
        assertEquals(
                true,
                summary.actionBuckets().stream().anyMatch(bucket -> "monitor".equals(bucket.action())));
        assertEquals(
                true,
                summary.actionBuckets().stream().anyMatch(bucket -> "reduce_lr".equals(bucket.action())));
    }

    @Test
    void exposesTypedGroupedPortfolioPolicyComparisonSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary summary =
                DiffusionOpdReportQueries.taskPortfolioPolicyComparisonSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(3, summary.profiles().size());
        assertEquals("early_stop_review", summary.strictestPrimaryAction());
        assertEquals("monitor", summary.loosestPrimaryAction());
        assertEquals(Boolean.TRUE, summary.primaryActionChangedAcrossPolicies());
        assertEquals(2, summary.maxReviewHotspotCount());
        assertEquals(1, summary.maxManualReviewCount());
    }

    @Test
    void exposesTypedPairedPortfolioPolicyComparisonSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary summary =
                DiffusionOpdReportQueries.taskStagePortfolioPolicyComparisonSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(3, summary.profiles().size());
        assertEquals("early_stop_review", summary.strictestPrimaryAction());
        assertEquals("monitor", summary.loosestPrimaryAction());
        assertEquals(Boolean.TRUE, summary.primaryActionChangedAcrossPolicies());
        assertEquals(2, summary.maxReviewHotspotCount());
        assertEquals(1, summary.maxManualReviewCount());
    }

    @Test
    void exposesTypedGroupedPortfolioPolicyRolloutPlans() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan plan =
                DiffusionOpdReportQueries.taskPortfolioPolicyRolloutPlan(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("default", plan.recommendedProfile().policyName());
        assertEquals("lenient", plan.fallbackProfile().policyName());
        assertEquals(Boolean.TRUE, plan.policyChangedRecommendation());
    }

    @Test
    void exposesTypedPairedPortfolioPolicyRolloutPlans() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan plan =
                DiffusionOpdReportQueries.taskStagePortfolioPolicyRolloutPlan(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("default", plan.recommendedProfile().policyName());
        assertEquals("lenient", plan.fallbackProfile().policyName());
        assertEquals(Boolean.TRUE, plan.policyChangedRecommendation());
    }

    @Test
    void exposesTypedGroupedWorkflowActionPackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryWorkflowActionPacket packet =
                DiffusionOpdReportQueries.taskPortfolioWorkflowActionPacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("default", packet.selectedPolicyName());
        assertEquals("lenient", packet.fallbackPolicyName());
        assertEquals("monitor", packet.primaryAction());
        assertEquals("monitor", packet.fallbackAction());
        assertEquals(Boolean.FALSE, packet.automationFriendly());
        assertEquals(Boolean.FALSE, packet.reviewRequired());
        assertEquals(2, packet.reviewHotspotCount());
        assertEquals(0, packet.manualReviewCount());
    }

    @Test
    void exposesTypedPairedWorkflowActionPackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryWorkflowActionPacket packet =
                DiffusionOpdReportQueries.taskStagePortfolioWorkflowActionPacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("default", packet.selectedPolicyName());
        assertEquals("lenient", packet.fallbackPolicyName());
        assertEquals("monitor", packet.primaryAction());
        assertEquals("monitor", packet.fallbackAction());
        assertEquals(Boolean.FALSE, packet.automationFriendly());
        assertEquals(Boolean.FALSE, packet.reviewRequired());
        assertEquals(2, packet.reviewHotspotCount());
        assertEquals(0, packet.manualReviewCount());
    }

    @Test
    void exposesTypedGroupedWorkflowExecutionPlans() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryWorkflowExecutionPlan plan =
                DiffusionOpdReportQueries.taskPortfolioWorkflowExecutionPlan(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("assisted", plan.executionMode());
        assertEquals(Boolean.TRUE, plan.approvalGate().approvalRequired());
        assertEquals("approval.assisted_execution", plan.approvalGate().gateCode());
        assertEquals("training-monitor", plan.escalationTarget());
        assertEquals(1, plan.cooldownRounds());
        assertEquals(2, plan.nextCheckWindowSize());
        assertEquals("Watch the next short loss window closely.", plan.checklist().getFirst());
    }

    @Test
    void exposesTypedPairedWorkflowExecutionPlans() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryWorkflowExecutionPlan plan =
                DiffusionOpdReportQueries.taskStagePortfolioWorkflowExecutionPlan(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("assisted", plan.executionMode());
        assertEquals(Boolean.TRUE, plan.approvalGate().approvalRequired());
        assertEquals("approval.assisted_execution", plan.approvalGate().gateCode());
        assertEquals("training-monitor", plan.escalationTarget());
        assertEquals(1, plan.cooldownRounds());
        assertEquals(2, plan.nextCheckWindowSize());
        assertEquals("Watch the next short loss window closely.", plan.checklist().getFirst());
    }

    @Test
    void exposesTypedGroupedWorkflowDispatchEnvelopes() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope envelope =
                DiffusionOpdReportQueries.taskPortfolioWorkflowDispatchEnvelope(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("taskId", envelope.targetField());
        assertEquals(List.of("caption"), envelope.targetValues());
        assertEquals("normal", envelope.dispatchPriority());
        assertEquals("deferred_recheck", envelope.retryPolicy());
        assertEquals(1, envelope.recheckAfterRounds());
        assertEquals(2, envelope.deferredRecheckWindowSize());
    }

    @Test
    void exposesTypedPairedWorkflowDispatchEnvelopes() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryWorkflowDispatchEnvelope envelope =
                DiffusionOpdReportQueries.taskStagePortfolioWorkflowDispatchEnvelope(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("taskId", envelope.firstField());
        assertEquals("stageName", envelope.secondField());
        assertEquals(List.of("caption,late"), envelope.targetPairs());
        assertEquals("normal", envelope.dispatchPriority());
        assertEquals("deferred_recheck", envelope.retryPolicy());
        assertEquals(1, envelope.recheckAfterRounds());
        assertEquals(2, envelope.deferredRecheckWindowSize());
    }

    @Test
    void exposesTypedGroupedWorkflowBatchManifests() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdGroupedHistoryWorkflowBatchManifest manifest =
                DiffusionOpdReportQueries.taskPortfolioWorkflowBatchManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(1, manifest.envelopeCount());
        assertEquals(1, manifest.targetSliceCount());
        assertEquals(1, manifest.approvalRequiredCount());
        assertEquals(0, manifest.manualReviewCount());
        assertEquals("normal", manifest.highestDispatchPriority());
        assertEquals(1, manifest.earliestRecheckAfterRounds());
        assertEquals(2, manifest.earliestDeferredRecheckWindowSize());
    }

    @Test
    void exposesTypedPairedWorkflowBatchManifests() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdPairedHistoryWorkflowBatchManifest manifest =
                DiffusionOpdReportQueries.taskStagePortfolioWorkflowBatchManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(1, manifest.envelopeCount());
        assertEquals(1, manifest.targetSliceCount());
        assertEquals(1, manifest.approvalRequiredCount());
        assertEquals(0, manifest.manualReviewCount());
        assertEquals("normal", manifest.highestDispatchPriority());
        assertEquals(1, manifest.earliestRecheckAfterRounds());
        assertEquals(2, manifest.earliestDeferredRecheckWindowSize());
    }

    @Test
    void exposesTypedWorkflowCampaignSummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignSummary summary =
                DiffusionOpdReportQueries.taskCampaignSummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(2, summary.batchCount());
        assertEquals(2, summary.envelopeCount());
        assertEquals(2, summary.targetSliceCount());
        assertEquals(2, summary.approvalRequiredCount());
        assertEquals(0, summary.manualReviewCount());
        assertEquals("normal", summary.highestDispatchPriority());
        assertEquals(1, summary.earliestRecheckAfterRounds());
        assertEquals(2, summary.earliestDeferredRecheckWindowSize());
    }

    @Test
    void exposesTypedWorkflowCampaignRolloutPacks() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignRolloutPack pack =
                DiffusionOpdReportQueries.taskCampaignRolloutPack(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("assisted", pack.recommendedExecutionMode());
        assertEquals("manual", pack.fallbackExecutionMode());
        assertEquals("staged_review", pack.primaryStrategy());
        assertEquals(Boolean.TRUE, pack.approvalRequired());
        assertEquals(2, pack.batchExecutionSteps().size());
        assertEquals("grouped", pack.batchExecutionSteps().getFirst().batchScope());
    }

    @Test
    void exposesTypedWorkflowCampaignApprovalPackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignApprovalPacket packet =
                DiffusionOpdReportQueries.taskCampaignApprovalPacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(Boolean.TRUE, packet.signOffRequired());
        assertEquals("training-monitor", packet.primaryApproverTarget());
        assertEquals("approval.gated_batches", packet.approvalReasonCodes().getFirst());
        assertEquals(2, packet.gatedBatchSteps().size());
        assertEquals(2, packet.blockedBatchCount());
        assertEquals("manual", packet.fallbackExecutionMode());
        assertEquals("grouped", packet.gatedBatchSteps().getFirst().batchScope());
    }

    @Test
    void exposesTypedWorkflowCampaignDecisionLogs() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDecisionLog log =
                DiffusionOpdReportQueries.taskCampaignDecisionLog(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, log.decisionCount());
        assertEquals(2, log.gatedDecisionCount());
        assertEquals("warning", log.highestSeverity());
        assertEquals("rollout", log.entries().getFirst().decisionType());
        assertEquals("campaign", log.entries().getFirst().scope());
    }

    @Test
    void exposesTypedWorkflowCampaignIncidentReports() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignIncidentReport report =
                DiffusionOpdReportQueries.taskCampaignIncidentReport(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(Boolean.TRUE, report.incidentOpen());
        assertEquals("warning", report.highestSeverity());
        assertEquals(2, report.blockedBatchCount());
        assertEquals(List.of("grouped", "paired"), report.blockedBatchScopes());
        assertEquals("training-monitor", report.primaryOperatorTarget());
        assertEquals("review_and_sign_off", report.recommendedOperatorAction());
    }

    @Test
    void exposesTypedWorkflowCampaignResolutionPackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignResolutionPacket packet =
                DiffusionOpdReportQueries.taskCampaignResolutionPacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("deferred", packet.suggestedOutcome());
        assertEquals("rejected", packet.fallbackOutcome());
        assertEquals(Boolean.TRUE, packet.operatorResponseRequired());
        assertEquals(2, packet.batchResolutions().size());
        assertEquals("grouped", packet.batchResolutions().getFirst().batchScope());
        assertEquals("deferred", packet.batchResolutions().getFirst().recommendedOutcome());
    }

    @Test
    void exposesTypedWorkflowCampaignClosureReports() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignClosureReport report =
                DiffusionOpdReportQueries.taskCampaignClosureReport(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("pending_operator_response", report.finalOutcome());
        assertEquals(Boolean.FALSE, report.closed());
        assertEquals(2, report.unresolvedOverrideCount());
        assertEquals("await_operator_confirmation", report.followUpAction());
    }

    @Test
    void exposesTypedWorkflowCampaignHandoffBundles() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignHandoffBundle bundle =
                DiffusionOpdReportQueries.taskCampaignHandoffBundle(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("training-monitor", bundle.handoffTarget());
        assertEquals("pending_review", bundle.exportStatus());
        assertEquals("warning", bundle.incidentReport().highestSeverity());
        assertEquals("pending_operator_response", bundle.closureReport().finalOutcome());
    }

    @Test
    void exposesTypedWorkflowCampaignExportManifests() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignExportManifest manifest =
                DiffusionOpdReportQueries.taskCampaignExportManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, manifest.recordCount());
        assertEquals("training-monitor", manifest.primaryDestination());
        assertEquals("pending_review", manifest.overallDeliveryStatus());
        assertEquals("decision_log", manifest.records().getFirst().artifactType());
        assertEquals("closure_report", manifest.records().getLast().artifactType());
    }

    @Test
    void exposesTypedWorkflowCampaignDeliveryReports() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDeliveryReport report =
                DiffusionOpdReportQueries.taskCampaignDeliveryReport(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(1, report.destinationCount());
        assertEquals(4, report.totalArtifacts());
        assertEquals(4, report.pendingArtifacts());
        assertEquals(0, report.blockedArtifacts());
        assertEquals(true, report.followUpRequired());
        assertEquals("training-monitor", report.primaryDestination());
        assertEquals("pending_review", report.overallDeliveryStatus());
        assertEquals(4, report.destinations().getFirst().pendingCount());
        assertEquals(true, report.destinations().getFirst().followUpRequired());
    }

    @Test
    void exposesTypedWorkflowCampaignDeliveryLedgers() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDeliveryLedger ledger =
                DiffusionOpdReportQueries.taskCampaignDeliveryLedger(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, ledger.entryCount());
        assertEquals(0, ledger.acknowledgedCount());
        assertEquals(4, ledger.pendingAcknowledgementCount());
        assertEquals(0, ledger.retryRequiredCount());
        assertEquals(true, ledger.followUpRequired());
        assertEquals("training-monitor", ledger.primaryDestination());
        assertEquals("pending_review", ledger.overallDeliveryStatus());
        assertEquals("pending_ack", ledger.entries().getFirst().acknowledgementStatus());
        assertEquals("await_review", ledger.entries().getFirst().retryPolicy());
    }

    @Test
    void exposesTypedWorkflowCampaignDeliveryReceipts() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDeliveryReceipts receipts =
                DiffusionOpdReportQueries.taskCampaignDeliveryReceipts(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, receipts.receiptCount());
        assertEquals(0, receipts.confirmedCount());
        assertEquals(4, receipts.pendingCount());
        assertEquals(0, receipts.operatorActionRequiredCount());
        assertEquals(true, receipts.followUpRequired());
        assertEquals("training-monitor", receipts.primaryDestination());
        assertEquals("pending_review", receipts.overallDeliveryStatus());
        assertEquals("pending_confirmation", receipts.receipts().getFirst().receiptStatus());
        assertEquals("awaiting_operator_confirmation", receipts.receipts().getFirst().operatorAcknowledgement());
    }

    @Test
    void exposesTypedWorkflowCampaignReceiptAcknowledgements() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReceiptAcknowledgements acknowledgements =
                DiffusionOpdReportQueries.taskCampaignReceiptAcknowledgements(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, acknowledgements.acknowledgementCount());
        assertEquals(0, acknowledgements.acceptedCount());
        assertEquals(4, acknowledgements.pendingCount());
        assertEquals(0, acknowledgements.rejectedCount());
        assertEquals(true, acknowledgements.followUpRequired());
        assertEquals("training-monitor", acknowledgements.primaryDestination());
        assertEquals("pending_review", acknowledgements.overallDeliveryStatus());
        assertEquals("pending_review", acknowledgements.acknowledgements().getFirst().acknowledgementOutcome());
        assertEquals("delivery-bot", acknowledgements.acknowledgements().getFirst().reviewer());
    }

    @Test
    void exposesTypedWorkflowCampaignAcknowledgementDecisions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignAcknowledgementDecisions decisions =
                DiffusionOpdReportQueries.taskCampaignAcknowledgementDecisions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, decisions.decisionCount());
        assertEquals(0, decisions.approvedCount());
        assertEquals(4, decisions.deferredCount());
        assertEquals(0, decisions.rejectedCount());
        assertEquals(true, decisions.followUpRequired());
        assertEquals("training-monitor", decisions.primaryDestination());
        assertEquals("pending_review", decisions.overallDeliveryStatus());
        assertEquals("deferred", decisions.decisions().getFirst().decision());
        assertEquals("review_queue", decisions.decisions().getFirst().resolutionRoute());
    }

    @Test
    void exposesTypedCampaignDecisionResolutions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDecisionResolutions resolutions =
                DiffusionOpdReportQueries.taskCampaignDecisionResolutions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, resolutions.resolutionCount());
        assertEquals(0, resolutions.executableCount());
        assertEquals(4, resolutions.blockedCount());
        assertEquals(4, resolutions.escalationRequiredCount());
        assertEquals(true, resolutions.followUpRequired());
        assertEquals("training-monitor", resolutions.primaryDestination());
        assertEquals("pending_review", resolutions.overallDeliveryStatus());
        assertEquals("awaiting_review", resolutions.resolutions().getFirst().resolution());
        assertEquals("blocked", resolutions.resolutions().getFirst().executionReadiness());
    }

    @Test
    void exposesTypedCampaignDispatchEligibilitySummaries() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDispatchEligibilitySummary summary =
                DiffusionOpdReportQueries.taskCampaignDispatchEligibilitySummary(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, summary.entryCount());
        assertEquals(0, summary.dispatchableCount());
        assertEquals(4, summary.blockedCount());
        assertEquals(4, summary.escalationRequiredCount());
        assertEquals(false, summary.anyDispatchable());
        assertEquals("training-monitor", summary.primaryDestination());
        assertEquals("pending_review", summary.overallDeliveryStatus());
        assertEquals("blocked", summary.entries().getFirst().eligibility());
        assertEquals(false, summary.entries().getFirst().dispatchable());
    }

    @Test
    void exposesTypedCampaignDispatchPlans() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignDispatchPlan plan =
                DiffusionOpdReportQueries.taskCampaignDispatchPlan(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(0, plan.stepCount());
        assertEquals(4, plan.blockedCount());
        assertEquals(4, plan.escalationRequiredCount());
        assertEquals(false, plan.executable());
        assertEquals("training-monitor", plan.primaryDestination());
        assertEquals("pending_review", plan.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignExecutionPackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignExecutionPacket packet =
                DiffusionOpdReportQueries.taskCampaignExecutionPacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(0, packet.runnableStepCount());
        assertEquals(4, packet.blockedEntryCount());
        assertEquals(4, packet.escalationRequiredCount());
        assertEquals(false, packet.executable());
        assertEquals("training-monitor", packet.primaryDestination());
        assertEquals("pending_review", packet.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignEngineHandoffEnvelopes() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignEngineHandoffEnvelope envelope =
                DiffusionOpdReportQueries.taskCampaignEngineHandoffEnvelope(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("training-monitor", envelope.engineId());
        assertEquals("review_queue", envelope.submissionMode());
        assertEquals("await_review", envelope.retryPolicy());
        assertEquals(true, envelope.requiresHumanReview());
        assertEquals(false, envelope.readyForDispatch());
        assertEquals(0, envelope.runnableStepCount());
        assertEquals(4, envelope.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignExecutionManifests() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignExecutionManifest manifest =
                DiffusionOpdReportQueries.taskCampaignExecutionManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("campaign-review", manifest.queueName());
        assertEquals("serial", manifest.batchMode());
        assertEquals("after_review", manifest.retryWindow());
        assertEquals(100, manifest.queueOrder());
        assertEquals(true, manifest.reviewRequired());
        assertEquals(false, manifest.dispatchReady());
        assertEquals(0, manifest.runnableStepCount());
        assertEquals(4, manifest.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignExecutionReviews() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignExecutionReview review =
                DiffusionOpdReportQueries.taskCampaignExecutionReview(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("training-monitor", review.reviewerTarget());
        assertEquals("pending_review", review.reviewOutcome());
        assertEquals(true, review.approvalRequired());
        assertEquals(false, review.releaseReady());
        assertEquals(0, review.runnableStepCount());
        assertEquals(4, review.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseDecisions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDecision decision =
                DiffusionOpdReportQueries.taskCampaignReleaseDecision(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("deferred", decision.releaseOutcome());
        assertEquals("blocked", decision.fallbackOutcome());
        assertEquals(false, decision.dispatchAllowed());
        assertEquals(true, decision.operatorFollowUpRequired());
        assertEquals(0, decision.runnableStepCount());
        assertEquals(4, decision.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleasePackets() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleasePacket packet =
                DiffusionOpdReportQueries.taskCampaignReleasePacket(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("training-monitor", packet.dispatchTarget());
        assertEquals("review_handoff", packet.packetOutcome());
        assertEquals("blocked_handoff", packet.fallbackPacketOutcome());
        assertEquals(false, packet.readyForEngineHandoff());
        assertEquals(true, packet.reviewEscalationOpen());
        assertEquals(0, packet.runnableStepCount());
        assertEquals(4, packet.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseLedgers() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseLedger ledger =
                DiffusionOpdReportQueries.taskCampaignReleaseLedger(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("pending_review", ledger.releaseStatus());
        assertEquals("awaiting_ack", ledger.acknowledgementStatus());
        assertEquals(0, ledger.attemptCount());
        assertEquals(false, ledger.handoffComplete());
        assertEquals(true, ledger.followUpRequired());
        assertEquals(0, ledger.runnableStepCount());
        assertEquals(4, ledger.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseReceipts() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseReceipt receipt =
                DiffusionOpdReportQueries.taskCampaignReleaseReceipt(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("pending_confirmation", receipt.receiptStatus());
        assertEquals("awaiting_operator_confirmation", receipt.finalAcknowledgement());
        assertEquals(false, receipt.confirmed());
        assertEquals(true, receipt.followUpRequired());
        assertEquals(0, receipt.runnableStepCount());
        assertEquals(4, receipt.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseAcknowledgements() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseAcknowledgement acknowledgement =
                DiffusionOpdReportQueries.taskCampaignReleaseAcknowledgement(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("pending_review", acknowledgement.acknowledgementOutcome());
        assertEquals("release-bot", acknowledgement.reviewer());
        assertEquals("operator-gate", acknowledgement.reviewerRole());
        assertEquals(false, acknowledgement.finalAcceptance());
        assertEquals(true, acknowledgement.followUpRequired());
        assertEquals(0, acknowledgement.runnableStepCount());
        assertEquals(4, acknowledgement.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseResolutions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseResolution resolution =
                DiffusionOpdReportQueries.taskCampaignReleaseResolution(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("deferred", resolution.resolution());
        assertEquals("training-monitor", resolution.escalationTarget());
        assertEquals(false, resolution.finalAccepted());
        assertEquals(false, resolution.dispatchEligible());
        assertEquals(0, resolution.runnableStepCount());
        assertEquals(4, resolution.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseClosures() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseClosure closure =
                DiffusionOpdReportQueries.taskCampaignReleaseClosure(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("pending_follow_up", closure.finalOutcome());
        assertEquals(false, closure.closed());
        assertEquals("await_manual_review", closure.followUpAction());
        assertEquals(false, closure.dispatchComplete());
        assertEquals(0, closure.runnableStepCount());
        assertEquals(4, closure.blockedEntryCount());
    }

    @Test
    void exposesTypedCampaignReleaseHandoffBundles() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseHandoffBundle bundle =
                DiffusionOpdReportQueries.taskCampaignReleaseHandoffBundle(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals("training-monitor", bundle.handoffTarget());
        assertEquals("pending_follow_up", bundle.exportStatus());
        assertEquals(false, bundle.terminal());
        assertEquals(0, bundle.runnableStepCount());
        assertEquals(4, bundle.blockedEntryCount());
        assertEquals("pending_follow_up", bundle.releaseClosure().finalOutcome());
    }

    @Test
    void exposesTypedCampaignReleaseExportManifests() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseExportManifest manifest =
                DiffusionOpdReportQueries.taskCampaignReleaseExportManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, manifest.recordCount());
        assertEquals("training-monitor", manifest.primaryDestination());
        assertEquals("pending_follow_up", manifest.overallDeliveryStatus());
        assertEquals("release_receipt", manifest.records().getFirst().artifactType());
        assertEquals("release_closure", manifest.records().getLast().artifactType());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryReports() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryReport report =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryReport(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(1, report.destinationCount());
        assertEquals(4, report.totalArtifacts());
        assertEquals(4, report.pendingArtifacts());
        assertEquals(4, report.blockedArtifacts());
        assertEquals(true, report.followUpRequired());
        assertEquals("training-monitor", report.primaryDestination());
        assertEquals("pending_follow_up", report.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryLedgers() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryLedger ledger =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryLedger(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, ledger.entryCount());
        assertEquals(0, ledger.acknowledgedCount());
        assertEquals(4, ledger.pendingAcknowledgementCount());
        assertEquals(4, ledger.retryRequiredCount());
        assertEquals(true, ledger.followUpRequired());
        assertEquals("training-monitor", ledger.primaryDestination());
        assertEquals("pending_follow_up", ledger.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryReceipts() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryReceipts receipts =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryReceipts(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, receipts.receiptCount());
        assertEquals(0, receipts.confirmedCount());
        assertEquals(4, receipts.pendingCount());
        assertEquals(4, receipts.operatorActionRequiredCount());
        assertEquals(true, receipts.followUpRequired());
        assertEquals("training-monitor", receipts.primaryDestination());
        assertEquals("pending_follow_up", receipts.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryAcknowledgements() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryAcknowledgements acknowledgements =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryAcknowledgements(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, acknowledgements.acknowledgementCount());
        assertEquals(0, acknowledgements.acceptedCount());
        assertEquals(4, acknowledgements.pendingCount());
        assertEquals(0, acknowledgements.rejectedCount());
        assertEquals(true, acknowledgements.followUpRequired());
        assertEquals("training-monitor", acknowledgements.primaryDestination());
        assertEquals("pending_follow_up", acknowledgements.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryDecisions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryDecisions decisions =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryDecisions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, decisions.decisionCount());
        assertEquals(0, decisions.acceptedCount());
        assertEquals(4, decisions.deferredCount());
        assertEquals(0, decisions.rejectedCount());
        assertEquals(true, decisions.followUpRequired());
        assertEquals("training-monitor", decisions.primaryDestination());
        assertEquals("pending_follow_up", decisions.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryResolutions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryResolutions resolutions =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryResolutions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, resolutions.resolutionCount());
        assertEquals(0, resolutions.acceptedCount());
        assertEquals(4, resolutions.deferredCount());
        assertEquals(0, resolutions.rejectedCount());
        assertEquals(true, resolutions.followUpRequired());
        assertEquals("training-monitor", resolutions.primaryDestination());
        assertEquals("pending_follow_up", resolutions.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryClosures() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryClosures closures =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryClosures(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, closures.closureCount());
        assertEquals(0, closures.closedCount());
        assertEquals(4, closures.openCount());
        assertEquals(true, closures.followUpRequired());
        assertEquals("training-monitor", closures.primaryDestination());
        assertEquals("pending_follow_up", closures.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryHandoffBundles() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        List<DiffusionOpdCampaignReleaseDeliveryHandoffBundle> bundles =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryHandoffBundles(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, bundles.size());
        assertEquals("training-monitor", bundles.getFirst().handoffTarget());
        assertEquals("pending_follow_up", bundles.getFirst().exportStatus());
        assertEquals(false, bundles.getFirst().terminal());
        assertEquals("release_receipt", bundles.getFirst().receipt().artifactType());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportManifest() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportManifest manifest =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportManifest(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, manifest.recordCount());
        assertEquals(4, manifest.handoffBundles().size());
        assertEquals("training-monitor", manifest.handoffBundles().getFirst().handoffTarget());
        assertEquals("training-monitor", manifest.primaryDestination());
        assertEquals("pending_follow_up", manifest.overallDeliveryStatus());
        assertEquals("release_receipt", manifest.records().getFirst().artifactType());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportReport() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportReport report =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportReport(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(1, report.destinationCount());
        assertEquals(4, report.totalArtifacts());
        assertEquals(4, report.pendingArtifacts());
        assertEquals(4, report.blockedArtifacts());
        assertEquals(true, report.followUpRequired());
        assertEquals("training-monitor", report.primaryDestination());
        assertEquals("pending_follow_up", report.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportLedger() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportLedger ledger =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportLedger(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, ledger.entryCount());
        assertEquals(0, ledger.acknowledgedCount());
        assertEquals(4, ledger.pendingAcknowledgementCount());
        assertEquals(4, ledger.retryRequiredCount());
        assertEquals(true, ledger.followUpRequired());
        assertEquals("training-monitor", ledger.primaryDestination());
        assertEquals("pending_follow_up", ledger.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportReceipts() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportReceipts receipts =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportReceipts(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, receipts.receiptCount());
        assertEquals(0, receipts.confirmedCount());
        assertEquals(4, receipts.pendingCount());
        assertEquals(4, receipts.operatorActionRequiredCount());
        assertEquals(true, receipts.followUpRequired());
        assertEquals("training-monitor", receipts.primaryDestination());
        assertEquals("pending_follow_up", receipts.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportAcknowledgements() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements acknowledgements =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportAcknowledgements(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, acknowledgements.acknowledgementCount());
        assertEquals(0, acknowledgements.acceptedCount());
        assertEquals(4, acknowledgements.pendingCount());
        assertEquals(0, acknowledgements.rejectedCount());
        assertEquals(true, acknowledgements.followUpRequired());
        assertEquals("training-monitor", acknowledgements.primaryDestination());
        assertEquals("pending_follow_up", acknowledgements.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportDecisions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportDecisions decisions =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportDecisions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, decisions.decisionCount());
        assertEquals(0, decisions.acceptedCount());
        assertEquals(4, decisions.deferredCount());
        assertEquals(0, decisions.rejectedCount());
        assertEquals(true, decisions.followUpRequired());
        assertEquals("training-monitor", decisions.primaryDestination());
        assertEquals("pending_follow_up", decisions.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportResolutions() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportResolutions resolutions =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportResolutions(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, resolutions.resolutionCount());
        assertEquals(0, resolutions.acceptedCount());
        assertEquals(4, resolutions.deferredCount());
        assertEquals(0, resolutions.rejectedCount());
        assertEquals(true, resolutions.followUpRequired());
        assertEquals("training-monitor", resolutions.primaryDestination());
        assertEquals("pending_follow_up", resolutions.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportClosures() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        DiffusionOpdCampaignReleaseDeliveryExportClosures closures =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportClosures(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, closures.closureCount());
        assertEquals(0, closures.closedCount());
        assertEquals(4, closures.openCount());
        assertEquals(true, closures.followUpRequired());
        assertEquals("training-monitor", closures.primaryDestination());
        assertEquals("pending_follow_up", closures.overallDeliveryStatus());
    }

    @Test
    void exposesTypedCampaignReleaseDeliveryExportHandoffBundles() {
        DiffusionOpdReport baseline = sampleReport();
        DiffusionOpdReport regressed = regressedReport();

        List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> bundles =
                DiffusionOpdReportQueries.taskCampaignReleaseDeliveryExportHandoffBundles(
                        baseline,
                        regressed,
                        1,
                        2,
                        3,
                        5);

        assertEquals(4, bundles.size());
        assertEquals("training-monitor", bundles.getFirst().handoffTarget());
        assertEquals("pending_follow_up", bundles.getFirst().exportStatus());
        assertEquals(false, bundles.getFirst().terminal());
        assertEquals("release_receipt", bundles.getFirst().receipt().artifactType());
    }

    @Test
    void exposesTypedGroupedFilters() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedHistorySummary> teachersForOcr =
                DiffusionOpdReportQueries.teacherSummariesForTask(report, "ocr");

        assertEquals(2, teachersForOcr.size());
        assertEquals("teacherKey", teachersForOcr.getFirst().field());
        assertEquals("ocr-base", teachersForOcr.getFirst().value());
        assertEquals(1, teachersForOcr.getFirst().summary().count());
    }

    @Test
    void exposesTypedPairedFilters() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedHistorySummary> taskTeacherForEarly =
                DiffusionOpdReportQueries.taskTeacherSummariesForStage(report, "early");

        assertEquals(2, taskTeacherForEarly.size());
        assertEquals("ocr,ocr-base", taskTeacherForEarly.getFirst().pair());
        assertEquals("ocr,ocr-early", taskTeacherForEarly.getLast().pair());
    }

    @Test
    void exposesTypedFilteredGroupedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTeacherSummariesForTaskByMeanLoss(report, "ocr", 1);

        assertEquals(1, leaderboard.size());
        assertEquals("ocr-base", leaderboard.getFirst().value());
        assertEquals(0.60d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedFilteredPairedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTaskTeacherSummariesForStageByMeanLoss(report, "early", 1);

        assertEquals(1, leaderboard.size());
        assertEquals("ocr,ocr-base", leaderboard.getFirst().pair());
        assertEquals(0.60d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedTopSingleSelections() {
        DiffusionOpdReport report = sampleReport();

        DiffusionOpdGroupedHistorySummary topTeacherForOcr =
                DiffusionOpdReportQueries.topTeacherSummaryForTaskByMeanLoss(report, "ocr");
        DiffusionOpdPairedHistorySummary topPairForEarly =
                DiffusionOpdReportQueries.topTaskTeacherSummaryForStageByMeanLoss(report, "early");

        assertEquals("ocr-base", topTeacherForOcr.value());
        assertEquals(0.60d, topTeacherForOcr.summary().meanLoss());
        assertEquals("ocr,ocr-base", topPairForEarly.pair());
        assertEquals(0.60d, topPairForEarly.summary().meanLoss());
    }

    private static DiffusionOpdReport sampleReport() {
        return new DiffusionOpdReport(
                new DiffusionOpdRunReport(1, 0.42d, 1234L, "ODE", 2L, 3L, 3L, false),
                new DiffusionOpdArtifactsReport("summary.json", "history.csv", "report.json", "checkpoints"),
                Map.of("ocr-base", Map.of("stage", "early")),
                Map.of("early", Map.of("weight", 1.0d)),
                Map.of("mode", "teacher-specialized", "taskCount", 2),
                Map.of("laneCount", 2, "fixtures", Map.of("ocr", "clip-ocr")),
                Map.of(),
                Map.of(),
                List.of(
                        Map.of(
                                "round", 1L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-base",
                                "stageName", "early",
                                "averageLoss", 0.60d),
                        Map.of(
                                "round", 2L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-early",
                                "stageName", "early",
                                "averageLoss", 0.50d),
                        Map.of(
                                "round", 3L,
                                "taskId", "caption",
                                "teacherKey", "caption-main",
                                "stageName", "late",
                                "averageLoss", 0.90d)));
    }

    private static DiffusionOpdReport improvedReport() {
        return new DiffusionOpdReport(
                new DiffusionOpdRunReport(1, 0.35d, 1200L, "ODE", 2L, 3L, 3L, false),
                new DiffusionOpdArtifactsReport("summary.json", "history.csv", "report.json", "checkpoints"),
                Map.of("ocr-base", Map.of("stage", "early")),
                Map.of("early", Map.of("weight", 1.0d)),
                Map.of("mode", "teacher-specialized", "taskCount", 2),
                Map.of("laneCount", 2, "fixtures", Map.of("ocr", "clip-ocr")),
                Map.of(),
                Map.of(),
                List.of(
                        Map.of(
                                "round", 1L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-base",
                                "stageName", "early",
                                "averageLoss", 0.55d),
                        Map.of(
                                "round", 2L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-early",
                                "stageName", "early",
                                "averageLoss", 0.40d),
                        Map.of(
                                "round", 3L,
                                "taskId", "caption",
                                "teacherKey", "caption-main",
                                "stageName", "late",
                                "averageLoss", 0.80d)));
    }

    private static DiffusionOpdReport regressedReport() {
        return new DiffusionOpdReport(
                new DiffusionOpdRunReport(1, 0.58d, 1300L, "ODE", 2L, 3L, 3L, false),
                new DiffusionOpdArtifactsReport("summary.json", "history.csv", "report.json", "checkpoints"),
                Map.of("ocr-base", Map.of("stage", "early")),
                Map.of("early", Map.of("weight", 1.0d)),
                Map.of("mode", "teacher-specialized", "taskCount", 2),
                Map.of("laneCount", 2, "fixtures", Map.of("ocr", "clip-ocr")),
                Map.of(),
                Map.of(),
                List.of(
                        Map.of(
                                "round", 1L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-base",
                                "stageName", "early",
                                "averageLoss", 0.65d),
                        Map.of(
                                "round", 2L,
                                "taskId", "ocr",
                                "teacherKey", "ocr-early",
                                "stageName", "early",
                                "averageLoss", 0.80d),
                        Map.of(
                                "round", 3L,
                                "taskId", "caption",
                                "teacherKey", "caption-main",
                                "stageName", "late",
                                "averageLoss", 0.90d)));
    }
}
