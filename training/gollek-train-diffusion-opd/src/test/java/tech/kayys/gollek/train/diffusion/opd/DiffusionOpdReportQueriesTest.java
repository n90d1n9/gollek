package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryRow;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSectionView;

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

        List<DiffusionOpdGroupedRoundHistorySummary> tasks =
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

        List<DiffusionOpdPairedRoundHistorySummary> pairs =
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

        List<DiffusionOpdGroupedRoundHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTaskSummariesByMeanLoss(report, 1);

        assertEquals(1, leaderboard.size());
        assertEquals("caption", leaderboard.getFirst().value());
        assertEquals(0.90d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedPairedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedRoundHistorySummary> leaderboard =
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
    void exposesTypedGroupedFilters() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedRoundHistorySummary> teachersForOcr =
                DiffusionOpdReportQueries.teacherSummariesForTask(report, "ocr");

        assertEquals(2, teachersForOcr.size());
        assertEquals("teacherKey", teachersForOcr.getFirst().field());
        assertEquals("ocr-base", teachersForOcr.getFirst().value());
        assertEquals(1, teachersForOcr.getFirst().summary().count());
    }

    @Test
    void exposesTypedPairedFilters() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedRoundHistorySummary> taskTeacherForEarly =
                DiffusionOpdReportQueries.taskTeacherSummariesForStage(report, "early");

        assertEquals(2, taskTeacherForEarly.size());
        assertEquals("ocr,ocr-base", taskTeacherForEarly.getFirst().pair());
        assertEquals("ocr,ocr-early", taskTeacherForEarly.getLast().pair());
    }

    @Test
    void exposesTypedFilteredGroupedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdGroupedRoundHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTeacherSummariesForTaskByMeanLoss(report, "ocr", 1);

        assertEquals(1, leaderboard.size());
        assertEquals("ocr-base", leaderboard.getFirst().value());
        assertEquals(0.60d, leaderboard.getFirst().summary().meanLoss());
    }

    @Test
    void exposesTypedFilteredPairedLeaderboards() {
        DiffusionOpdReport report = sampleReport();

        List<DiffusionOpdPairedRoundHistorySummary> leaderboard =
                DiffusionOpdReportQueries.topTaskTeacherSummariesForStageByMeanLoss(report, "early", 1);

        assertEquals(1, leaderboard.size());
        assertEquals("ocr,ocr-base", leaderboard.getFirst().pair());
        assertEquals(0.60d, leaderboard.getFirst().summary().meanLoss());
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
}
