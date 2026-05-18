package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;

class DiffusionOpdReportsTest {

    @Test
    void supportsCompactRoundHistoryAggregates() {
        DiffusionOpdReport report = sampleReport();

        assertEquals(0.55d, DiffusionOpdReports.select(report, "roundHistory:task=ocr:meanLoss"));
        assertEquals(0.55d, DiffusionOpdReports.select(report, "roundHistory:task=ocr:avgLoss"));
        assertEquals(0.60d, DiffusionOpdReports.select(report, "roundHistory:task=ocr:maxLoss"));
        assertEquals(0.50d, DiffusionOpdReports.select(report, "roundHistory:task=ocr:minLoss"));
        assertEquals(0.50d, DiffusionOpdReports.select(report, "roundHistory:teacher=ocr-early:lastLoss"));
        assertEquals(2L, DiffusionOpdReports.select(report, "roundHistory:teacher=ocr-early:lastRound"));
    }

    @Test
    void supportsRoundHistorySummaryProjection() {
        DiffusionOpdReport report = sampleReport();

        Object value = DiffusionOpdReports.select(report, "roundHistory:task=ocr:summary");
        Map<?, ?> summary = assertInstanceOf(Map.class, value);

        assertEquals(2, summary.get("count"));
        assertEquals(0.55d, summary.get("meanLoss"));
        Map<?, ?> last = assertInstanceOf(Map.class, summary.get("last"));
        assertEquals(2L, last.get("round"));
        assertEquals("ocr-early", last.get("teacherKey"));
    }

    private static DiffusionOpdReport sampleReport() {
        return new DiffusionOpdReport(
                new DiffusionOpdRunReport(1, 0.42d, 1234L, "ODE", 2L, 3L, 3L, false),
                new DiffusionOpdArtifactsReport("summary.json", "history.csv", "report.json", "checkpoints"),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
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
