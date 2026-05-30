package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookStatsRendererTest {

    @Test
    void describePreviewFormatsNumericSummaryAndEscapesHtml() {
        NotebookPreview preview = NotebookStatsRenderer.describePreview(
                "/tmp/<stats>.csv",
                "CSV",
                4,
                2,
                List.of(new NotebookStats.NumericSummary("<score>", 3, 1, 2.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
        );

        assertTrue(preview.plain().contains("Describe(/tmp/<stats>.csv, format=CSV, rows=4, columns=2, numericColumns=1)"));
        assertTrue(preview.plain().contains("<score> | 3 | 1 | 2 | 0.5000 | 1 | 1.5000 | 2 | 2.5000 | 3"));
        assertTrue(preview.html().contains("/tmp/&lt;stats&gt;.csv"));
        assertTrue(preview.html().contains("&lt;score&gt;"));
        assertFalse(preview.html().contains("<score>"));
    }

    @Test
    void correlationPreviewCanRenderPlainTableWithoutHeatmap() {
        List<NotebookStats.NumericColumn> columns = List.of(
                new NotebookStats.NumericColumn("x", 0),
                new NotebookStats.NumericColumn("y", 1)
        );
        NotebookPreview preview = NotebookStatsRenderer.correlationPreview(
                "/tmp/corr.csv",
                "CSV",
                3,
                columns,
                new Double[][]{
                        {1.0, -0.5},
                        {-0.5, 1.0}
                },
                false
        );

        assertTrue(preview.plain().contains("Correlation(/tmp/corr.csv, format=CSV, rows=3, numericColumns=2, heatmap=false)"));
        assertTrue(preview.plain().contains("column | x | y"));
        assertTrue(preview.plain().contains("x | 1 | -0.5000"));
        assertTrue(preview.html().contains("<b>Correlation</b>"));
        assertFalse(preview.html().contains("Correlation heatmap"));
    }

    @Test
    void correlationPreviewCanRenderHeatmapAndEscapesColumns() {
        List<NotebookStats.NumericColumn> columns = List.of(
                new NotebookStats.NumericColumn("<left>", 0),
                new NotebookStats.NumericColumn("right", 1)
        );
        NotebookPreview preview = NotebookStatsRenderer.correlationPreview(
                "/tmp/<corr>.csv",
                "CSV",
                3,
                columns,
                new Double[][]{
                        {1.0, null},
                        {null, 1.0}
                },
                true
        );

        assertTrue(preview.plain().contains("Correlation(/tmp/<corr>.csv, format=CSV, rows=3, numericColumns=2, heatmap=true)"));
        assertTrue(preview.plain().contains("<left> | 1 | "));
        assertTrue(preview.html().contains("/tmp/&lt;corr&gt;.csv"));
        assertTrue(preview.html().contains("&lt;left&gt;"));
        assertTrue(preview.html().contains("Correlation heatmap"));
    }
}
