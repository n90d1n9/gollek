package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertTrue;

import tech.kayys.gollek.jupyter.NotebookCharts.HistogramBin;
import tech.kayys.gollek.jupyter.NotebookCharts.LinePlotPoint;
import tech.kayys.gollek.jupyter.NotebookCharts.ScatterPoint;

import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class NotebookChartRendererTest {

    @Test
    void linePlotPreviewIncludesTruncationAndEscapedHtml() {
        List<LinePlotPoint> points = IntStream.rangeClosed(1, 21)
                .mapToObj(i -> new LinePlotPoint("<e" + i + ">", i / 10.0))
                .toList();

        NotebookPreview preview = NotebookChartRenderer.linePlotPreview(
                "/tmp/<loss>.csv",
                "CSV",
                "<epoch>",
                "loss",
                points,
                2
        );

        assertTrue(preview.plain().contains("LinePlot(/tmp/<loss>.csv, format=CSV, x=<epoch>, y=loss, points=21, skipped=2)"));
        assertTrue(preview.plain().contains("<e1> | 0.1000"));
        assertTrue(preview.plain().contains("... 1 more points"));
        assertTrue(preview.html().contains("/tmp/&lt;loss&gt;.csv"));
        assertTrue(preview.html().contains("x=&lt;epoch&gt;"));
        assertTrue(preview.html().contains("<svg"));
    }

    @Test
    void scatterPlotPreviewFormatsPointsAndEscapesHtml() {
        NotebookPreview preview = NotebookChartRenderer.scatterPlotPreview(
                "/tmp/<scatter>.csv",
                "CSV",
                "<weight>",
                "height",
                List.of(new ScatterPoint(1.25, 2.5), new ScatterPoint(3.0, 4.0)),
                1
        );

        assertTrue(preview.plain().contains("ScatterPlot(/tmp/<scatter>.csv, format=CSV, x=<weight>, y=height, points=2, skipped=1)"));
        assertTrue(preview.plain().contains("1.2500 | 2.5000"));
        assertTrue(preview.html().contains("/tmp/&lt;scatter&gt;.csv"));
        assertTrue(preview.html().contains("x=&lt;weight&gt;"));
        assertTrue(preview.html().contains("<b>Scatter Plot</b>"));
    }

    @Test
    void histogramPreviewIncludesBinsAndEscapedSvgTitle() {
        NotebookPreview preview = NotebookChartRenderer.histogramPreview(
                "/tmp/<hist>.csv",
                "CSV",
                "<loss>",
                5,
                1,
                List.of(new HistogramBin(0.1, 0.3, 2), new HistogramBin(0.3, 0.5, 3)),
                0.1,
                0.5
        );

        assertTrue(preview.plain().contains("Histogram(/tmp/<hist>.csv, format=CSV, column=<loss>, values=5, skipped=1, bins=2, min=0.1000, max=0.5000)"));
        assertTrue(preview.plain().contains("[0.1000, 0.3000) | 2"));
        assertTrue(preview.html().contains("/tmp/&lt;hist&gt;.csv"));
        assertTrue(preview.html().contains("column=&lt;loss&gt;"));
        assertTrue(preview.html().contains("Histogram: &lt;loss&gt;"));
    }
}
