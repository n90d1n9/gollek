package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookCharts.buildHistogramSvg;
import static tech.kayys.gollek.jupyter.NotebookCharts.buildLinePlotSvg;
import static tech.kayys.gollek.jupyter.NotebookCharts.buildScatterPlotSvg;
import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;

import tech.kayys.gollek.jupyter.NotebookCharts.HistogramBin;
import tech.kayys.gollek.jupyter.NotebookCharts.LinePlotPoint;
import tech.kayys.gollek.jupyter.NotebookCharts.ScatterPoint;

import java.util.List;
import java.util.stream.Collectors;

final class NotebookChartRenderer {

    private NotebookChartRenderer() {
    }

    static NotebookPreview linePlotPreview(
            String target,
            String label,
            String xColumn,
            String yColumn,
            List<LinePlotPoint> points,
            int skipped
    ) {
        String title = yColumn + " by " + xColumn;
        String plain = "LinePlot(" + target
                + ", format=" + label
                + ", x=" + xColumn
                + ", y=" + yColumn
                + ", points=" + points.size()
                + ", skipped=" + skipped
                + ")\n"
                + points.stream()
                .limit(20)
                .map(point -> point.xLabel() + " | " + formatNullableStat(point.y()))
                .collect(Collectors.joining("\n"))
                + (points.size() > 20 ? "\n... " + (points.size() - 20) + " more points" : "");
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Line Plot</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>format=" + label
                + " x=" + escapeHtml(xColumn)
                + " y=" + escapeHtml(yColumn)
                + " points=" + points.size()
                + " skipped=" + skipped
                + "</span>"
                + buildLinePlotSvg(title, xColumn, yColumn, points)
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview scatterPlotPreview(
            String target,
            String label,
            String xColumn,
            String yColumn,
            List<ScatterPoint> points,
            int skipped
    ) {
        String title = yColumn + " vs " + xColumn;
        String plain = "ScatterPlot(" + target
                + ", format=" + label
                + ", x=" + xColumn
                + ", y=" + yColumn
                + ", points=" + points.size()
                + ", skipped=" + skipped
                + ")\n"
                + points.stream()
                .limit(20)
                .map(point -> formatNullableStat(point.x()) + " | " + formatNullableStat(point.y()))
                .collect(Collectors.joining("\n"))
                + (points.size() > 20 ? "\n... " + (points.size() - 20) + " more points" : "");
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Scatter Plot</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>format=" + label
                + " x=" + escapeHtml(xColumn)
                + " y=" + escapeHtml(yColumn)
                + " points=" + points.size()
                + " skipped=" + skipped
                + "</span>"
                + buildScatterPlotSvg(title, xColumn, yColumn, points)
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview histogramPreview(
            String target,
            String label,
            String column,
            int values,
            int skipped,
            List<HistogramBin> bins,
            double min,
            double max
    ) {
        String plain = "Histogram(" + target
                + ", format=" + label
                + ", column=" + column
                + ", values=" + values
                + ", skipped=" + skipped
                + ", bins=" + bins.size()
                + ", min=" + formatNullableStat(min)
                + ", max=" + formatNullableStat(max)
                + ")\n"
                + bins.stream()
                .map(bin -> "[" + formatNullableStat(bin.start()) + ", " + formatNullableStat(bin.end()) + ") | " + bin.count())
                .collect(Collectors.joining("\n"));
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Histogram</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>format=" + label
                + " column=" + escapeHtml(column)
                + " values=" + values
                + " skipped=" + skipped
                + " bins=" + bins.size()
                + " min=" + escapeHtml(formatNullableStat(min))
                + " max=" + escapeHtml(formatNullableStat(max))
                + "</span>"
                + buildHistogramSvg(column, bins)
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
