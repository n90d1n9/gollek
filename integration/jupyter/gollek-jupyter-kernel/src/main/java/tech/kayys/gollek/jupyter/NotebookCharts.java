package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;
import static tech.kayys.gollek.jupyter.NotebookTables.formatPercent;
import static tech.kayys.gollek.jupyter.NotebookStats.groupAggregateValue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

import tech.kayys.gollek.jupyter.NotebookStats.GroupResult;
import tech.kayys.gollek.jupyter.NotebookStats.MissingColumn;
import tech.kayys.gollek.jupyter.NotebookStats.NumericColumn;
import tech.kayys.gollek.jupyter.NotebookStats.ValueCount;

final class NotebookCharts {

    private NotebookCharts() {
    }

    record LinePlotPoint(String xLabel, double y) {}

    record ScatterPoint(double x, double y) {}

    record HistogramBin(double start, double end, int count) {}

    static String buildLinePlotSvg(String title, String xLabel, String yLabel, List<LinePlotPoint> points) {
        int width = 720;
        int height = 320;
        int left = 64;
        int top = 36;
        int plotWidth = 600;
        int plotHeight = 220;
        double min = points.stream().mapToDouble(LinePlotPoint::y).min().orElse(0.0);
        double max = points.stream().mapToDouble(LinePlotPoint::y).max().orElse(1.0);
        if (Double.compare(min, max) == 0) {
            min -= 1.0;
            max += 1.0;
        }
        double range = max - min;
        StringBuilder polyline = new StringBuilder();
        for (int i = 0; i < points.size(); i++) {
            double x = points.size() == 1
                    ? left + plotWidth / 2.0
                    : left + (plotWidth * i / (double) (points.size() - 1));
            double y = top + ((max - points.get(i).y()) / range) * plotHeight;
            if (!polyline.isEmpty()) {
                polyline.append(' ');
            }
            polyline.append(formatSvgNumber(x)).append(',').append(formatSvgNumber(y));
        }
        StringBuilder circles = new StringBuilder();
        if (points.size() <= 120) {
            for (int i = 0; i < points.size(); i++) {
                double x = points.size() == 1
                        ? left + plotWidth / 2.0
                        : left + (plotWidth * i / (double) (points.size() - 1));
                double y = top + ((max - points.get(i).y()) / range) * plotHeight;
                circles.append("<circle cx='").append(formatSvgNumber(x))
                        .append("' cy='").append(formatSvgNumber(y))
                        .append("' r='3' fill='#0b6bcb'><title>")
                        .append(escapeHtml(points.get(i).xLabel() + ": " + formatNullableStat(points.get(i).y())))
                        .append("</title></circle>");
            }
        }
        String firstLabel = points.getFirst().xLabel();
        String lastLabel = points.getLast().xLabel();
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='22' font-size='15' font-weight='700' fill='#111'>" + escapeHtml(title) + "</text>"
                + "<line x1='" + left + "' y1='" + top + "' x2='" + left + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<line x1='" + left + "' y1='" + (top + plotHeight) + "' x2='" + (left + plotWidth) + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<text x='8' y='" + (top + 5) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(max)) + "</text>"
                + "<text x='8' y='" + (top + plotHeight) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(min)) + "</text>"
                + "<text x='" + left + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(firstLabel) + "</text>"
                + "<text x='" + (left + plotWidth - 80) + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(lastLabel) + "</text>"
                + "<text x='" + (left + plotWidth / 2 - 30) + "' y='" + (height - 16) + "' font-size='12' fill='#333'>" + escapeHtml(xLabel) + "</text>"
                + "<text x='14' y='" + (top + plotHeight / 2) + "' transform='rotate(-90 14 " + (top + plotHeight / 2) + ")' font-size='12' fill='#333'>" + escapeHtml(yLabel) + "</text>"
                + "<polyline points='" + polyline + "' fill='none' stroke='#0b6bcb' stroke-width='2.5'/>"
                + circles
                + "</svg>";
    }

    static String buildScatterPlotSvg(String title, String xLabel, String yLabel, List<ScatterPoint> points) {
        int width = 720;
        int height = 340;
        int left = 70;
        int top = 42;
        int plotWidth = 590;
        int plotHeight = 220;
        double minX = points.stream().mapToDouble(ScatterPoint::x).min().orElse(0.0);
        double maxX = points.stream().mapToDouble(ScatterPoint::x).max().orElse(1.0);
        double minY = points.stream().mapToDouble(ScatterPoint::y).min().orElse(0.0);
        double maxY = points.stream().mapToDouble(ScatterPoint::y).max().orElse(1.0);
        if (Double.compare(minX, maxX) == 0) {
            minX -= 1.0;
            maxX += 1.0;
        }
        if (Double.compare(minY, maxY) == 0) {
            minY -= 1.0;
            maxY += 1.0;
        }
        double xRange = maxX - minX;
        double yRange = maxY - minY;
        int rendered = Math.min(points.size(), 1000);
        StringBuilder circles = new StringBuilder();
        for (int i = 0; i < rendered; i++) {
            ScatterPoint point = points.get(i);
            double x = left + ((point.x() - minX) / xRange) * plotWidth;
            double y = top + ((maxY - point.y()) / yRange) * plotHeight;
            circles.append("<circle cx='").append(formatSvgNumber(x))
                    .append("' cy='").append(formatSvgNumber(y))
                    .append("' r='3.5' fill='#0f766e' fill-opacity='0.72' stroke='#064e3b' stroke-opacity='0.35'>")
                    .append("<title>")
                    .append(escapeHtml(xLabel + "=" + formatNullableStat(point.x()) + ", " + yLabel + "=" + formatNullableStat(point.y())))
                    .append("</title></circle>");
        }
        String renderedLabel = points.size() > rendered ? "showing " + rendered + " of " + points.size() + " points" : points.size() + " points";
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='24' font-size='15' font-weight='700' fill='#111'>" + escapeHtml(title) + "</text>"
                + "<text x='16' y='42' font-size='11' fill='#555'>" + escapeHtml(renderedLabel) + "</text>"
                + "<line x1='" + left + "' y1='" + top + "' x2='" + left + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<line x1='" + left + "' y1='" + (top + plotHeight) + "' x2='" + (left + plotWidth) + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<text x='8' y='" + (top + 5) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(maxY)) + "</text>"
                + "<text x='8' y='" + (top + plotHeight) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(minY)) + "</text>"
                + "<text x='" + left + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(minX)) + "</text>"
                + "<text x='" + (left + plotWidth - 70) + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(maxX)) + "</text>"
                + "<text x='" + (left + plotWidth / 2 - 30) + "' y='" + (height - 18) + "' font-size='12' fill='#333'>" + escapeHtml(xLabel) + "</text>"
                + "<text x='16' y='" + (top + plotHeight / 2) + "' transform='rotate(-90 16 " + (top + plotHeight / 2) + ")' font-size='12' fill='#333'>" + escapeHtml(yLabel) + "</text>"
                + circles
                + "</svg>";
    }

    static List<HistogramBin> buildHistogramBins(List<Double> values, int requestedBins, double min, double max) {
        int binCount = Math.max(1, Math.min(requestedBins, values.size()));
        List<HistogramBin> bins = new ArrayList<>();
        if (Double.compare(min, max) == 0) {
            bins.add(new HistogramBin(min, max, values.size()));
            return bins;
        }
        int[] counts = new int[binCount];
        double width = (max - min) / binCount;
        for (double value : values) {
            int index = (int) ((value - min) / width);
            if (index >= binCount) {
                index = binCount - 1;
            }
            counts[Math.max(0, index)]++;
        }
        for (int i = 0; i < binCount; i++) {
            double start = min + width * i;
            double end = i == binCount - 1 ? max : start + width;
            bins.add(new HistogramBin(start, end, counts[i]));
        }
        return bins;
    }

    static String buildHistogramSvg(String column, List<HistogramBin> bins) {
        int width = 720;
        int height = 320;
        int left = 64;
        int top = 40;
        int plotWidth = 600;
        int plotHeight = 220;
        int maxCount = bins.stream().mapToInt(HistogramBin::count).max().orElse(1);
        double barGap = 4.0;
        double barWidth = Math.max(1.0, (plotWidth - barGap * (bins.size() - 1)) / bins.size());
        StringBuilder bars = new StringBuilder();
        for (int i = 0; i < bins.size(); i++) {
            HistogramBin bin = bins.get(i);
            double barHeight = maxCount == 0 ? 0 : plotHeight * bin.count() / (double) maxCount;
            double x = left + i * (barWidth + barGap);
            double y = top + plotHeight - barHeight;
            bars.append("<rect x='").append(formatSvgNumber(x))
                    .append("' y='").append(formatSvgNumber(y))
                    .append("' width='").append(formatSvgNumber(barWidth))
                    .append("' height='").append(formatSvgNumber(barHeight))
                    .append("' fill='#0b6bcb'><title>")
                    .append(escapeHtml("[" + formatNullableStat(bin.start()) + ", " + formatNullableStat(bin.end()) + "): " + bin.count()))
                    .append("</title></rect>");
        }
        HistogramBin first = bins.getFirst();
        HistogramBin last = bins.getLast();
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='24' font-size='15' font-weight='700' fill='#111'>Histogram: " + escapeHtml(column) + "</text>"
                + "<line x1='" + left + "' y1='" + top + "' x2='" + left + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<line x1='" + left + "' y1='" + (top + plotHeight) + "' x2='" + (left + plotWidth) + "' y2='" + (top + plotHeight) + "' stroke='#555'/>"
                + "<text x='18' y='" + (top + 6) + "' font-size='11' fill='#555'>" + maxCount + "</text>"
                + "<text x='42' y='" + (top + plotHeight) + "' font-size='11' fill='#555'>0</text>"
                + bars
                + "<text x='" + left + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(first.start())) + "</text>"
                + "<text x='" + (left + plotWidth - 70) + "' y='" + (top + plotHeight + 20) + "' font-size='11' fill='#555'>" + escapeHtml(formatNullableStat(last.end())) + "</text>"
                + "<text x='" + (left + plotWidth / 2 - 35) + "' y='" + (height - 16) + "' font-size='12' fill='#333'>" + escapeHtml(column) + "</text>"
                + "<text x='14' y='" + (top + plotHeight / 2) + "' transform='rotate(-90 14 " + (top + plotHeight / 2) + ")' font-size='12' fill='#333'>count</text>"
                + "</svg>";
    }

    static String buildMissingSvg(List<MissingColumn> columns, int totalRows, int maxColumns) {
        List<MissingColumn> missingColumns = columns.stream()
                .filter(column -> column.missing() > 0)
                .sorted(Comparator.comparingInt(MissingColumn::missing).reversed().thenComparing(MissingColumn::name))
                .limit(maxColumns)
                .toList();
        if (missingColumns.isEmpty()) {
            return "<div style='margin-top:8px;color:#166534'>No blank cells found.</div>";
        }
        int width = 720;
        int left = 170;
        int top = 44;
        int rowHeight = 28;
        int barWidth = 460;
        int height = top + missingColumns.size() * rowHeight + 34;
        int max = missingColumns.stream().mapToInt(MissingColumn::missing).max().orElse(1);
        StringBuilder bars = new StringBuilder();
        for (int i = 0; i < missingColumns.size(); i++) {
            MissingColumn column = missingColumns.get(i);
            int y = top + i * rowHeight;
            double widthRatio = column.missing() / (double) max;
            double actualBarWidth = Math.max(2.0, barWidth * widthRatio);
            bars.append("<text x='").append(left - 8).append("' y='").append(y + 17)
                    .append("' font-size='11' fill='#374151' text-anchor='end'>")
                    .append(escapeHtml(truncateLabel(column.name(), 22)))
                    .append("</text>")
                    .append("<rect x='").append(left).append("' y='").append(y)
                    .append("' width='").append(formatSvgNumber(actualBarWidth))
                    .append("' height='18' rx='3' fill='#dc2626'>")
                    .append("<title>")
                    .append(escapeHtml(column.name() + ": " + column.missing() + " missing (" + formatPercent(column.missing(), totalRows) + ")"))
                    .append("</title></rect>")
                    .append("<text x='").append(left + actualBarWidth + 6).append("' y='").append(y + 14)
                    .append("' font-size='11' fill='#374151'>")
                    .append(column.missing())
                    .append(" (")
                    .append(formatPercent(column.missing(), totalRows))
                    .append(")</text>");
        }
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='24' font-size='15' font-weight='700' fill='#111'>Missing values</text>"
                + bars
                + "</svg>";
    }

    static String buildValueCountsSvg(String column, List<ValueCount> counts, int totalRows) {
        int width = 720;
        int left = 170;
        int top = 44;
        int rowHeight = 28;
        int barWidth = 460;
        int height = top + counts.size() * rowHeight + 34;
        int max = counts.stream().mapToInt(ValueCount::count).max().orElse(1);
        StringBuilder bars = new StringBuilder();
        for (int i = 0; i < counts.size(); i++) {
            ValueCount valueCount = counts.get(i);
            int y = top + i * rowHeight;
            double widthRatio = valueCount.count() / (double) max;
            double actualBarWidth = Math.max(2.0, barWidth * widthRatio);
            bars.append("<text x='").append(left - 8).append("' y='").append(y + 17)
                    .append("' font-size='11' fill='#374151' text-anchor='end'>")
                    .append(escapeHtml(truncateLabel(valueCount.value(), 22)))
                    .append("</text>")
                    .append("<rect x='").append(left).append("' y='").append(y)
                    .append("' width='").append(formatSvgNumber(actualBarWidth))
                    .append("' height='18' rx='3' fill='#2563eb'>")
                    .append("<title>")
                    .append(escapeHtml(valueCount.value() + ": " + valueCount.count() + " (" + formatPercent(valueCount.count(), totalRows) + ")"))
                    .append("</title></rect>")
                    .append("<text x='").append(left + actualBarWidth + 6).append("' y='").append(y + 14)
                    .append("' font-size='11' fill='#374151'>")
                    .append(valueCount.count())
                    .append(" (")
                    .append(formatPercent(valueCount.count(), totalRows))
                    .append(")</text>");
        }
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='24' font-size='15' font-weight='700' fill='#111'>Value counts: " + escapeHtml(column) + "</text>"
                + bars
                + "</svg>";
    }

    static String buildGroupBySvg(String groupColumn, String aggregate, List<GroupResult> results) {
        List<GroupResult> chartRows = results.stream()
                .filter(result -> groupAggregateValue(result, aggregate) != null)
                .limit(40)
                .toList();
        if (chartRows.isEmpty()) {
            return "<div style='margin-top:8px;color:#555'>No numeric aggregate values to chart.</div>";
        }
        int width = 720;
        int left = 170;
        int top = 44;
        int rowHeight = 28;
        int barWidth = 460;
        int height = top + chartRows.size() * rowHeight + 34;
        double max = chartRows.stream()
                .map(result -> groupAggregateValue(result, aggregate))
                .filter(value -> value != null)
                .mapToDouble(Double::doubleValue)
                .max()
                .orElse(1.0);
        if (max <= 0.0) {
            max = 1.0;
        }
        StringBuilder bars = new StringBuilder();
        for (int i = 0; i < chartRows.size(); i++) {
            GroupResult result = chartRows.get(i);
            Double value = groupAggregateValue(result, aggregate);
            int y = top + i * rowHeight;
            double widthRatio = Math.max(0.0, value == null ? 0.0 : value / max);
            double actualBarWidth = Math.max(2.0, barWidth * widthRatio);
            bars.append("<text x='").append(left - 8).append("' y='").append(y + 17)
                    .append("' font-size='11' fill='#374151' text-anchor='end'>")
                    .append(escapeHtml(truncateLabel(result.group(), 22)))
                    .append("</text>")
                    .append("<rect x='").append(left).append("' y='").append(y)
                    .append("' width='").append(formatSvgNumber(actualBarWidth))
                    .append("' height='18' rx='3' fill='#7c3aed'>")
                    .append("<title>")
                    .append(escapeHtml(result.group() + ": " + formatNullableStat(value)))
                    .append("</title></rect>")
                    .append("<text x='").append(left + actualBarWidth + 6).append("' y='").append(y + 14)
                    .append("' font-size='11' fill='#374151'>")
                    .append(escapeHtml(formatNullableStat(value)))
                    .append("</text>");
        }
        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height
                + "' style='display:block;margin-top:8px;background:#fff'>"
                + "<text x='16' y='24' font-size='15' font-weight='700' fill='#111'>GroupBy: " + escapeHtml(aggregate + " by " + groupColumn) + "</text>"
                + bars
                + "</svg>";
    }

    static String formatCorrelation(Double value) {
        return value == null ? "" : formatNullableStat(value);
    }

    static String renderCorrelationHeatmap(List<NumericColumn> columns, Double[][] correlations) {
        int count = columns.size();
        int cell = count <= 10 ? 36 : count <= 20 ? 28 : 22;
        int left = Math.min(180, Math.max(72, longestColumnName(columns) * 7 + 20));
        int top = 74;
        int width = left + count * cell + 20;
        int height = top + count * cell + 54;
        boolean showText = cell >= 28;
        StringBuilder svg = new StringBuilder()
                .append("<div style='margin-top:10px;overflow:auto'>")
                .append("<svg role='img' aria-label='Correlation heatmap' width='").append(width)
                .append("' height='").append(height)
                .append("' viewBox='0 0 ").append(width).append(' ').append(height)
                .append("' xmlns='http://www.w3.org/2000/svg'>")
                .append("<rect width='100%' height='100%' fill='#ffffff'/>")
                .append("<text x='0' y='18' font-size='13' font-weight='700' fill='#111827'>Correlation heatmap</text>")
                .append("<text x='0' y='38' font-size='11' fill='#6b7280'>blue = negative, white = near zero, red = positive</text>");
        for (int i = 0; i < count; i++) {
            int x = left + i * cell + cell / 2;
            svg.append("<text x='").append(x).append("' y='").append(top - 8)
                    .append("' font-size='10' fill='#374151' text-anchor='start' transform='rotate(-45 ")
                    .append(x).append(' ').append(top - 8).append(")'>")
                    .append(escapeHtml(columns.get(i).name()))
                    .append("</text>");
        }
        for (int row = 0; row < count; row++) {
            int y = top + row * cell;
            svg.append("<text x='").append(left - 8).append("' y='").append(y + cell / 2 + 4)
                    .append("' font-size='11' fill='#374151' text-anchor='end'>")
                    .append(escapeHtml(columns.get(row).name()))
                    .append("</text>");
            for (int column = 0; column < count; column++) {
                int x = left + column * cell;
                Double value = correlations[row][column];
                String formatted = formatCorrelation(value);
                svg.append("<rect x='").append(x).append("' y='").append(y)
                        .append("' width='").append(cell).append("' height='").append(cell)
                        .append("' fill='").append(correlationColor(value))
                        .append("' stroke='#ffffff' stroke-width='1'>")
                        .append("<title>")
                        .append(escapeHtml(columns.get(row).name() + " vs " + columns.get(column).name() + ": " + (formatted.isBlank() ? "n/a" : formatted)))
                        .append("</title></rect>");
                if (showText && !formatted.isBlank()) {
                    svg.append("<text x='").append(x + cell / 2).append("' y='").append(y + cell / 2 + 4)
                            .append("' font-size='10' fill='").append(Math.abs(value) > 0.55 ? "#ffffff" : "#111827")
                            .append("' text-anchor='middle'>")
                            .append(escapeHtml(formatHeatmapValue(value)))
                            .append("</text>");
                }
            }
        }
        int legendY = top + count * cell + 24;
        svg.append("<rect x='").append(left).append("' y='").append(legendY)
                .append("' width='38' height='10' fill='#1d4ed8'/>")
                .append("<rect x='").append(left + 38).append("' y='").append(legendY)
                .append("' width='38' height='10' fill='#f8fafc'/>")
                .append("<rect x='").append(left + 76).append("' y='").append(legendY)
                .append("' width='38' height='10' fill='#b91c1c'/>")
                .append("<text x='").append(left).append("' y='").append(legendY + 26)
                .append("' font-size='10' fill='#6b7280'>-1</text>")
                .append("<text x='").append(left + 52).append("' y='").append(legendY + 26)
                .append("' font-size='10' fill='#6b7280'>0</text>")
                .append("<text x='").append(left + 102).append("' y='").append(legendY + 26)
                .append("' font-size='10' fill='#6b7280'>1</text>")
                .append("</svg></div>");
        return svg.toString();
    }

    static String formatSvgNumber(double value) {
        return String.format(Locale.ROOT, "%.2f", value);
    }

    private static int longestColumnName(List<NumericColumn> columns) {
        return columns.stream().map(NumericColumn::name).mapToInt(String::length).max().orElse(8);
    }

    private static String correlationColor(Double value) {
        if (value == null) {
            return "#f3f4f6";
        }
        double clamped = Math.max(-1.0, Math.min(1.0, value));
        if (clamped < 0.0) {
            double amount = Math.abs(clamped);
            return mixHex("#f8fafc", "#1d4ed8", amount);
        }
        return mixHex("#f8fafc", "#b91c1c", clamped);
    }

    private static String mixHex(String from, String to, double amount) {
        int r = mixChannel(from.substring(1, 3), to.substring(1, 3), amount);
        int g = mixChannel(from.substring(3, 5), to.substring(3, 5), amount);
        int b = mixChannel(from.substring(5, 7), to.substring(5, 7), amount);
        return String.format(Locale.ROOT, "#%02x%02x%02x", r, g, b);
    }

    private static int mixChannel(String from, String to, double amount) {
        int start = Integer.parseInt(from, 16);
        int end = Integer.parseInt(to, 16);
        return (int) Math.round(start + (end - start) * amount);
    }

    private static String formatHeatmapValue(double value) {
        if (Math.rint(value) == value) {
            return Long.toString((long) value);
        }
        return String.format(Locale.ROOT, "%.2f", value);
    }

    private static String truncateLabel(String value, int maxLength) {
        if (value.length() <= maxLength) {
            return value;
        }
        return value.substring(0, Math.max(0, maxLength - 3)) + "...";
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
