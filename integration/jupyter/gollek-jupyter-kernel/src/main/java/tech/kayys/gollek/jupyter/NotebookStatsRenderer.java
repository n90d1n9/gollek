package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookCharts.formatCorrelation;
import static tech.kayys.gollek.jupyter.NotebookCharts.renderCorrelationHeatmap;
import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;

import tech.kayys.gollek.jupyter.NotebookStats.NumericColumn;
import tech.kayys.gollek.jupyter.NotebookStats.NumericSummary;

import java.util.List;
import java.util.stream.Collectors;

final class NotebookStatsRenderer {

    private NotebookStatsRenderer() {
    }

    static NotebookPreview describePreview(
            String target,
            String label,
            int rows,
            int columns,
            List<NumericSummary> summaries
    ) {
        StringBuilder plain = new StringBuilder("Describe(")
                .append(target)
                .append(", format=").append(label)
                .append(", rows=").append(rows)
                .append(", columns=").append(columns)
                .append(", numericColumns=").append(summaries.size())
                .append(")\n")
                .append("column | count | missing | mean | std | min | p25 | median | p75 | max");
        for (NumericSummary summary : summaries) {
            plain.append('\n')
                    .append(summary.name()).append(" | ")
                    .append(summary.count()).append(" | ")
                    .append(summary.missing()).append(" | ")
                    .append(formatNullableStat(summary.mean())).append(" | ")
                    .append(formatNullableStat(summary.std())).append(" | ")
                    .append(formatNullableStat(summary.min())).append(" | ")
                    .append(formatNullableStat(summary.p25())).append(" | ")
                    .append(formatNullableStat(summary.median())).append(" | ")
                    .append(formatNullableStat(summary.p75())).append(" | ")
                    .append(formatNullableStat(summary.max()));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Describe</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" rows=").append(rows)
                .append(" columns=").append(columns)
                .append(" numericColumns=").append(summaries.size())
                .append("</span>")
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Column</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Count</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Missing</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Mean</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Std</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Min</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>P25</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Median</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>P75</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Max</th>")
                .append("</tr></thead><tbody>");
        for (NumericSummary summary : summaries) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>").append(escapeHtml(summary.name())).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(summary.count()).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(summary.missing()).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.mean()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.std()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.min()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.p25()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.median()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.p75()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(summary.max()))).append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview correlationPreview(
            String target,
            String label,
            int rows,
            List<NumericColumn> numericColumns,
            Double[][] correlations,
            boolean heatmap
    ) {
        String[][] values = formattedCorrelations(numericColumns, correlations);
        StringBuilder plain = new StringBuilder("Correlation(")
                .append(target)
                .append(", format=").append(label)
                .append(", rows=").append(rows)
                .append(", numericColumns=").append(numericColumns.size())
                .append(", heatmap=").append(heatmap)
                .append(")\n")
                .append("column | ")
                .append(numericColumns.stream().map(NumericColumn::name).collect(Collectors.joining(" | ")));
        for (int i = 0; i < numericColumns.size(); i++) {
            plain.append('\n')
                    .append(numericColumns.get(i).name())
                    .append(" | ")
                    .append(String.join(" | ", values[i]));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Correlation</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" rows=").append(rows)
                .append(" numericColumns=").append(numericColumns.size())
                .append(" heatmap=").append(heatmap)
                .append("</span>")
                .append(heatmap ? renderCorrelationHeatmap(numericColumns, correlations) : "")
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr><th style='border:1px solid #ccc;padding:4px;text-align:left'>Column</th>");
        for (NumericColumn column : numericColumns) {
            html.append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(escapeHtml(column.name()))
                    .append("</th>");
        }
        html.append("</tr></thead><tbody>");
        for (int i = 0; i < numericColumns.size(); i++) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>")
                    .append(escapeHtml(numericColumns.get(i).name()))
                    .append("</td>");
            for (String value : values[i]) {
                html.append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                        .append(escapeHtml(value))
                        .append("</td>");
            }
            html.append("</tr>");
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    private static String[][] formattedCorrelations(List<NumericColumn> numericColumns, Double[][] correlations) {
        String[][] values = new String[numericColumns.size()][numericColumns.size()];
        for (int i = 0; i < numericColumns.size(); i++) {
            for (int j = 0; j < numericColumns.size(); j++) {
                values[i][j] = formatCorrelation(correlations[i][j]);
            }
        }
        return values;
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
