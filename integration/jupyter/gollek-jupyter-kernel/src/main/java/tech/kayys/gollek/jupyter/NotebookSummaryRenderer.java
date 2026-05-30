package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookStats.groupAggregateValue;
import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;
import static tech.kayys.gollek.jupyter.NotebookTables.formatPercent;

import tech.kayys.gollek.jupyter.NotebookStats.GroupResult;
import tech.kayys.gollek.jupyter.NotebookStats.MissingColumn;
import tech.kayys.gollek.jupyter.NotebookStats.SchemaColumn;
import tech.kayys.gollek.jupyter.NotebookStats.ValueCount;

import java.util.List;

final class NotebookSummaryRenderer {

    private NotebookSummaryRenderer() {
    }

    static NotebookPreview valueCountsPreview(
            String target,
            String label,
            String column,
            int totalRows,
            int uniqueRows,
            List<ValueCount> visible,
            int otherRows,
            String chartSvg
    ) {
        StringBuilder plain = new StringBuilder("ValueCounts(")
                .append(target)
                .append(", format=").append(label)
                .append(", column=").append(column)
                .append(", rows=").append(totalRows)
                .append(", unique=").append(uniqueRows)
                .append(", top=").append(visible.size())
                .append(", other=").append(otherRows)
                .append(")\n")
                .append("value | count | percent");
        for (ValueCount valueCount : visible) {
            plain.append('\n')
                    .append(valueCount.value())
                    .append(" | ")
                    .append(valueCount.count())
                    .append(" | ")
                    .append(formatPercent(valueCount.count(), totalRows));
        }
        if (otherRows > 0) {
            plain.append("\n(other) | ")
                    .append(otherRows)
                    .append(" | ")
                    .append(formatPercent(otherRows, totalRows));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Value Counts</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" column=").append(escapeHtml(column))
                .append(" rows=").append(totalRows)
                .append(" unique=").append(uniqueRows)
                .append(" top=").append(visible.size())
                .append(" other=").append(otherRows)
                .append("</span>")
                .append(chartSvg)
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Value</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Count</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Percent</th>")
                .append("</tr></thead><tbody>");
        for (ValueCount valueCount : visible) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>")
                    .append(escapeHtml(valueCount.value()))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(valueCount.count())
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(formatPercent(valueCount.count(), totalRows))
                    .append("</td>")
                    .append("</tr>");
        }
        if (otherRows > 0) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;color:#555'>(other)</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(otherRows)
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(formatPercent(otherRows, totalRows))
                    .append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview groupByPreview(
            String target,
            String label,
            String groupColumn,
            String valueColumn,
            String aggregate,
            int totalRows,
            int totalGroups,
            int hiddenGroups,
            int skipped,
            List<GroupResult> visible,
            String chartSvg
    ) {
        String valueLabel = valueColumn == null ? "" : ", value=" + valueColumn;
        StringBuilder plain = new StringBuilder("GroupBy(")
                .append(target)
                .append(", format=").append(label)
                .append(", group=").append(groupColumn)
                .append(valueLabel)
                .append(", agg=").append(aggregate)
                .append(", rows=").append(totalRows)
                .append(", groups=").append(totalGroups)
                .append(", shown=").append(visible.size())
                .append(", hidden=").append(hiddenGroups)
                .append(", skipped=").append(skipped)
                .append(")\n");
        if (aggregate.equals("count")) {
            plain.append("group | rows | count");
            for (GroupResult result : visible) {
                plain.append('\n')
                        .append(result.group())
                        .append(" | ")
                        .append(result.rows())
                        .append(" | ")
                        .append(result.rows());
            }
        } else {
            plain.append("group | rows | numeric | ").append(aggregate);
            for (GroupResult result : visible) {
                plain.append('\n')
                        .append(result.group())
                        .append(" | ")
                        .append(result.rows())
                        .append(" | ")
                        .append(result.numeric())
                        .append(" | ")
                        .append(formatNullableStat(groupAggregateValue(result, aggregate)));
            }
        }
        if (hiddenGroups > 0) {
            plain.append("\n... ").append(hiddenGroups).append(" more groups");
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>GroupBy</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" group=").append(escapeHtml(groupColumn));
        if (valueColumn != null) {
            html.append(" value=").append(escapeHtml(valueColumn));
        }
        html.append(" agg=").append(aggregate)
                .append(" rows=").append(totalRows)
                .append(" groups=").append(totalGroups)
                .append(" shown=").append(visible.size())
                .append(" hidden=").append(hiddenGroups)
                .append(" skipped=").append(skipped)
                .append("</span>")
                .append(chartSvg)
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Group</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Rows</th>");
        if (!aggregate.equals("count")) {
            html.append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Numeric</th>");
        }
        html.append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>")
                .append(escapeHtml(aggregate))
                .append("</th>")
                .append("</tr></thead><tbody>");
        for (GroupResult result : visible) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>")
                    .append(escapeHtml(result.group()))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(result.rows())
                    .append("</td>");
            if (!aggregate.equals("count")) {
                html.append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                        .append(result.numeric())
                        .append("</td>");
            }
            html.append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(escapeHtml(formatNullableStat(groupAggregateValue(result, aggregate))))
                    .append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table>");
        if (hiddenGroups > 0) {
            html.append("<div style='margin-top:6px;color:#555'>")
                    .append(hiddenGroups)
                    .append(" more groups not shown.</div>");
        }
        html.append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview schemaPreview(
            String target,
            String label,
            int totalRows,
            List<SchemaColumn> columns
    ) {
        StringBuilder plain = new StringBuilder("Schema(")
                .append(target)
                .append(", format=").append(label)
                .append(", rows=").append(totalRows)
                .append(", columns=").append(columns.size())
                .append(")\n")
                .append("column | type | nonEmpty | missing | missingPercent | examples");
        for (SchemaColumn column : columns) {
            plain.append('\n')
                    .append(column.name())
                    .append(" | ")
                    .append(column.type())
                    .append(" | ")
                    .append(column.nonEmpty())
                    .append(" | ")
                    .append(column.missing())
                    .append(" | ")
                    .append(formatPercent(column.missing(), totalRows))
                    .append(" | ")
                    .append(String.join(", ", column.examples()));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Schema</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" rows=").append(totalRows)
                .append(" columns=").append(columns.size())
                .append("</span>")
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Column</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Type</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Non-empty</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Missing</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Missing %</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Examples</th>")
                .append("</tr></thead><tbody>");
        for (SchemaColumn column : columns) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>")
                    .append(escapeHtml(column.name()))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px'>")
                    .append(escapeHtml(column.type()))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(column.nonEmpty())
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(column.missing())
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(formatPercent(column.missing(), totalRows))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px'>")
                    .append(escapeHtml(String.join(", ", column.examples())))
                    .append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview missingPreview(
            String target,
            String label,
            int totalRows,
            List<MissingColumn> columns,
            long columnsWithMissing,
            String chartSvg
    ) {
        StringBuilder plain = new StringBuilder("Missing(")
                .append(target)
                .append(", format=").append(label)
                .append(", rows=").append(totalRows)
                .append(", columns=").append(columns.size())
                .append(", columnsWithMissing=").append(columnsWithMissing)
                .append(")\n")
                .append("column | missing | present | percent");
        for (MissingColumn column : columns) {
            plain.append('\n')
                    .append(column.name())
                    .append(" | ")
                    .append(column.missing())
                    .append(" | ")
                    .append(column.present())
                    .append(" | ")
                    .append(formatPercent(column.missing(), totalRows));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Missing Values</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" rows=").append(totalRows)
                .append(" columns=").append(columns.size())
                .append(" columnsWithMissing=").append(columnsWithMissing)
                .append("</span>")
                .append(chartSvg)
                .append("<table style='margin-top:8px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Column</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Missing</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Present</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Percent</th>")
                .append("</tr></thead><tbody>");
        for (MissingColumn column : columns) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>")
                    .append(escapeHtml(column.name()))
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(column.missing())
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(column.present())
                    .append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>")
                    .append(formatPercent(column.missing(), totalRows))
                    .append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
