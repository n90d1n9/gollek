package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookTables.formatDelimitedPreviewRow;
import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;
import static tech.kayys.gollek.jupyter.NotebookTables.getCell;

import tech.kayys.gollek.jupyter.NotebookStats.ColumnProfile;

import java.util.List;

final class NotebookTableRenderer {

    private NotebookTableRenderer() {
    }

    static String plainRows(List<String> header, List<List<String>> rows, boolean numbered) {
        StringBuilder plain = new StringBuilder();
        if (numbered) {
            plain.append("# | ");
        }
        plain.append(String.join(" | ", header));
        for (int i = 0; i < rows.size(); i++) {
            plain.append('\n');
            if (numbered) {
                plain.append(i + 1).append(" | ");
            }
            plain.append(formatDelimitedPreviewRow(rows.get(i), header.size()));
        }
        return plain.toString();
    }

    static String htmlRows(List<String> header, List<List<String>> rows, boolean numbered) {
        return htmlRows(header, rows, numbered, false);
    }

    static String scrollableHtmlRows(List<String> header, List<List<String>> rows, boolean numbered) {
        return "<div style='margin-top:6px;max-height:420px;overflow:auto;border:1px solid #ddd;border-radius:4px'>"
                + htmlRows(header, rows, numbered, true)
                + "</div>";
    }

    static String plainColumnProfile(List<ColumnProfile> profiles) {
        StringBuilder plain = new StringBuilder("Profile\n")
                .append("column | nonEmpty | missing | numeric | min | max | mean");
        for (ColumnProfile profile : profiles) {
            plain.append('\n')
                    .append(profile.name()).append(" | ")
                    .append(profile.nonEmpty()).append(" | ")
                    .append(profile.missing()).append(" | ")
                    .append(profile.numeric()).append(" | ")
                    .append(formatNullableStat(profile.min())).append(" | ")
                    .append(formatNullableStat(profile.max())).append(" | ")
                    .append(formatNullableStat(profile.mean()));
        }
        return plain.toString();
    }

    static String htmlColumnProfile(List<ColumnProfile> profiles) {
        StringBuilder html = new StringBuilder()
                .append("<div style='margin-top:10px'><b>Column Profile</b></div>")
                .append("<table style='margin-top:6px;border-collapse:collapse'>")
                .append("<thead><tr>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:left'>Column</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Non-empty</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Missing</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Numeric</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Min</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Max</th>")
                .append("<th style='border:1px solid #ccc;padding:4px;text-align:right'>Mean</th>")
                .append("</tr></thead><tbody>");
        for (ColumnProfile profile : profiles) {
            html.append("<tr>")
                    .append("<td style='border:1px solid #ccc;padding:4px;font-weight:600'>").append(escapeHtml(profile.name())).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(profile.nonEmpty()).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(profile.missing()).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(profile.numeric()).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(profile.min()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(profile.max()))).append("</td>")
                    .append("<td style='border:1px solid #ccc;padding:4px;text-align:right'>").append(escapeHtml(formatNullableStat(profile.mean()))).append("</td>")
                    .append("</tr>");
        }
        html.append("</tbody></table>");
        return html.toString();
    }

    static NotebookPreview delimitedTablePreview(
            String label,
            String target,
            List<String> header,
            List<List<String>> allRows,
            List<List<String>> previewRows,
            boolean profile,
            List<ColumnProfile> profiles
    ) {
        boolean truncated = allRows.size() > previewRows.size();
        StringBuilder plain = new StringBuilder(label)
                .append("(").append(target)
                .append(", rows=").append(allRows.size())
                .append(", columns=").append(header.size())
                .append(", previewRows=").append(previewRows.size());
        if (truncated) {
            plain.append(", truncated=true");
        }
        if (profile) {
            plain.append(", profile=true");
        }
        plain.append(")\n")
                .append(plainRows(header, previewRows, true));
        if (truncated) {
            plain.append("\n... ").append(allRows.size() - previewRows.size()).append(" more rows");
        }
        if (profile) {
            plain.append("\n\n").append(plainColumnProfile(profiles));
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>").append(escapeHtml(label)).append("</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>rows=").append(allRows.size())
                .append(" columns=").append(header.size())
                .append(" previewRows=").append(previewRows.size());
        if (truncated) {
            html.append(" truncated=true");
        }
        html.append("</span>")
                .append(scrollableHtmlRows(header, previewRows, true));
        if (truncated) {
            html.append("<div style='margin-top:6px;color:#555'>")
                    .append(allRows.size() - previewRows.size())
                    .append(" more rows not shown. Use <code>-n</code> to change the preview size.</div>");
        }
        if (profile) {
            html.append(htmlColumnProfile(profiles));
        }
        html.append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview samplePreview(
            String target,
            String label,
            List<String> header,
            int totalRows,
            List<List<String>> sampledRows,
            String seedLabel
    ) {
        StringBuilder plain = new StringBuilder("Sample(")
                .append(target)
                .append(", format=").append(label)
                .append(", rows=").append(totalRows)
                .append(", sample=").append(sampledRows.size())
                .append(", columns=").append(header.size())
                .append(", seed=").append(seedLabel)
                .append(")\n")
                .append(plainRows(header, sampledRows, false));

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Sample</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" rows=").append(totalRows)
                .append(" sample=").append(sampledRows.size())
                .append(" columns=").append(header.size())
                .append(" seed=").append(escapeHtml(seedLabel))
                .append("</span>")
                .append(htmlRows(header, sampledRows, false))
                .append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview sortPreview(
            String target,
            String label,
            String column,
            String order,
            String mode,
            List<String> header,
            int totalRows,
            List<List<String>> visibleCells
    ) {
        boolean truncated = totalRows > visibleCells.size();
        StringBuilder plain = new StringBuilder("Sort(")
                .append(target)
                .append(", format=").append(label)
                .append(", column=").append(column)
                .append(", order=").append(order)
                .append(", mode=").append(mode)
                .append(", rows=").append(totalRows)
                .append(", previewRows=").append(visibleCells.size())
                .append(", columns=").append(header.size());
        if (truncated) {
            plain.append(", truncated=true");
        }
        plain.append(")\n")
                .append(plainRows(header, visibleCells, true));
        if (truncated) {
            plain.append("\n... ").append(totalRows - visibleCells.size()).append(" more rows");
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Sort</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" column=").append(escapeHtml(column))
                .append(" order=").append(order)
                .append(" mode=").append(mode)
                .append(" rows=").append(totalRows)
                .append(" previewRows=").append(visibleCells.size())
                .append(" columns=").append(header.size());
        if (truncated) {
            html.append(" truncated=true");
        }
        html.append("</span>")
                .append(htmlRows(header, visibleCells, true));
        if (truncated) {
            html.append("<div style='margin-top:6px;color:#555'>")
                    .append(totalRows - visibleCells.size())
                    .append(" more rows not shown. Use <code>-n</code> to change the preview size.</div>");
        }
        html.append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    static NotebookPreview filterPreview(
            String target,
            String label,
            String column,
            String operator,
            String value,
            String predicate,
            List<String> header,
            int totalRows,
            int matchedRows,
            List<List<String>> visibleRows
    ) {
        boolean truncated = matchedRows > visibleRows.size();
        StringBuilder plain = new StringBuilder("Filter(")
                .append(target)
                .append(", format=").append(label)
                .append(", column=").append(column)
                .append(", op=").append(operator);
        if (value != null) {
            plain.append(", value=").append(value);
        }
        plain.append(", rows=").append(totalRows)
                .append(", matched=").append(matchedRows)
                .append(", previewRows=").append(visibleRows.size())
                .append(", columns=").append(header.size());
        if (truncated) {
            plain.append(", truncated=true");
        }
        plain.append(")\n")
                .append(plainRows(header, visibleRows, true));
        if (truncated) {
            plain.append("\n... ").append(matchedRows - visibleRows.size()).append(" more matched rows");
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Filter</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>format=").append(label)
                .append(" predicate=").append(escapeHtml(predicate))
                .append(" rows=").append(totalRows)
                .append(" matched=").append(matchedRows)
                .append(" previewRows=").append(visibleRows.size())
                .append(" columns=").append(header.size());
        if (truncated) {
            html.append(" truncated=true");
        }
        html.append("</span>")
                .append(htmlRows(header, visibleRows, true));
        if (truncated) {
            html.append("<div style='margin-top:6px;color:#555'>")
                    .append(matchedRows - visibleRows.size())
                    .append(" more matched rows not shown. Use <code>-n</code> to change the preview size.</div>");
        }
        html.append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    private static String htmlRows(List<String> header, List<List<String>> rows, boolean numbered, boolean stickyHeader) {
        StringBuilder html = new StringBuilder()
                .append("<table style='")
                .append(stickyHeader ? "border-collapse:collapse;width:100%" : "margin-top:8px;border-collapse:collapse")
                .append("'>")
                .append("<thead><tr>");
        if (numbered) {
            appendHeaderCell(html, "#", true, stickyHeader);
        }
        for (String column : header) {
            appendHeaderCell(html, column, false, stickyHeader);
        }
        html.append("</tr></thead><tbody>");
        for (int i = 0; i < rows.size(); i++) {
            html.append("<tr>");
            if (numbered) {
                html.append("<td style='border:1px solid #ccc;padding:4px;text-align:right;color:#555'>")
                        .append(i + 1)
                        .append("</td>");
            }
            List<String> row = rows.get(i);
            for (int column = 0; column < header.size(); column++) {
                html.append("<td style='border:1px solid #ccc;padding:4px'>")
                        .append(escapeHtml(getCell(row, column)))
                        .append("</td>");
            }
            html.append("</tr>");
        }
        html.append("</tbody></table>");
        return html.toString();
    }

    private static void appendHeaderCell(StringBuilder html, String value, boolean numeric, boolean stickyHeader) {
        html.append("<th style='");
        if (stickyHeader) {
            html.append("position:sticky;top:0;background:#f6f8fa;");
        }
        html.append("border:1px solid #ccc;padding:4px;text-align:")
                .append(numeric ? "right" : "left")
                .append("'>")
                .append(escapeHtml(value))
                .append("</th>");
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
