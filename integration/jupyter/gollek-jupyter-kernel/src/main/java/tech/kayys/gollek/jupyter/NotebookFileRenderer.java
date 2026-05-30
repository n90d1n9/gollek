package tech.kayys.gollek.jupyter;

import tech.kayys.gollek.jupyter.NotebookFiles.DiskUsageStats;
import tech.kayys.gollek.jupyter.NotebookFiles.TextPreview;

import java.util.List;

final class NotebookFileRenderer {

    private NotebookFileRenderer() {
    }

    static NotebookPreview workingDirectoryPreview(String directory) {
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Working Directory</b><br><code>" + escapeHtml(directory) + "</code></div>";
        return new NotebookPreview(directory, html);
    }

    static NotebookPreview workingDirectoryChangedPreview(String directory) {
        String plain = "Working directory changed to " + directory;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Working Directory Changed</b><br><code>" + escapeHtml(directory) + "</code></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview directoryPreview(String directory, List<String> entries) {
        String plain = "Directory(" + directory + ")\n" + String.join("\n", entries);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Directory</b> <code>").append(escapeHtml(directory)).append("</code>")
                .append("<ul style='margin-top:6px'>");
        for (String entry : entries) {
            html.append("<li>").append(escapeHtml(entry)).append("</li>");
        }
        html.append("</ul></div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview treePreview(String root, List<String> lines) {
        String plain = String.join("\n", lines);
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Tree</b> <code>" + escapeHtml(root) + "</code>"
                + "<pre style='margin-top:6px'>" + escapeHtml(plain) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview diskUsagePreview(String target, String type, DiskUsageStats stats) {
        String summary = "files=" + stats.files()
                + " directories=" + stats.directories()
                + " bytes=" + stats.bytes()
                + " human=" + NotebookFiles.formatBytes(stats.bytes())
                + " entriesScanned=" + stats.entriesScanned()
                + (stats.truncated() ? " truncated=true" : "");
        String plain = "DU(" + target + ", type=" + type + ")\n" + summary;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Disk Usage</b> <code>" + escapeHtml(target) + "</code>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(summary) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview filePreview(String target, TextPreview preview) {
        String plain = "File(" + target + ", bytes=" + preview.bytes()
                + (preview.truncated() ? ", truncated=true" : "")
                + ")\n"
                + preview.body();
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>File</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>bytes=" + preview.bytes() + (preview.truncated() ? " truncated=true" : "") + "</span>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(preview.body()) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview linePreview(String label, String target, List<String> previewLines, int totalLines) {
        String body = String.join("\n", previewLines);
        String plain = label + "(" + target + ", lines=" + previewLines.size() + "/" + totalLines + ")\n" + body;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>" + label + "</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>lines=" + previewLines.size() + "/" + totalLines + "</span>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(body) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview matchesPreview(String plainLabel, String htmlLabel, String pattern, List<String> matches) {
        String plain = matches.isEmpty()
                ? plainLabel + "(pattern=" + pattern + ", matches=0)"
                : plainLabel + "(pattern=" + pattern + ", matches=" + matches.size() + ")\n" + String.join("\n", matches);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>").append(htmlLabel).append("</b> <code>").append(escapeHtml(pattern)).append("</code>")
                .append("<br><span style='color:#555'>matches=").append(matches.size()).append("</span>");
        if (matches.isEmpty()) {
            html.append("<br><span style='color:#555'>No matches found.</span>");
        } else {
            html.append("<pre style='margin-top:6px;white-space:pre-wrap'>")
                    .append(escapeHtml(String.join("\n", matches)))
                    .append("</pre>");
        }
        html.append("</div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview wordCountPreview(String target, long lines, long words, int chars, long bytes) {
        String summary = "lines=" + lines + " words=" + words + " chars=" + chars + " bytes=" + bytes;
        String plain = "WC(" + target + ")\n" + summary;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Word Count</b> <code>" + escapeHtml(target) + "</code>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>"
                + escapeHtml(summary)
                + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview sha256Preview(String target, long bytes, String hash) {
        String plain = "SHA256(" + target + ", bytes=" + bytes + ")\n" + hash;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>SHA-256</b> <code>" + escapeHtml(target) + "</code>"
                + "<br><span style='color:#555'>bytes=" + bytes + "</span>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(hash) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview diffPreview(
            String left,
            String right,
            int changed,
            int added,
            int removed,
            List<String> diffLines
    ) {
        String summary = "Diff(left=" + left
                + ", right=" + right
                + ", changed=" + changed
                + ", added=" + added
                + ", removed=" + removed
                + ")";
        String plain = diffLines.isEmpty()
                ? summary + "\nNo line differences found."
                : summary + "\n" + String.join("\n", diffLines);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Diff</b><br><code>").append(escapeHtml(left)).append("</code>")
                .append("<br><code>").append(escapeHtml(right)).append("</code>")
                .append("<br><span style='color:#555'>changed=").append(changed)
                .append(" added=").append(added)
                .append(" removed=").append(removed)
                .append("</span>");
        if (diffLines.isEmpty()) {
            html.append("<br><span style='color:#555'>No line differences found.</span>");
        } else {
            html.append("<pre style='margin-top:6px;white-space:pre-wrap'>");
            for (String line : diffLines) {
                if (line.startsWith("- ")) {
                    html.append("<span style='color:#b31d28'>").append(escapeHtml(line)).append("</span>");
                } else if (line.startsWith("+ ")) {
                    html.append("<span style='color:#22863a'>").append(escapeHtml(line)).append("</span>");
                } else {
                    html.append(escapeHtml(line));
                }
                html.append('\n');
            }
            html.append("</pre>");
        }
        html.append("</div>");
        return new NotebookPreview(plain, html.toString());
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
