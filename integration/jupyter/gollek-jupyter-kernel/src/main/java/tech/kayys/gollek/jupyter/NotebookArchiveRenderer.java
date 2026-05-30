package tech.kayys.gollek.jupyter;

import tech.kayys.gollek.jupyter.NotebookArchives.ArchiveListing;

import java.nio.charset.StandardCharsets;
import java.util.List;

final class NotebookArchiveRenderer {

    private NotebookArchiveRenderer() {
    }

    static NotebookPreview archiveListingPreview(String label, String target, ArchiveListing listing) {
        List<String> entries = listing.entries();
        long bytes = listing.bytes();
        String plain = entries.isEmpty()
                ? label + "(" + target + ", entries=0)"
                : label + "(" + target + ", entries=" + entries.size() + ", bytes=" + bytes + ")\n" + String.join("\n", entries);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>").append(label).append("</b> <code>").append(escapeHtml(target)).append("</code>")
                .append("<br><span style='color:#555'>entries=").append(entries.size())
                .append(" bytes=").append(bytes)
                .append("</span>");
        if (entries.isEmpty()) {
            html.append("<br><span style='color:#555'>No entries found.</span>");
        } else {
            html.append("<pre style='margin-top:6px;white-space:pre-wrap'>")
                    .append(escapeHtml(String.join("\n", entries)))
                    .append("</pre>");
        }
        html.append("</div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview archiveEntryPreview(
            String plainLabel,
            String htmlLabel,
            String target,
            String entryPart,
            long bytes,
            byte[] previewBytes,
            boolean truncated
    ) {
        String body = new String(previewBytes, StandardCharsets.UTF_8);
        String plain = plainLabel + "(" + target + "!" + entryPart
                + ", bytes=" + bytes
                + (truncated ? ", truncated=true" : "")
                + ")\n"
                + body;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>" + htmlLabel + "</b> <code>" + escapeHtml(target) + "!" + escapeHtml(entryPart) + "</code>"
                + "<br><span style='color:#555'>bytes=" + bytes + (truncated ? " truncated=true" : "") + "</span>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(body) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview extractPreview(
            boolean dryRun,
            String archive,
            String entryPart,
            String output,
            int bytes
    ) {
        String plain = "Extract(dryRun=" + dryRun
                + ", archive=" + archive
                + ", entry=" + entryPart
                + ", output=" + output
                + ", bytes=" + bytes
                + ")";
        if (!dryRun) {
            plain += "\nWrote " + output;
        }
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Extract</b> " + (dryRun ? "dry-run" : "written")
                + "<br><span style='color:#555'>bytes=" + bytes + "</span>"
                + "<br><code>" + escapeHtml(archive) + "!" + escapeHtml(entryPart) + "</code>"
                + "<br><code>" + escapeHtml(output) + "</code>"
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
