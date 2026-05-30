package tech.kayys.gollek.jupyter;

import tech.kayys.gollek.jupyter.NotebookDependencies.MavenLookup;

import java.util.List;

final class NotebookDependencyRenderer {

    private NotebookDependencyRenderer() {
    }

    static NotebookPreview classpathPreview(List<String> entries) {
        String plain = "Classpath(entries=" + entries.size() + ")\n" + String.join("\n", entries);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Classpath</b> entries=").append(entries.size())
                .append("<ul style='margin-top:6px'>");
        for (String entry : entries) {
            html.append("<li>").append(escapeHtml(entry)).append("</li>");
        }
        html.append("</ul></div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview dependenciesPreview(List<String> entries) {
        String plain = entries.isEmpty()
                ? "Dependencies(dynamic=0)"
                : "Dependencies(dynamic=" + entries.size() + ")\n" + String.join("\n", entries);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Dependencies</b> dynamic=").append(entries.size());
        if (entries.isEmpty()) {
            html.append("<br><span style='color:#555'>No dynamic notebook dependencies loaded yet.</span>");
        } else {
            html.append("<ul style='margin-top:6px'>");
            for (String entry : entries) {
                html.append("<li>").append(escapeHtml(entry)).append("</li>");
            }
            html.append("</ul>");
        }
        html.append("</div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview jarAddedPreview(String jar) {
        String plain = "Added jar to notebook classpath: " + jar;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Classpath Updated</b><br><code>" + escapeHtml(jar) + "</code></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview remoteMavenArtifactAddedPreview(String coords, String jar) {
        return mavenArtifactAddedPreview("Fetched and added Maven artifact", "Remote Maven Artifact Added", coords, jar);
    }

    static NotebookPreview localMavenArtifactAddedPreview(String coords, String jar) {
        return mavenArtifactAddedPreview("Added Maven artifact from local cache", "Local Maven Artifact Added", coords, jar);
    }

    static NotebookPreview missingMavenArtifactPreview(String coords, MavenLookup lookup, boolean allowRemote, boolean explain) {
        String remoteHint = "mvn dependency:get -Dartifact=" + coords;
        StringBuilder plain = new StringBuilder()
                .append("Artifact not found in local Maven cache: ").append(coords);
        if (explain || allowRemote) {
            plain.append("\nsearched:")
                    .append("\n- ").append(lookup.m2Candidate())
                    .append("\n- ").append(lookup.gradleCandidateDir());
        }
        if (allowRemote) {
            plain.append("\nremoteResolution=not-available-in-kernel")
                    .append("\nfetchHint=").append("%maven --allow-remote --fetch ").append(coords)
                    .append("\nnextStep=").append(remoteHint)
                    .append("\nretry=").append("%maven ").append(coords);
        }

        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Artifact Missing</b><br><code>").append(escapeHtml(coords)).append("</code>");
        if (explain || allowRemote) {
            html.append("<br><span style='color:#555'>searched:</span><ul>")
                    .append("<li>").append(escapeHtml(lookup.m2Candidate().toString())).append("</li>")
                    .append("<li>").append(escapeHtml(lookup.gradleCandidateDir().toString())).append("</li>")
                    .append("</ul>");
        }
        if (allowRemote) {
            html.append("<span style='color:#555'>remote resolution is not performed by the kernel in this environment.</span>")
                    .append("<br><code>").append(escapeHtml("%maven --allow-remote --fetch " + coords)).append("</code>")
                    .append("<br><code>").append(escapeHtml(remoteHint)).append("</code>")
                    .append("<br><code>").append(escapeHtml("%maven " + coords)).append("</code>");
        }
        html.append("</div>");
        return new NotebookPreview(plain.toString(), html.toString());
    }

    private static NotebookPreview mavenArtifactAddedPreview(String plainPrefix, String htmlLabel, String coords, String jar) {
        String plain = plainPrefix + ": " + coords + "\n" + jar;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>" + htmlLabel + "</b><br><code>" + escapeHtml(coords) + "</code>"
                + "<br><code>" + escapeHtml(jar) + "</code></div>";
        return new NotebookPreview(plain, html);
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
