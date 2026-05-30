package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookDocuments.highlightIni;
import static tech.kayys.gollek.jupyter.NotebookDocuments.highlightToml;
import static tech.kayys.gollek.jupyter.NotebookDocuments.highlightXml;
import static tech.kayys.gollek.jupyter.NotebookDocuments.highlightYaml;
import static tech.kayys.gollek.jupyter.NotebookDocuments.renderMarkdownToHtml;

import java.io.IOException;
import java.io.StringReader;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;

final class NotebookDocumentRenderer {

    private NotebookDocumentRenderer() {
    }

    static NotebookPreview markdownPreview(String target, String markdown) {
        String plain = "Markdown(" + target + ")\n" + markdown;
        String html = "<div style='font-family:system-ui,sans-serif;border:1px solid #ccc;padding:12px;border-radius:6px'>"
                + "<div style='font-family:monospace;margin-bottom:10px'><b>Markdown</b> <code>" + escapeHtml(target) + "</code></div>"
                + renderMarkdownToHtml(markdown)
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview htmlPreview(String target, String htmlSource) {
        String plain = "HTML(" + target + ")\n" + htmlSource;
        String html = "<div style='border:1px solid #ccc;padding:8px;border-radius:6px'>"
                + "<div style='font-family:monospace;margin-bottom:10px'><b>HTML</b> <code>" + escapeHtml(target) + "</code></div>"
                + htmlSource
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview jsonPreview(String target, String prettyJson) {
        String plain = "JSON(" + target + ")\n" + prettyJson;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>JSON</b> <code>" + escapeHtml(target) + "</code>"
                + "<pre style='margin-top:6px;white-space:pre-wrap'>" + escapeHtml(prettyJson) + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview yamlPreview(String target, String yaml) {
        return highlightedSourcePreview("YAML", target, yaml, highlightYaml(yaml));
    }

    static NotebookPreview tomlPreview(String target, String toml) {
        return highlightedSourcePreview("TOML", target, toml, highlightToml(toml));
    }

    static NotebookPreview xmlPreview(String target, String xml) {
        return highlightedSourcePreview("XML", target, xml, highlightXml(xml));
    }

    static NotebookPreview iniPreview(String target, String ini) {
        return highlightedSourcePreview("INI", target, ini, highlightIni(ini));
    }

    static NotebookPreview propertiesPreview(String target, String source) throws IOException {
        Properties properties = new Properties();
        try (StringReader reader = new StringReader(source)) {
            properties.load(reader);
        }
        List<String> keys = properties.stringPropertyNames().stream().sorted().toList();
        String plain = "Properties(" + target + ", entries=" + keys.size() + ")\n"
                + keys.stream()
                .map(key -> key + "=" + properties.getProperty(key))
                .collect(Collectors.joining("\n"));
        StringBuilder html = keyValueTable("Properties", target);
        for (String key : keys) {
            appendKeyValueRow(html, key, properties.getProperty(key));
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain, html.toString());
    }

    static NotebookPreview envFilePreview(String target, String source) {
        LinkedHashMap<String, String> entries = parseEnvEntries(source);
        String plain = "EnvFile(" + target + ", entries=" + entries.size() + ")\n"
                + entries.entrySet().stream()
                .map(entry -> entry.getKey() + "=" + entry.getValue())
                .collect(Collectors.joining("\n"));
        StringBuilder html = keyValueTable("Env File", target);
        for (Map.Entry<String, String> entry : entries.entrySet()) {
            appendKeyValueRow(html, entry.getKey(), entry.getValue());
        }
        html.append("</tbody></table></div>");
        return new NotebookPreview(plain, html.toString());
    }

    private static NotebookPreview highlightedSourcePreview(String label, String target, String source, String highlightedSource) {
        String plain = label + "(" + target + ")\n" + source;
        String html = "<div style='border:1px solid #ccc;padding:8px;border-radius:6px'>"
                + "<div style='font-family:monospace;margin-bottom:10px'><b>" + label + "</b> <code>" + escapeHtml(target) + "</code></div>"
                + "<pre style='margin:0;white-space:pre-wrap;font-family:monospace'>"
                + highlightedSource
                + "</pre></div>";
        return new NotebookPreview(plain, html);
    }

    private static LinkedHashMap<String, String> parseEnvEntries(String source) {
        LinkedHashMap<String, String> entries = new LinkedHashMap<>();
        for (String line : source.split("\\R")) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                continue;
            }
            int equals = trimmed.indexOf('=');
            if (equals <= 0) {
                continue;
            }
            String key = trimmed.substring(0, equals).trim();
            String value = trimmed.substring(equals + 1).trim();
            if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
                value = value.substring(1, value.length() - 1);
            }
            entries.put(key, value);
        }
        return entries;
    }

    private static StringBuilder keyValueTable(String title, String target) {
        return new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:6px'>")
                .append("<div style='margin-bottom:10px'><b>").append(title).append("</b> <code>").append(escapeHtml(target)).append("</code></div>")
                .append("<table style='border-collapse:collapse'>")
                .append("<thead><tr><th style='border:1px solid #ccc;padding:4px;text-align:left'>Key</th><th style='border:1px solid #ccc;padding:4px;text-align:left'>Value</th></tr></thead><tbody>");
    }

    private static void appendKeyValueRow(StringBuilder html, String key, String value) {
        html.append("<tr>")
                .append("<td style='border:1px solid #ccc;padding:4px;color:#005cc5;font-weight:600'>").append(escapeHtml(key)).append("</td>")
                .append("<td style='border:1px solid #ccc;padding:4px'>").append(escapeHtml(value)).append("</td>")
                .append("</tr>");
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
