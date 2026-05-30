package tech.kayys.gollek.jupyter;

final class NotebookDocuments {

    private NotebookDocuments() {
    }

    static String prettyPrintJson(String json) {
        StringBuilder out = new StringBuilder();
        int indent = 0;
        boolean inString = false;
        boolean escaping = false;
        boolean sawStructural = false;
        for (int i = 0; i < json.length(); i++) {
            char ch = json.charAt(i);
            if (escaping) {
                out.append(ch);
                escaping = false;
                continue;
            }
            if (ch == '\\' && inString) {
                out.append(ch);
                escaping = true;
                continue;
            }
            if (ch == '"') {
                inString = !inString;
                out.append(ch);
                continue;
            }
            if (inString) {
                out.append(ch);
                continue;
            }
            if (Character.isWhitespace(ch)) {
                continue;
            }
            switch (ch) {
                case '{', '[' -> {
                    sawStructural = true;
                    out.append(ch).append('\n');
                    indent++;
                    appendIndent(out, indent);
                }
                case '}', ']' -> {
                    sawStructural = true;
                    out.append('\n');
                    indent = Math.max(0, indent - 1);
                    appendIndent(out, indent);
                    out.append(ch);
                }
                case ',' -> {
                    sawStructural = true;
                    out.append(ch).append('\n');
                    appendIndent(out, indent);
                }
                case ':' -> {
                    sawStructural = true;
                    out.append(": ");
                }
                default -> out.append(ch);
            }
        }
        if (inString) {
            throw new IllegalArgumentException("unterminated JSON string");
        }
        String pretty = out.toString().strip();
        if (!sawStructural || pretty.isEmpty()) {
            throw new IllegalArgumentException("input does not look like JSON");
        }
        return pretty;
    }

    static String renderMarkdownToHtml(String markdown) {
        StringBuilder html = new StringBuilder();
        boolean inList = false;
        boolean inCode = false;
        StringBuilder paragraph = new StringBuilder();
        for (String line : markdown.split("\\R", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith("```")) {
                if (paragraph.length() > 0) {
                    appendMarkdownParagraph(html, paragraph.toString());
                    paragraph.setLength(0);
                }
                if (inList) {
                    html.append("</ul>");
                    inList = false;
                }
                html.append(inCode ? "</code></pre>" : "<pre style='background:#f6f8fa;padding:8px;border-radius:4px;overflow:auto'><code>");
                inCode = !inCode;
                continue;
            }
            if (inCode) {
                html.append(escapeHtml(line)).append('\n');
                continue;
            }
            if (trimmed.isEmpty()) {
                if (paragraph.length() > 0) {
                    appendMarkdownParagraph(html, paragraph.toString());
                    paragraph.setLength(0);
                }
                if (inList) {
                    html.append("</ul>");
                    inList = false;
                }
                continue;
            }
            if (trimmed.startsWith("#")) {
                if (paragraph.length() > 0) {
                    appendMarkdownParagraph(html, paragraph.toString());
                    paragraph.setLength(0);
                }
                if (inList) {
                    html.append("</ul>");
                    inList = false;
                }
                int level = 0;
                while (level < trimmed.length() && trimmed.charAt(level) == '#') {
                    level++;
                }
                level = Math.min(level, 6);
                String content = trimmed.substring(level).trim();
                html.append("<h").append(level).append(">").append(renderInlineMarkdown(content)).append("</h").append(level).append(">");
                continue;
            }
            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (paragraph.length() > 0) {
                    appendMarkdownParagraph(html, paragraph.toString());
                    paragraph.setLength(0);
                }
                if (!inList) {
                    html.append("<ul>");
                    inList = true;
                }
                html.append("<li>").append(renderInlineMarkdown(trimmed.substring(2).trim())).append("</li>");
                continue;
            }
            if (paragraph.length() > 0) {
                paragraph.append(' ');
            }
            paragraph.append(trimmed);
        }
        if (paragraph.length() > 0) {
            appendMarkdownParagraph(html, paragraph.toString());
        }
        if (inList) {
            html.append("</ul>");
        }
        if (inCode) {
            html.append("</code></pre>");
        }
        return html.toString();
    }

    static String highlightYaml(String yaml) {
        StringBuilder html = new StringBuilder();
        for (String line : yaml.split("\\R", -1)) {
            String escapedLine = escapeHtml(line);
            String trimmed = line.stripLeading();
            int indent = line.length() - trimmed.length();
            String indentText = escapeHtml(line.substring(0, Math.min(indent, line.length())));
            if (trimmed.startsWith("#")) {
                html.append(indentText)
                        .append("<span style='color:#6a737d'>")
                        .append(escapeHtml(trimmed))
                        .append("</span>");
            } else {
                int colon = trimmed.indexOf(':');
                if (colon > 0) {
                    String key = escapeHtml(trimmed.substring(0, colon));
                    String rest = escapeHtml(trimmed.substring(colon));
                    html.append(indentText)
                            .append("<span style='color:#005cc5;font-weight:600'>")
                            .append(key)
                            .append("</span>")
                            .append(rest);
                } else {
                    html.append(escapedLine);
                }
            }
            html.append('\n');
        }
        return html.toString();
    }

    static String highlightToml(String toml) {
        StringBuilder html = new StringBuilder();
        for (String line : toml.split("\\R", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith("#")) {
                html.append("<span style='color:#6a737d'>")
                        .append(escapeHtml(line))
                        .append("</span>");
            } else if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
                html.append("<span style='color:#d73a49;font-weight:600'>")
                        .append(escapeHtml(line))
                        .append("</span>");
            } else {
                int equals = line.indexOf('=');
                if (equals > 0) {
                    String key = escapeHtml(line.substring(0, equals).trim());
                    String value = escapeHtml(line.substring(equals + 1).trim());
                    html.append("<span style='color:#005cc5;font-weight:600'>")
                            .append(key)
                            .append("</span> = ")
                            .append(value);
                } else {
                    html.append(escapeHtml(line));
                }
            }
            html.append('\n');
        }
        return html.toString();
    }

    static String highlightXml(String xml) {
        StringBuilder html = new StringBuilder();
        for (String line : xml.split("\\R", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith("<!--") && trimmed.endsWith("-->")) {
                html.append("<span style='color:#6a737d'>")
                        .append(escapeHtml(line))
                        .append("</span>");
            } else {
                String escaped = escapeHtml(line);
                escaped = escaped.replaceAll("&lt;(/?)([A-Za-z0-9:_-]+)", "&lt;<span style='color:#d73a49;font-weight:600'>$1$2</span>");
                escaped = escaped.replaceAll("([A-Za-z_:][-A-Za-z0-9_:.]*)(=)(\"[^\"]*\")", "<span style='color:#005cc5;font-weight:600'>$1</span>$2$3");
                html.append(escaped);
            }
            html.append('\n');
        }
        return html.toString();
    }

    static String highlightIni(String ini) {
        StringBuilder html = new StringBuilder();
        for (String line : ini.split("\\R", -1)) {
            String trimmed = line.trim();
            if (trimmed.startsWith("#") || trimmed.startsWith(";")) {
                html.append("<span style='color:#6a737d'>")
                        .append(escapeHtml(line))
                        .append("</span>");
            } else if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
                html.append("<span style='color:#d73a49;font-weight:600'>")
                        .append(escapeHtml(line))
                        .append("</span>");
            } else {
                int equals = line.indexOf('=');
                if (equals > 0) {
                    String key = escapeHtml(line.substring(0, equals).trim());
                    String value = escapeHtml(line.substring(equals + 1).trim());
                    html.append("<span style='color:#005cc5;font-weight:600'>")
                            .append(key)
                            .append("</span> = ")
                            .append(value);
                } else {
                    html.append(escapeHtml(line));
                }
            }
            html.append('\n');
        }
        return html.toString();
    }

    private static void appendIndent(StringBuilder out, int indent) {
        for (int i = 0; i < indent; i++) {
            out.append("  ");
        }
    }

    private static void appendMarkdownParagraph(StringBuilder html, String paragraph) {
        html.append("<p>").append(renderInlineMarkdown(paragraph)).append("</p>");
    }

    private static String renderInlineMarkdown(String text) {
        String escaped = escapeHtml(text);
        escaped = escaped.replaceAll("`([^`]+)`", "<code>$1</code>");
        escaped = escaped.replaceAll("\\*\\*([^*]+)\\*\\*", "<strong>$1</strong>");
        escaped = escaped.replaceAll("\\*([^*]+)\\*", "<em>$1</em>");
        return escaped;
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
