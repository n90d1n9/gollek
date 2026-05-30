package tech.kayys.gollek.jupyter;

final class NotebookHtml {

    private NotebookHtml() {
    }

    static String escape(String value) {
        return escapeText(value)
                .replace("\"", "&quot;")
                .replace("'", "&#39;");
    }

    static String escapeText(String value) {
        return value
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }
}
