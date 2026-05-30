package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;

final class NotebookRuntimeRenderer {

    private NotebookRuntimeRenderer() {
    }

    static NotebookPreview resetPreview() {
        String plain = "Notebook session reset. Default Gollek imports and helpers were reloaded.";
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Notebook Session Reset</b><br>"
                + "Default Gollek imports and helpers were reloaded."
                + "</div>";
        return new NotebookPreview(plain, html);
    }

    static DisplayData timedResultDisplay(String timing, Object result, DisplayData rich) {
        if (result == null) {
            return new DisplayData(timing + " result=<no display>");
        }
        if (rich == null) {
            return new DisplayData(timing + " result=" + result);
        }

        DisplayData timed = new DisplayData(rich);
        timed.putData(MIMEType.TEXT_PLAIN, timing + " result=" + rich.getData(MIMEType.TEXT_PLAIN));
        if (rich.hasDataForType(MIMEType.TEXT_HTML)) {
            timed.putData(MIMEType.TEXT_HTML,
                    "<div style='font-family:monospace;color:#555;margin-bottom:6px'>" + escapeHtml(timing) + "</div>"
                            + rich.getData(MIMEType.TEXT_HTML));
        }
        return timed;
    }

    static NotebookPreview envPreview(String name, String value) {
        String plain = value == null ? name + "=<unset>" : name + "=" + value;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Environment</b><br><code>" + escapeHtml(plain) + "</code></div>";
        return new NotebookPreview(plain, html);
    }

    static NotebookPreview classLocationPreview(String className, String location) {
        String plain = "Class " + className + "\nlocation=" + location;
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Class Location</b><br><code>" + escapeHtml(className) + "</code>"
                + "<br><code>" + escapeHtml(location) + "</code></div>";
        return new NotebookPreview(plain, html);
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
