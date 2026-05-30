package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;

class NotebookRuntimeRendererTest {

    @Test
    void resetPreviewKeepsNotebookMessage() {
        NotebookPreview preview = NotebookRuntimeRenderer.resetPreview();

        assertEquals("Notebook session reset. Default Gollek imports and helpers were reloaded.", preview.plain());
        assertTrue(preview.html().contains("Notebook Session Reset"));
        assertTrue(preview.html().contains("Default Gollek imports and helpers were reloaded."));
    }

    @Test
    void timedResultDisplayHandlesNullAndPlainResults() {
        DisplayData nullResult = NotebookRuntimeRenderer.timedResultDisplay("elapsedMs=1.000", null, null);
        DisplayData plainResult = NotebookRuntimeRenderer.timedResultDisplay("elapsedMs=1.000", 42, null);

        assertEquals("elapsedMs=1.000 result=<no display>", nullResult.getData(MIMEType.TEXT_PLAIN));
        assertEquals("elapsedMs=1.000 result=42", plainResult.getData(MIMEType.TEXT_PLAIN));
        assertFalse(plainResult.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void timedResultDisplayWrapsRichHtmlWithoutEscapingPayload() {
        DisplayData rich = new DisplayData("Chart(unsafe=<ok>)");
        rich.putData(MIMEType.TEXT_HTML, "<svg><text><ok></text></svg>");

        DisplayData timed = NotebookRuntimeRenderer.timedResultDisplay("runs=2 avgMs=1.000", "ignored", rich);

        assertEquals("runs=2 avgMs=1.000 result=Chart(unsafe=<ok>)", timed.getData(MIMEType.TEXT_PLAIN));
        assertTrue(timed.getData(MIMEType.TEXT_HTML).toString().contains("runs=2 avgMs=1.000"));
        assertTrue(timed.getData(MIMEType.TEXT_HTML).toString().contains("<svg><text><ok></text></svg>"));
    }

    @Test
    void envPreviewEscapesValuesAndSupportsUnset() {
        NotebookPreview set = NotebookRuntimeRenderer.envPreview("TOKEN", "<secret>");
        NotebookPreview unset = NotebookRuntimeRenderer.envPreview("MISSING", null);

        assertEquals("TOKEN=<secret>", set.plain());
        assertTrue(set.html().contains("TOKEN=&lt;secret&gt;"));
        assertEquals("MISSING=<unset>", unset.plain());
        assertTrue(unset.html().contains("MISSING=&lt;unset&gt;"));
    }

    @Test
    void classLocationPreviewEscapesClassAndPath() {
        NotebookPreview preview = NotebookRuntimeRenderer.classLocationPreview(
                "demo.<Unsafe>",
                "/tmp/<unsafe>.jar"
        );

        assertTrue(preview.plain().contains("Class demo.<Unsafe>"));
        assertTrue(preview.plain().contains("location=/tmp/<unsafe>.jar"));
        assertTrue(preview.html().contains("demo.&lt;Unsafe&gt;"));
        assertTrue(preview.html().contains("/tmp/&lt;unsafe&gt;.jar"));
    }
}
