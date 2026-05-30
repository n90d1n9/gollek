package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorDisplayLineChartTest {

    @Test
    void renderSupportsNotebookLineChart() {
        NotebookLineChart chart = NotebookLineChart.of(
                "Training Loss", "epoch", "loss", 0.91, 0.68, 0.52, 0.42, 0.31
        );

        DisplayData data = TensorDisplay.render(chart, null);

        assertNotNull(data);
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("LineChart(title=Training Loss, points=5"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("min=0.31"));
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("max=0.91"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("Training Loss"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("epoch"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("loss"));
    }
}
