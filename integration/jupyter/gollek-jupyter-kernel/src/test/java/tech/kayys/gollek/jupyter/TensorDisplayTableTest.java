package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorDisplayTableTest {

    @Test
    void renderSupportsMetricsStyleTableRows() {
        List<Map<String, Object>> rows = metricsRows();

        DisplayData data = TensorDisplay.render(rows, null);

        assertNotNull(data);
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("Table(rows=2, columns=[epoch, loss, accuracy])"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("<th"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("accuracy"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("0.88"));
    }

    @Test
    void renderSupportsSingleRecordSummary() {
        Map<String, Object> row = metricsRows().getFirst();

        DisplayData data = TensorDisplay.render(row, null);

        assertNotNull(data);
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("Record(keys=[epoch, loss, accuracy])"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("value"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("0.91"));
    }

    private static List<Map<String, Object>> metricsRows() {
        List<Map<String, Object>> rows = new ArrayList<>();

        Map<String, Object> row1 = new LinkedHashMap<>();
        row1.put("epoch", 1);
        row1.put("loss", 0.91);
        row1.put("accuracy", 0.72);
        rows.add(row1);

        Map<String, Object> row2 = new LinkedHashMap<>();
        row2.put("epoch", 2);
        row2.put("loss", 0.42);
        row2.put("accuracy", 0.88);
        rows.add(row2);

        return rows;
    }
}
