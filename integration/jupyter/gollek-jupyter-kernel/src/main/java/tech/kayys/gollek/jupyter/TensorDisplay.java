package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.Renderer;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import javax.imageio.ImageIO;

/**
 * Rich display rendering for Gollek types in Jupyter.
 * Falls back to plain text for unknown types.
 */
public class TensorDisplay {
    private static final DecimalFormat DECIMAL = new DecimalFormat("0.####", DecimalFormatSymbols.getInstance(Locale.ROOT));
    private static final int SAMPLE_LIMIT = 6;

    private TensorDisplay() {}

    /**
     * Returns a {@link DisplayData} with HTML + plain text for known Gollek
     * types, or {@code null} to let the caller fall back to {@code toString()}.
     */
    public static DisplayData render(Object value, Renderer renderer) {
        if (value == null) return null;

        if (value instanceof NotebookAudioClip clip) {
            return renderAudioClip(clip);
        }
        if (value instanceof BufferedImage image) {
            return renderBufferedImage(image);
        }
        if (value instanceof NotebookLineChart chart) {
            return renderLineChart(chart);
        }
        if (value instanceof Map<?, ?> map) {
            return renderMap(map);
        }
        if (value instanceof List<?> list) {
            DisplayData table = renderListTable(list);
            if (table != null) {
                return table;
            }
        }

        // Use reflection so this compiles even if gollek-ml-tensor is absent at compile time
        try {
            Class<?> tensorClass = Class.forName("tech.kayys.gollek.spi.tensor.Tensor");
            if (tensorClass.isInstance(value)) {
                return renderTensor(value, tensorClass);
            }
        } catch (ClassNotFoundException ignored) {}

        // Current repo tensor API lives under core.tensor rather than the old spi package.
        try {
            Class<?> tensorClass = Class.forName("tech.kayys.gollek.core.tensor.Tensor");
            if (tensorClass.isInstance(value)) {
                return renderCoreTensor(value, tensorClass);
            }
        } catch (ClassNotFoundException ignored) {}

        try {
            Class<?> gradTensorClass = Class.forName("tech.kayys.gollek.ml.autograd.GradTensor");
            if (gradTensorClass.isInstance(value)) {
                return renderGradTensor(value, gradTensorClass);
            }
        } catch (ClassNotFoundException ignored) {}

        return null;
    }

    private static DisplayData renderTensor(Object tensor, Class<?> cls) {
        try {
            long[] shape = (long[]) cls.getMethod("shape").invoke(tensor);
            String device = (String) cls.getMethod("device").invoke(tensor);
            String dtype = cls.getMethod("dtype").invoke(tensor).toString();
            TensorSummary summary = legacyTensorSummary(tensor, cls, shape);

            String html = buildHtml(tensor, cls, shape, device, dtype, summary);
            String plain = buildPlainSummary("Tensor", shape, device, dtype, summary, null);

            DisplayData data = new DisplayData(plain);
            data.putData("text/html", html);
            return data;
        } catch (Exception e) {
            return null;
        }
    }

    private static String buildHtml(Object tensor, Class<?> cls,
                                    long[] shape, String device, String dtype,
                                    TensorSummary summary) {
        StringBuilder sb = new StringBuilder();
        sb.append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>");
        sb.append("<b>Tensor</b> shape=").append(java.util.Arrays.toString(shape))
          .append(" device=").append(device).append(" dtype=").append(dtype);
        appendHtmlStats(sb, summary, null);

        // Heatmap for 2-D tensors up to 32×32
        if (shape.length == 2 && shape[0] <= 32 && shape[1] <= 32) {
            String img = heatmap(tensor, cls, (int) shape[0], (int) shape[1]);
            if (img != null) {
                sb.append("<br><img src='data:image/png;base64,").append(img)
                  .append("' style='image-rendering:pixelated;max-width:256px'/>");
            }
        }
        sb.append("</div>");
        return sb.toString();
    }

    private static String heatmap(Object tensor, Class<?> cls, int rows, int cols) {
        try {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            double[][] data = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double v = (double) cls.getMethod("getDouble", int.class, int.class)
                                          .invoke(tensor, i, j);
                    data[i][j] = v;
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
            }
            double range = max - min == 0 ? 1 : max - min;
            BufferedImage img = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    float hue = 0.6f * (1f - (float) ((data[i][j] - min) / range));
                    img.setRGB(j, i, Color.HSBtoRGB(hue, 1f, 1f));
                }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ImageIO.write(img, "png", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (Exception e) {
            return null;
        }
    }

    private static DisplayData renderCoreTensor(Object tensor, Class<?> tensorClass) {
        try {
            Object shapeObj = tensorClass.getMethod("shape").invoke(tensor);
            long[] shape = (long[]) shapeObj.getClass().getMethod("dims").invoke(shapeObj);
            String device = tensorClass.getMethod("device").invoke(tensor).toString();
            String dtype = tensorClass.getMethod("dtype").invoke(tensor).toString();
            TensorSummary summary = coreTensorSummary(tensor, shape, dtype);

            String html = buildCoreTensorHtml(tensor, shape, device, dtype, summary);
            String plain = buildPlainSummary("Tensor", shape, device, dtype, summary, null);

            DisplayData data = new DisplayData(plain);
            data.putData("text/html", html);
            return data;
        } catch (Exception e) {
            return null;
        }
    }

    private static String buildCoreTensorHtml(Object tensor, long[] shape, String device, String dtype,
                                              TensorSummary summary) {
        StringBuilder sb = new StringBuilder();
        sb.append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>");
        sb.append("<b>Tensor</b> shape=").append(java.util.Arrays.toString(shape))
                .append(" device=").append(device).append(" dtype=").append(dtype);
        appendHtmlStats(sb, summary, null);

        if (shape.length == 2 && shape[0] <= 32 && shape[1] <= 32) {
            String img = coreTensorHeatmap(tensor, shape);
            if (img != null) {
                sb.append("<br><img src='data:image/png;base64,").append(img)
                        .append("' style='image-rendering:pixelated;max-width:256px'/>");
            }
        }

        sb.append("</div>");
        return sb.toString();
    }

    private static String coreTensorHeatmap(Object tensor, long[] shape) {
        try {
            Object buffer = tensor.getClass().getMethod("buffer").invoke(tensor);
            MemorySegment segment = (MemorySegment) buffer.getClass().getMethod("segment").invoke(buffer);
            int rows = (int) shape[0];
            int cols = (int) shape[1];
            double[][] data = new double[rows][cols];
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int flatIndex = i * cols + j;
                    double value = segment.get(ValueLayout.JAVA_FLOAT, flatIndex * Float.BYTES);
                    data[i][j] = value;
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }

            double range = max - min == 0 ? 1 : max - min;
            BufferedImage img = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float hue = 0.6f * (1f - (float) ((data[i][j] - min) / range));
                    img.setRGB(j, i, Color.HSBtoRGB(hue, 1f, 1f));
                }
            }

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ImageIO.write(img, "png", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (Exception e) {
            return null;
        }
    }

    private static DisplayData renderGradTensor(Object tensor, Class<?> gradTensorClass) {
        try {
            long[] shape = (long[]) gradTensorClass.getMethod("shape").invoke(tensor);
            float[] values = (float[]) gradTensorClass.getMethod("data").invoke(tensor);
            boolean requiresGrad = (boolean) gradTensorClass.getMethod("requiresGrad").invoke(tensor);
            TensorSummary summary = summaryFromFloats(values);

            String html = buildGradTensorHtml(values, shape, requiresGrad, summary);
            String plain = buildPlainSummary("GradTensor", shape, null, null, summary,
                    "requiresGrad=" + requiresGrad);

            DisplayData data = new DisplayData(plain);
            data.putData("text/html", html);
            return data;
        } catch (Exception e) {
            return null;
        }
    }

    private static String buildGradTensorHtml(float[] values, long[] shape, boolean requiresGrad,
                                              TensorSummary summary) {
        StringBuilder sb = new StringBuilder();
        sb.append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>");
        sb.append("<b>GradTensor</b> shape=").append(java.util.Arrays.toString(shape))
                .append(" requiresGrad=").append(requiresGrad);
        appendHtmlStats(sb, summary, "requiresGrad=" + requiresGrad);

        if (shape.length == 2 && shape[0] <= 32 && shape[1] <= 32) {
            String img = gradTensorHeatmap(values, (int) shape[0], (int) shape[1]);
            if (img != null) {
                sb.append("<br><img src='data:image/png;base64,").append(img)
                        .append("' style='image-rendering:pixelated;max-width:256px'/>");
            }
        }

        sb.append("</div>");
        return sb.toString();
    }

    private static String gradTensorHeatmap(float[] values, int rows, int cols) {
        try {
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            double[][] data = new double[rows][cols];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double value = values[i * cols + j];
                    data[i][j] = value;
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }

            double range = max - min == 0 ? 1 : max - min;
            BufferedImage img = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float hue = 0.6f * (1f - (float) ((data[i][j] - min) / range));
                    img.setRGB(j, i, Color.HSBtoRGB(hue, 1f, 1f));
                }
            }

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ImageIO.write(img, "png", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (Exception e) {
            return null;
        }
    }

    private static DisplayData renderMap(Map<?, ?> map) {
        List<String> keys = new ArrayList<>();
        for (Object key : map.keySet()) {
            keys.add(String.valueOf(key));
        }
        String plain = "Record(keys=" + keys + ")";
        StringBuilder html = new StringBuilder();
        html.append("<div style='font-family:monospace'>")
                .append("<b>Record</b>")
                .append("<table style='border-collapse:collapse;margin-top:6px'>")
                .append("<thead><tr><th style='text-align:left;border-bottom:1px solid #ccc;padding:4px 8px'>key</th>")
                .append("<th style='text-align:left;border-bottom:1px solid #ccc;padding:4px 8px'>value</th></tr></thead><tbody>");
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            html.append("<tr><td style='padding:4px 8px;border-bottom:1px solid #eee'>")
                    .append(escapeHtml(String.valueOf(entry.getKey())))
                    .append("</td><td style='padding:4px 8px;border-bottom:1px solid #eee'>")
                    .append(escapeHtml(String.valueOf(entry.getValue())))
                    .append("</td></tr>");
        }
        html.append("</tbody></table></div>");
        DisplayData data = new DisplayData(plain);
        data.putData("text/html", html.toString());
        return data;
    }

    private static DisplayData renderBufferedImage(BufferedImage image) {
        String encoded = encodePng(image);
        if (encoded == null) {
            return null;
        }
        String plain = "Image(width=" + image.getWidth() + ", height=" + image.getHeight() + ", type=" + image.getType() + ")";
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>Image</b> width=" + image.getWidth()
                + " height=" + image.getHeight()
                + " type=" + image.getType()
                + "<br><img src='data:image/png;base64," + encoded + "' style='max-width:256px;image-rendering:auto'/>"
                + "</div>";
        DisplayData data = new DisplayData(plain);
        data.putData("text/html", html);
        data.putData("image/png", encoded);
        return data;
    }

    private static DisplayData renderAudioClip(NotebookAudioClip clip) {
        String encoded = Base64.getEncoder().encodeToString(clip.wavBytes());
        String plain = "Audio(title=" + clip.title()
                + ", sampleRate=" + clip.sampleRate()
                + ", channels=" + clip.channels()
                + ", bytes=" + clip.byteSize() + ")";
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>"
                + "<b>" + escapeHtml(clip.title()) + "</b>"
                + "<br><span style='color:#555'>sampleRate=" + clip.sampleRate()
                + " channels=" + clip.channels()
                + " bytes=" + clip.byteSize()
                + "</span>"
                + "<br><audio controls src='data:audio/wav;base64," + encoded + "'></audio>"
                + "</div>";
        DisplayData data = new DisplayData(plain);
        data.putData("text/html", html);
        try {
            data.putData(MIMEType.parse("audio/wav"), encoded);
        } catch (Exception e) {
            data.putData("audio/wav", encoded);
        }
        return data;
    }

    private static DisplayData renderLineChart(NotebookLineChart chart) {
        double[] values = chart.values();
        TensorSummary summary = summaryFromDoubles(values);
        String plain = "LineChart(title=" + chart.title()
                + ", points=" + values.length
                + ", min=" + formatNumber(summary.min)
                + ", max=" + formatNumber(summary.max)
                + ", sample=" + formatSample(summary.sample) + ")";

        StringBuilder html = new StringBuilder();
        html.append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>").append(escapeHtml(chart.title())).append("</b>")
                .append("<br><span style='color:#555'>points=").append(values.length)
                .append(" min=").append(formatNumber(summary.min))
                .append(" max=").append(formatNumber(summary.max))
                .append(" sample=").append(formatSample(summary.sample))
                .append("</span>")
                .append("<br>")
                .append(buildLineChartSvg(chart))
                .append("</div>");

        DisplayData data = new DisplayData(plain);
        data.putData("text/html", html.toString());
        return data;
    }

    private static DisplayData renderListTable(List<?> list) {
        if (list.isEmpty() || !(list.getFirst() instanceof Map<?, ?>)) {
            return null;
        }
        List<Map<?, ?>> rows = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> map)) {
                return null;
            }
            rows.add(map);
        }

        Set<String> columns = new LinkedHashSet<>();
        for (Map<?, ?> row : rows) {
            for (Object key : row.keySet()) {
                columns.add(String.valueOf(key));
            }
        }
        List<String> orderedColumns = new ArrayList<>(columns);
        String plain = "Table(rows=" + rows.size() + ", columns=" + orderedColumns + ")";

        StringBuilder html = new StringBuilder();
        html.append("<div style='font-family:monospace'>")
                .append("<b>Table</b> rows=").append(rows.size())
                .append(" columns=").append(escapeHtml(orderedColumns.toString()))
                .append("<table style='border-collapse:collapse;margin-top:6px'>")
                .append("<thead><tr>");
        for (String column : orderedColumns) {
            html.append("<th style='text-align:left;border-bottom:1px solid #ccc;padding:4px 8px'>")
                    .append(escapeHtml(column))
                    .append("</th>");
        }
        html.append("</tr></thead><tbody>");
        for (Map<?, ?> row : rows) {
            html.append("<tr>");
            for (String column : orderedColumns) {
                Object cell = row.get(column);
                html.append("<td style='padding:4px 8px;border-bottom:1px solid #eee'>")
                        .append(escapeHtml(String.valueOf(cell)))
                        .append("</td>");
            }
            html.append("</tr>");
        }
        html.append("</tbody></table></div>");

        DisplayData data = new DisplayData(plain);
        data.putData("text/html", html.toString());
        return data;
    }

    private static TensorSummary legacyTensorSummary(Object tensor, Class<?> cls, long[] shape) {
        try {
            if (shape.length == 0) {
                return TensorSummary.empty();
            }
            if (shape.length == 1) {
                int size = (int) shape[0];
                float[] values = new float[size];
                for (int i = 0; i < size; i++) {
                    values[i] = ((Number) cls.getMethod("getDouble", int.class).invoke(tensor, i)).floatValue();
                }
                return summaryFromFloats(values);
            }
            if (shape.length == 2) {
                int rows = (int) shape[0];
                int cols = (int) shape[1];
                float[] values = new float[rows * cols];
                int offset = 0;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        values[offset++] = ((Number) cls.getMethod("getDouble", int.class, int.class)
                                .invoke(tensor, i, j)).floatValue();
                    }
                }
                return summaryFromFloats(values);
            }
        } catch (Exception ignored) {}
        return TensorSummary.empty();
    }

    private static TensorSummary coreTensorSummary(Object tensor, long[] shape, String dtype) {
        try {
            if (!supportsFloatSummary(dtype)) {
                return TensorSummary.empty();
            }
            Object buffer = tensor.getClass().getMethod("buffer").invoke(tensor);
            MemorySegment segment = (MemorySegment) buffer.getClass().getMethod("segment").invoke(buffer);
            long elementCount = elementCount(shape);
            if (elementCount <= 0 || elementCount > Integer.MAX_VALUE) {
                return TensorSummary.empty();
            }
            float[] values = new float[(int) elementCount];
            for (int i = 0; i < values.length; i++) {
                values[i] = segment.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            }
            return summaryFromFloats(values);
        } catch (Exception e) {
            return TensorSummary.empty();
        }
    }

    private static TensorSummary summaryFromFloats(float[] values) {
        if (values == null || values.length == 0) {
            return TensorSummary.empty();
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        int sampleCount = Math.min(values.length, SAMPLE_LIMIT);
        String[] sample = new String[sampleCount];
        for (int i = 0; i < values.length; i++) {
            double value = values[i];
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (i < sampleCount) {
                sample[i] = formatNumber(value);
            }
        }
        return new TensorSummary(values.length, min, max, sample);
    }

    private static TensorSummary summaryFromDoubles(double[] values) {
        if (values == null || values.length == 0) {
            return TensorSummary.empty();
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        int sampleCount = Math.min(values.length, SAMPLE_LIMIT);
        String[] sample = new String[sampleCount];
        for (int i = 0; i < values.length; i++) {
            double value = values[i];
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (i < sampleCount) {
                sample[i] = formatNumber(value);
            }
        }
        return new TensorSummary(values.length, min, max, sample);
    }

    private static long elementCount(long[] shape) {
        long count = 1L;
        for (long dim : shape) {
            count *= dim;
        }
        return count;
    }

    private static boolean supportsFloatSummary(String dtype) {
        return "F32".equals(dtype) || "F16".equals(dtype) || "BF16".equals(dtype);
    }

    private static String buildPlainSummary(String kind, long[] shape, String device, String dtype,
                                            TensorSummary summary, String extra) {
        StringBuilder sb = new StringBuilder(kind)
                .append("(shape=").append(java.util.Arrays.toString(shape));
        if (device != null) {
            sb.append(", device=").append(device);
        }
        if (dtype != null) {
            sb.append(", dtype=").append(dtype);
        }
        if (extra != null && !extra.isBlank()) {
            sb.append(", ").append(extra);
        }
        if (summary.available()) {
            sb.append(", size=").append(summary.size)
                    .append(", min=").append(formatNumber(summary.min))
                    .append(", max=").append(formatNumber(summary.max))
                    .append(", sample=").append(formatSample(summary.sample));
        }
        sb.append(")");
        return sb.toString();
    }

    private static void appendHtmlStats(StringBuilder sb, TensorSummary summary, String extra) {
        if (!summary.available() && (extra == null || extra.isBlank())) {
            return;
        }
        sb.append("<br><span style='color:#555'>");
        if (extra != null && !extra.isBlank()) {
            sb.append(extra);
            if (summary.available()) {
                sb.append(" | ");
            }
        }
        if (summary.available()) {
            sb.append("size=").append(summary.size)
                    .append(" min=").append(formatNumber(summary.min))
                    .append(" max=").append(formatNumber(summary.max))
                    .append(" sample=").append(formatSample(summary.sample));
        }
        sb.append("</span>");
    }

    private static String formatSample(String[] sample) {
        return "[" + String.join(", ", sample) + "]";
    }

    private static String formatNumber(double value) {
        synchronized (DECIMAL) {
            return DECIMAL.format(value);
        }
    }

    private static String escapeHtml(String value) {
        return value
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }

    private static String buildLineChartSvg(NotebookLineChart chart) {
        double[] values = chart.values();
        int width = 320;
        int height = 160;
        int left = 30;
        int top = 10;
        int plotWidth = width - 50;
        int plotHeight = height - 40;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        double range = max - min == 0 ? 1.0 : max - min;
        StringBuilder points = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            double x = left + (values.length == 1 ? plotWidth / 2.0 : (plotWidth * i / (double) (values.length - 1)));
            double normalized = (values[i] - min) / range;
            double y = top + plotHeight - (normalized * plotHeight);
            if (i > 0) {
                points.append(' ');
            }
            points.append(formatNumber(x)).append(',').append(formatNumber(y));
        }

        return "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height + "'>"
                + "<rect x='0' y='0' width='" + width + "' height='" + height + "' fill='white'/>"
                + "<line x1='" + left + "' y1='" + (top + plotHeight) + "' x2='" + (left + plotWidth) + "' y2='" + (top + plotHeight) + "' stroke='#bbb'/>"
                + "<line x1='" + left + "' y1='" + top + "' x2='" + left + "' y2='" + (top + plotHeight) + "' stroke='#bbb'/>"
                + "<polyline fill='none' stroke='#2563eb' stroke-width='2' points='" + points + "'/>"
                + "<text x='" + left + "' y='" + (height - 8) + "' font-size='11' fill='#555'>" + escapeHtml(chart.xLabel()) + "</text>"
                + "<text x='8' y='" + top + "' font-size='11' fill='#555'>" + escapeHtml(chart.yLabel()) + "</text>"
                + "</svg>";
    }

    private static String encodePng(BufferedImage image) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ImageIO.write(image, "png", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (Exception e) {
            return null;
        }
    }

    private record TensorSummary(int size, double min, double max, String[] sample) {
        static TensorSummary empty() {
            return new TensorSummary(0, 0.0, 0.0, new String[0]);
        }

        boolean available() {
            return size > 0;
        }
    }
}
