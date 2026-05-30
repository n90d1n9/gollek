package tech.kayys.gollek.jupyter;

import java.util.Arrays;

/**
 * Minimal notebook-native line chart payload for inline Jupyter rendering.
 */
public final class NotebookLineChart {
    private final String title;
    private final String xLabel;
    private final String yLabel;
    private final double[] values;

    private NotebookLineChart(String title, String xLabel, String yLabel, double[] values) {
        this.title = title;
        this.xLabel = xLabel;
        this.yLabel = yLabel;
        this.values = values;
    }

    public static NotebookLineChart of(String title, String xLabel, String yLabel, double... values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("values must not be empty");
        }
        return new NotebookLineChart(title, xLabel, yLabel, values.clone());
    }

    public String title() {
        return title;
    }

    public String xLabel() {
        return xLabel;
    }

    public String yLabel() {
        return yLabel;
    }

    public double[] values() {
        return values.clone();
    }

    public int size() {
        return values.length;
    }

    @Override
    public String toString() {
        return "NotebookLineChart(title=" + title + ", points=" + values.length
                + ", values=" + Arrays.toString(values) + ")";
    }
}
