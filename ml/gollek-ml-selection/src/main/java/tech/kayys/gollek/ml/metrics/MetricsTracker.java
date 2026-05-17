package tech.kayys.gollek.ml.metrics;

import tech.kayys.gollek.ml.autograd.VectorOps;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Lightweight tracker for scalar training metrics.
 */
public final class MetricsTracker {

    private final Map<String, List<float[]>> history = new LinkedHashMap<>();

    public void log(String name, float value, int step) {
        history.computeIfAbsent(name, ignored -> new ArrayList<>())
            .add(new float[] { step, value });
    }

    public void logAll(Map<String, Float> metrics, int step) {
        metrics.forEach((name, value) -> log(name, value, step));
    }

    public List<float[]> get(String name) {
        return Collections.unmodifiableList(history.getOrDefault(name, List.of()));
    }

    public float latest(String name) {
        List<float[]> values = history.get(name);
        return (values == null || values.isEmpty()) ? Float.NaN : values.get(values.size() - 1)[1];
    }

    public float min(String name) {
        float[] values = values(name);
        if (values.length == 0) {
            return Float.NaN;
        }
        float min = Float.MAX_VALUE;
        for (float value : values) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    public float max(String name) {
        float[] values = values(name);
        return values.length == 0 ? Float.NaN : VectorOps.max(values);
    }

    public float mean(String name) {
        float[] values = values(name);
        return values.length == 0 ? Float.NaN : VectorOps.sum(values) / values.length;
    }

    public Map<String, Float> summary() {
        Map<String, Float> summary = new LinkedHashMap<>();
        history.keySet().forEach(name -> summary.put(name, latest(name)));
        return Collections.unmodifiableMap(summary);
    }

    public void exportCsv(Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("step,name,value");
        for (var entry : history.entrySet()) {
            for (float[] stepAndValue : entry.getValue()) {
                lines.add((int) stepAndValue[0] + "," + entry.getKey() + "," + stepAndValue[1]);
            }
        }
        Files.write(path, lines);
    }

    public void reset() {
        history.clear();
    }

    public Set<String> metricNames() {
        return Collections.unmodifiableSet(history.keySet());
    }

    private float[] values(String name) {
        List<float[]> values = history.get(name);
        if (values == null || values.isEmpty()) {
            return new float[0];
        }
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i)[1];
        }
        return array;
    }
}
