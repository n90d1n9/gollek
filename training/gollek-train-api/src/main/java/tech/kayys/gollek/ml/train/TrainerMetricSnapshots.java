package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;
import tech.kayys.gollek.trainer.api.TrainerSession;

/**
 * Metric lifecycle and snapshot helpers for {@link CanonicalTrainer}.
 */
final class TrainerMetricSnapshots {
    private TrainerMetricSnapshots() {
    }

    interface FailureRecorder {
        String invalidValue(String phase, String metricName, double value, boolean optimizerStepSkipped);

        String invalidDetail(String phase, String metricName, String detailPath, double value);

        String invalidName(String phase, String metricName);

        String invalidFailure(
                String phase,
                String metricName,
                String kind,
                String detailPath,
                RuntimeException error);
    }

    static List<TrainingMetric> instantiate(
            List<Supplier<? extends TrainingMetric>> metricFactories,
            String phase) {
        if (metricFactories.isEmpty()) {
            return List.of();
        }
        List<TrainingMetric> metrics = new ArrayList<>(metricFactories.size());
        Set<String> metricNames = new LinkedHashSet<>();
        for (Supplier<? extends TrainingMetric> factory : metricFactories) {
            TrainingMetric metric = Objects.requireNonNull(factory.get(), phase + " metric factory returned null");
            String metricName = requireName(metric.name());
            if (!metricNames.add(metricName)) {
                throw new IllegalArgumentException(phase + " metric name must be unique, duplicate: " + metricName);
            }
            metrics.add(metric);
        }
        return List.copyOf(metrics);
    }

    static String requireName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("metric name must not be blank");
        }
        return name.trim();
    }

    static void reset(List<TrainingMetric> metrics) {
        for (TrainingMetric metric : metrics) {
            metric.reset();
        }
    }

    static Map<String, Double> snapshotValues(
            TrainerSession session,
            List<TrainingMetric> metrics,
            String phase,
            boolean finiteRequired,
            FailureRecorder failures) {
        if (metrics.isEmpty()) {
            return Map.of();
        }
        Objects.requireNonNull(failures, "failures must not be null");
        Map<String, Double> values = new LinkedHashMap<>();
        Set<String> metricNames = new LinkedHashSet<>();
        for (TrainingMetric metric : metrics) {
            String metricName = requireName(metric.name());
            requireUniqueSnapshotName(session, phase, metricName, metricNames, failures);
            double value;
            try {
                value = metric.value();
            } catch (RuntimeException error) {
                String message = failures.invalidFailure(phase, metricName, "value", null, error);
                stop(session);
                throw new IllegalArgumentException(message, error);
            }
            if (!Double.isFinite(value) && finiteRequired) {
                String message = failures.invalidValue(phase, metricName, value, false);
                stop(session);
                throw new IllegalArgumentException(message);
            }
            values.put(metricName, value);
        }
        return Map.copyOf(values);
    }

    static Map<String, Object> snapshotDetails(
            TrainerSession session,
            List<TrainingMetric> metrics,
            String phase,
            boolean finiteRequired,
            FailureRecorder failures) {
        if (metrics.isEmpty()) {
            return Map.of();
        }
        Objects.requireNonNull(failures, "failures must not be null");
        Map<String, Object> values = new LinkedHashMap<>();
        for (TrainingMetric metric : metrics) {
            if (metric instanceof DetailedTrainingMetric detailedMetric) {
                String metricName = requireName(metric.name());
                Map<String, Object> details;
                try {
                    details = detailedMetric.details();
                } catch (RuntimeException error) {
                    if (finiteRequired) {
                        String message = failures.invalidFailure(phase, metricName, "detail", "details", error);
                        stop(session);
                        throw new IllegalArgumentException(message, error);
                    }
                    continue;
                }
                if (details != null && !details.isEmpty()) {
                    Object immutableDetails = immutableSnapshot(details);
                    if (finiteRequired) {
                        requireFiniteDetail(session, phase, metricName, immutableDetails, "details", failures);
                    }
                    values.put(metricName, immutableDetails);
                }
            }
        }
        return values.isEmpty() ? Map.of() : Collections.unmodifiableMap(values);
    }

    static Object immutableSnapshot(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                copy.put(String.valueOf(entry.getKey()), immutableSnapshot(entry.getValue()));
            }
            return Collections.unmodifiableMap(copy);
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> copy = new ArrayList<>();
            for (Object item : iterable) {
                copy.add(immutableSnapshot(item));
            }
            return Collections.unmodifiableList(copy);
        }
        if (value != null && value.getClass().isArray()) {
            int length = java.lang.reflect.Array.getLength(value);
            List<Object> copy = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                copy.add(immutableSnapshot(java.lang.reflect.Array.get(value, i)));
            }
            return Collections.unmodifiableList(copy);
        }
        return value;
    }

    private static void requireUniqueSnapshotName(
            TrainerSession session,
            String phase,
            String metricName,
            Set<String> metricNames,
            FailureRecorder failures) {
        if (metricNames.add(metricName)) {
            return;
        }
        String message = failures.invalidName(phase, metricName);
        stop(session);
        throw new IllegalArgumentException(message);
    }

    private static void requireFiniteDetail(
            TrainerSession session,
            String phase,
            String metricName,
            Object detail,
            String path,
            FailureRecorder failures) {
        if (detail instanceof Double doubleValue && !Double.isFinite(doubleValue.doubleValue())) {
            failInvalidDetail(session, phase, metricName, path, doubleValue.doubleValue(), failures);
        }
        if (detail instanceof Float floatValue && !Float.isFinite(floatValue.floatValue())) {
            failInvalidDetail(session, phase, metricName, path, floatValue.doubleValue(), failures);
        }
        if (detail instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                requireFiniteDetail(
                        session,
                        phase,
                        metricName,
                        entry.getValue(),
                        path + "." + entry.getKey(),
                        failures);
            }
        } else if (detail instanceof Iterable<?> iterable) {
            int index = 0;
            for (Object item : iterable) {
                requireFiniteDetail(session, phase, metricName, item, path + "[" + index++ + "]", failures);
            }
        }
    }

    private static void failInvalidDetail(
            TrainerSession session,
            String phase,
            String metricName,
            String detailPath,
            double value,
            FailureRecorder failures) {
        String message = failures.invalidDetail(phase, metricName, detailPath, value);
        stop(session);
        throw new IllegalArgumentException(message);
    }

    private static void stop(TrainerSession session) {
        if (session != null) {
            session.stop();
        }
    }
}
