package tech.kayys.gollek.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.train.DetailedTrainingMetric;
import tech.kayys.gollek.ml.train.TrainerAccelerationMetadata;
import tech.kayys.gollek.ml.train.TrainerAccelerationPolicy;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.ml.train.TrainingMetric;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Executes the Gollek.DL evaluation loop and keeps the public facade thin.
 */
final class GollekEvaluationRuntime {
    private GollekEvaluationRuntime() {
    }

    static Gollek.DL.EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            Gollek.DL.EvaluationOptions options,
            List<? extends java.util.function.Supplier<? extends TrainingMetric>> metricFactories) {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(loader, "loader must not be null");
        Objects.requireNonNull(loss, "loss must not be null");

        Gollek.DL.EvaluationOptions resolvedOptions = options == null
                ? Gollek.DL.evaluationOptions().build()
                : options;
        String requestedDevice = resolvedOptions.device();
        List<TrainingMetric> resolvedMetrics = instantiateMetrics(metricFactories);
        boolean restoreTraining = model.isTraining();
        double weightedLoss = 0.0;
        long lossWeight = 0;
        long sampleCount = 0;
        int batchCount = 0;

        try (Acceleration.Scope ignored = Acceleration.prefer(requestedDevice);
                NoGrad ignoredNoGrad = NoGrad.enter()) {
            Acceleration.BackendStatus startStatus = Acceleration.status(requestedDevice);
            TrainerAccelerationPolicy.requireNoFallback(
                    requestedDevice,
                    startStatus,
                    resolvedOptions.failOnAcceleratorFallback());
            model.eval();
            for (Batch batch : loader) {
                Objects.requireNonNull(batch, "evaluation loader produced null batch");
                GradTensor prediction = model.forward(batch.inputs());
                GradTensor lossTensor = Objects.requireNonNull(
                        loss.compute(prediction, batch.labels()),
                        "loss function returned null");
                double batchLoss = requireFiniteLoss(lossTensor.item());
                long weight = Math.max(1L, batch.labels().numel());
                weightedLoss += batchLoss * weight;
                lossWeight += weight;
                sampleCount += batchSampleCount(batch);
                batchCount++;
                for (TrainingMetric metric : resolvedMetrics) {
                    metric.update(prediction, batch.labels());
                }
            }

            Map<String, Double> metricValues = metricValues(resolvedMetrics);
            Map<String, Object> metricDetails = metricDetails(resolvedMetrics);
            Acceleration.BackendStatus endStatus = Acceleration.status(requestedDevice);
            Map<String, Object> metadata = metadata(requestedDevice, startStatus, endStatus, resolvedOptions, resolvedMetrics);

            double meanLoss = lossWeight == 0 ? Double.NaN : weightedLoss / lossWeight;
            return new Gollek.DL.EvaluationSummary(
                    meanLoss,
                    batchCount,
                    sampleCount,
                    Collections.unmodifiableMap(metricValues),
                    metricDetails,
                    Collections.unmodifiableMap(metadata));
        } finally {
            if (restoreTraining) {
                model.train();
            } else {
                model.eval();
            }
        }
    }

    private static List<TrainingMetric> instantiateMetrics(
            List<? extends java.util.function.Supplier<? extends TrainingMetric>> metricFactories) {
        if (metricFactories == null || metricFactories.isEmpty()) {
            return List.of();
        }
        List<TrainingMetric> metrics = new ArrayList<>(metricFactories.size());
        for (java.util.function.Supplier<? extends TrainingMetric> factory : metricFactories) {
            TrainingMetric metric = Objects.requireNonNull(factory, "metric factory must not be null")
                    .get();
            Objects.requireNonNull(metric, "metric factory returned null");
            requireMetricName(metric.name());
            metric.reset();
            metrics.add(metric);
        }
        return metrics;
    }

    private static Map<String, Double> metricValues(List<TrainingMetric> metrics) {
        Map<String, Double> values = new LinkedHashMap<>();
        for (TrainingMetric metric : metrics) {
            values.put(requireMetricName(metric.name()), metric.value());
        }
        return values;
    }

    private static Map<String, Object> metricDetails(List<TrainingMetric> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> details = new LinkedHashMap<>();
        for (TrainingMetric metric : metrics) {
            if (metric instanceof DetailedTrainingMetric detailedMetric) {
                Map<String, Object> metricDetails = detailedMetric.details();
                if (metricDetails != null && !metricDetails.isEmpty()) {
                    details.put(requireMetricName(metric.name()), metricDetails);
                }
            }
        }
        return details.isEmpty() ? Map.of() : Collections.unmodifiableMap(details);
    }

    private static Map<String, Object> metadata(
            String requestedDevice,
            Acceleration.BackendStatus startStatus,
            Acceleration.BackendStatus endStatus,
            Gollek.DL.EvaluationOptions options,
            List<TrainingMetric> metrics) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        TrainerAccelerationMetadata.put(metadata, requestedDevice, endStatus, startStatus);
        metadata.put("failOnAcceleratorFallback", options.failOnAcceleratorFallback());
        metadata.put("metricsEnabled", !metrics.isEmpty());
        metadata.put("metricDetails", metricDetails(metrics));
        return metadata;
    }

    private static String requireMetricName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("metric name must not be blank");
        }
        return name.trim();
    }

    private static double requireFiniteLoss(double loss) {
        if (!Double.isFinite(loss)) {
            throw new IllegalArgumentException("evaluation loss must be finite, got " + loss);
        }
        return loss;
    }

    private static long batchSampleCount(Batch batch) {
        long[] shape = batch.labels().shape();
        return shape.length == 0 ? 1L : Math.max(1L, shape[0]);
    }
}
