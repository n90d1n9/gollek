package tech.kayys.gollek.ml;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.Gollek.DL.EvaluationOptions;
import tech.kayys.gollek.ml.Gollek.DL.EvaluationSummary;
import tech.kayys.gollek.ml.Gollek.DL.TrainingPreset;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Evaluation helpers exposed through {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlEvaluationFacade extends GollekDlFactoryFacade {
    protected GollekDlEvaluationFacade() {
    }

    public static EvaluationOptions.Builder evaluationOptions() {
        return EvaluationOptions.builder();
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, preset, evaluationOptions().build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            String deviceId,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, preset, evaluationOptions().device(deviceId).build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            EvaluationOptions options,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        Objects.requireNonNull(preset, "preset must not be null");
        return evaluate(model, loader, Gollek.DL.presetLoss(preset), options, metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, loss, evaluationOptions().build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            String deviceId,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, loss, evaluationOptions().device(deviceId).build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            EvaluationOptions options,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories =
                metrics == null ? List.of() : Arrays.asList(metrics);
        return evaluate(model, loader, loss, options, metricFactories);
    }

    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            String deviceId,
            List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
        return evaluate(model, loader, loss, evaluationOptions().device(deviceId).build(), metricFactories);
    }

    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            EvaluationOptions options,
            List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
        return GollekEvaluationRuntime.evaluate(model, loader, loss, options, metricFactories);
    }
}
