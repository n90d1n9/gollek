package tech.kayys.gollek.ml;

import static tech.kayys.gollek.ml.GollekTrainingOptionValidators.*;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.ml.multimodal.VisionBuilder;
import tech.kayys.gollek.ml.multimodal.MultimodalBuilder;
import tech.kayys.gollek.ml.multimodal.VideoBuilder;
import tech.kayys.gollek.ml.multimodal.AudioBuilder;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.ml.train.TrainerAccelerationPolicy;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.sdk.api.GollekSdk;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Main entry point for the Gollek ML framework.
 *
 * <p>
 * Provides static factory methods for creating tensors, building neural
 * networks, and querying device availability — mirroring the top-level
 * {@code torch} namespace in PyTorch, while {@code Gollek.ML} provides
 * a scikit-learn style API for traditional machine learning.
 */
public final class Gollek {

    /** Framework version. */
    public static final String VERSION = "0.1.1";

    private Gollek() {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Deep Learning (PyTorch Style)
    // ══════════════════════════════════════════════════════════════════════

    @SuppressWarnings("deprecation")
    public static class DL extends GollekDlQualityProfileFacade {
        public enum TrainingPreset {
            REGRESSION_MSE_ADAMW,
            REGRESSION_MSE_SGD,
            REGRESSION_HUBER_ADAMW,
            REGRESSION_HUBER_SGD,
            REGRESSION_PINBALL_ADAMW,
            REGRESSION_PINBALL_SGD,
            REGRESSION_INTERVAL_SCORE_ADAMW,
            REGRESSION_INTERVAL_SCORE_SGD,
            REGRESSION_GAUSSIAN_NLL_ADAMW,
            REGRESSION_GAUSSIAN_NLL_SGD,
            REGRESSION_POISSON_NLL_ADAMW,
            REGRESSION_POISSON_NLL_SGD,
            REGRESSION_TWEEDIE_NLL_ADAMW,
            REGRESSION_TWEEDIE_NLL_SGD,
            REGRESSION_NEGATIVE_BINOMIAL_NLL_ADAMW,
            REGRESSION_NEGATIVE_BINOMIAL_NLL_SGD,
            REGRESSION_ZERO_INFLATED_POISSON_NLL_ADAMW,
            REGRESSION_ZERO_INFLATED_POISSON_NLL_SGD,
            REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_ADAMW,
            REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_SGD,
            CLASSIFICATION_CROSS_ENTROPY_ADAMW,
            CLASSIFICATION_CROSS_ENTROPY_SGD,
            CLASSIFICATION_FOCAL_ADAMW,
            CLASSIFICATION_FOCAL_SGD,
            CAUSAL_LANGUAGE_MODELING_ADAMW,
            CAUSAL_LANGUAGE_MODELING_SGD,
            CAUSAL_LM_ADAMW,
            CAUSAL_LM_SGD,
            BINARY_FOCAL_WITH_LOGITS_ADAMW,
            BINARY_FOCAL_WITH_LOGITS_SGD,
            BINARY_BCE_WITH_LOGITS_ADAMW,
            BINARY_BCE_WITH_LOGITS_SGD
        }

        public record TrainingOptions(
                double gradientClip,
                double gradientClipValue,
                int earlyStoppingPatience,
                double earlyStoppingMinDelta,
                String earlyStoppingMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode,
                Path checkpointDir,
                boolean resumeFromCheckpoint,
                boolean failOnCheckpointLoadError,
                boolean saveBestModelCheckpoint,
                boolean restoreBestModelAtEnd,
                String bestModelMonitorMetric,
                CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode,
                int gradientAccumulationSteps,
                boolean mixedPrecision,
                GradScaler gradScaler,
                String device,
                boolean failOnAcceleratorFallback,
                boolean dataDistributionDiagnostics,
                List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories,
                SchedulerFactory schedulerFactory,
                CanonicalTrainer.SchedulerStepUnit schedulerStepUnit,
                String schedulerMonitorMetric,
                float[] crossEntropyClassWeights,
                Float focalGamma,
                Float focalAlpha,
                float[] focalClassWeights,
                Float causalLanguageModelingIgnoreIndex,
                float[] bcePositiveWeights,
                float[] pinballQuantiles,
                Float intervalAlpha,
                Float intervalCrossingPenalty,
                Float tweediePower,
                Boolean tweedieLogInput,
                Float tweedieEps,
                Boolean negativeBinomialLogInput,
                Boolean negativeBinomialIncludeConstant,
                Float negativeBinomialEps,
                Boolean zeroInflatedPoissonLogRateInput,
                Boolean zeroInflatedPoissonIncludeConstant,
                Float zeroInflatedPoissonEps,
                Boolean zeroInflatedNegativeBinomialLogInput,
                Boolean zeroInflatedNegativeBinomialIncludeConstant,
                Float zeroInflatedNegativeBinomialEps) {

            public TrainingOptions {
                gradientClip = normalizeNonNegativeDouble(gradientClip);
                gradientClipValue = normalizeNonNegativeDouble(gradientClipValue);
                device = device == null || device.isBlank() ? "auto" : device.trim();
                metricFactories = List.copyOf(metricFactories == null ? List.of() : metricFactories);
                schedulerStepUnit = schedulerStepUnit == null
                        ? CanonicalTrainer.SchedulerStepUnit.BATCH
                        : schedulerStepUnit;
                schedulerMonitorMetric = normalizeBestModelMonitorMetric(schedulerMonitorMetric);
                earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(earlyStoppingMonitorMetric);
                earlyStoppingMonitorMode = earlyStoppingMonitorMode == null
                        ? (earlyStoppingMonitorMetric == null
                                ? CanonicalTrainer.BestModelMonitorMode.MIN
                                : CanonicalTrainer.BestModelMonitorMode.MAX)
                        : earlyStoppingMonitorMode;
                bestModelMonitorMetric = normalizeBestModelMonitorMetric(bestModelMonitorMetric);
                bestModelMonitorMode = bestModelMonitorMode == null
                        ? (bestModelMonitorMetric == null
                                ? CanonicalTrainer.BestModelMonitorMode.MIN
                                : CanonicalTrainer.BestModelMonitorMode.MAX)
                        : bestModelMonitorMode;
                crossEntropyClassWeights = normalizeCrossEntropyClassWeights(crossEntropyClassWeights);
                focalGamma = normalizeFocalGamma(focalGamma);
                focalAlpha = normalizeFocalAlpha(focalAlpha);
                focalClassWeights = normalizeFocalClassWeights(focalClassWeights);
                causalLanguageModelingIgnoreIndex =
                        normalizeCausalLanguageModelingIgnoreIndex(causalLanguageModelingIgnoreIndex);
                bcePositiveWeights = normalizeBcePositiveWeights(bcePositiveWeights);
                pinballQuantiles = normalizePinballQuantiles(pinballQuantiles);
                intervalAlpha = normalizeIntervalAlpha(intervalAlpha);
                intervalCrossingPenalty = normalizeIntervalCrossingPenalty(intervalCrossingPenalty);
                tweediePower = normalizeTweediePower(tweediePower);
                tweedieEps = normalizePositiveFloat(tweedieEps, "tweedieEps");
                negativeBinomialEps = normalizePositiveFloat(negativeBinomialEps, "negativeBinomialEps");
                zeroInflatedPoissonEps = normalizePositiveFloat(zeroInflatedPoissonEps, "zeroInflatedPoissonEps");
                zeroInflatedNegativeBinomialEps = normalizePositiveFloat(
                        zeroInflatedNegativeBinomialEps,
                        "zeroInflatedNegativeBinomialEps");
                if (gradScaler != null) {
                    mixedPrecision = true;
                }
            }

            public TrainingOptions(
                    double gradientClip,
                    int earlyStoppingPatience,
                    double earlyStoppingMinDelta,
                    Path checkpointDir,
                    boolean resumeFromCheckpoint,
                    boolean failOnCheckpointLoadError,
                    int gradientAccumulationSteps,
                    String device) {
                this(
                        gradientClip,
                        0.0,
                        earlyStoppingPatience,
                        earlyStoppingMinDelta,
                        null,
                        CanonicalTrainer.BestModelMonitorMode.MIN,
                        checkpointDir,
                        resumeFromCheckpoint,
                        failOnCheckpointLoadError,
                        true,
                        false,
                        null,
                        CanonicalTrainer.BestModelMonitorMode.MIN,
                        gradientAccumulationSteps,
                        false,
                        null,
                        device,
                        false,
                        false,
                        List.of(),
                        null,
                        CanonicalTrainer.SchedulerStepUnit.BATCH,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null);
            }

            public static Builder builder() {
                return new Builder();
            }

            public static final class Builder extends GollekTrainingCoreBuilderSupport<Builder> {
                private Builder() {
                }

                public TrainingOptions build() {
                    return new TrainingOptions(
                            gradientClip,
                            gradientClipValue,
                            earlyStoppingPatience,
                            earlyStoppingMinDelta,
                            earlyStoppingMonitorMetric,
                            earlyStoppingMonitorMode,
                            checkpointDir,
                            resumeFromCheckpoint,
                            failOnCheckpointLoadError,
                            saveBestModelCheckpoint,
                            restoreBestModelAtEnd,
                            bestModelMonitorMetric,
                            bestModelMonitorMode,
                            gradientAccumulationSteps,
                            mixedPrecision,
                            gradScaler,
                            device,
                            failOnAcceleratorFallback,
                            dataDistributionDiagnostics,
                            metricFactories,
                            schedulerFactory,
                            schedulerStepUnit,
                            schedulerMonitorMetric,
                            crossEntropyClassWeights,
                            focalGamma,
                            focalAlpha,
                            focalClassWeights,
                            causalLanguageModelingIgnoreIndex,
                            bcePositiveWeights,
                            pinballQuantiles,
                            intervalAlpha,
                            intervalCrossingPenalty,
                            tweediePower,
                            tweedieLogInput,
                            tweedieEps,
                            negativeBinomialLogInput,
                            negativeBinomialIncludeConstant,
                            negativeBinomialEps,
                            zeroInflatedPoissonLogRateInput,
                            zeroInflatedPoissonIncludeConstant,
                            zeroInflatedPoissonEps,
                            zeroInflatedNegativeBinomialLogInput,
                            zeroInflatedNegativeBinomialIncludeConstant,
                            zeroInflatedNegativeBinomialEps);
                }
            }
        }

        @FunctionalInterface
        public interface SchedulerFactory {
            LRScheduler create(Optimizer optimizer);
        }

        public record EvaluationOptions(
                String device,
                boolean failOnAcceleratorFallback) {

            public EvaluationOptions {
                device = TrainerAccelerationPolicy.normalizeDevice(device);
            }

            public static Builder builder() {
                return new Builder();
            }

            public static final class Builder {
                private String device = "auto";
                private boolean failOnAcceleratorFallback = false;

                private Builder() {
                }

                public Builder device(String device) {
                    this.device = TrainerAccelerationPolicy.normalizeDevice(device);
                    return this;
                }

                public Builder accelerator(String device) {
                    return device(device);
                }

                public Builder failOnAcceleratorFallback(boolean failOnAcceleratorFallback) {
                    this.failOnAcceleratorFallback = failOnAcceleratorFallback;
                    return this;
                }

                public Builder requireAccelerator() {
                    return failOnAcceleratorFallback(true);
                }

                public EvaluationOptions build() {
                    return new EvaluationOptions(device, failOnAcceleratorFallback);
                }
            }
        }

        public record EvaluationSummary(
                double loss,
                int batchCount,
                long sampleCount,
                Map<String, Double> metrics,
                Map<String, Object> metricDetails,
                Map<String, Object> metadata) {

            public EvaluationSummary(
                    double loss,
                    int batchCount,
                    long sampleCount,
                    Map<String, Double> metrics,
                    Map<String, Object> metadata) {
                this(loss, batchCount, sampleCount, metrics, Map.of(), metadata);
            }

            public EvaluationSummary {
                metrics = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metrics == null ? Map.of() : metrics));
                metricDetails = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metricDetails == null ? Map.of() : metricDetails));
                metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                        metadata == null ? Map.of() : metadata));
            }

            public double metric(String name) {
                return metrics.getOrDefault(name, Double.NaN);
            }
        }

        static Optimizer presetOptimizer(NNModule model, float learningRate, TrainingPreset preset) {
            return GollekTrainingPresetSupport.presetOptimizer(model, learningRate, preset);
        }

        static TrainingLossFunction presetLoss(TrainingPreset preset) {
            return GollekTrainingPresetSupport.presetLoss(preset);
        }

        static TrainingLossFunction presetLoss(
                TrainingPreset preset,
                float[] crossEntropyClassWeights,
                Float focalGamma,
                Float focalAlpha,
                float[] focalClassWeights,
                Float causalLanguageModelingIgnoreIndex,
                float[] bcePositiveWeights,
                float[] pinballQuantiles,
                Float intervalAlpha,
                Float intervalCrossingPenalty,
                Float tweediePower,
                Boolean tweedieLogInput,
                Float tweedieEps,
                Boolean negativeBinomialLogInput,
                Boolean negativeBinomialIncludeConstant,
                Float negativeBinomialEps,
                Boolean zeroInflatedPoissonLogRateInput,
                Boolean zeroInflatedPoissonIncludeConstant,
                Float zeroInflatedPoissonEps,
                Boolean zeroInflatedNegativeBinomialLogInput,
                Boolean zeroInflatedNegativeBinomialIncludeConstant,
                Float zeroInflatedNegativeBinomialEps) {
            return GollekTrainingPresetSupport.presetLoss(
                    preset,
                    crossEntropyClassWeights,
                    focalGamma,
                    focalAlpha,
                    focalClassWeights,
                    causalLanguageModelingIgnoreIndex,
                    bcePositiveWeights,
                    pinballQuantiles,
                    intervalAlpha,
                    intervalCrossingPenalty,
                    tweediePower,
                    tweedieLogInput,
                    tweedieEps,
                    negativeBinomialLogInput,
                    negativeBinomialIncludeConstant,
                    negativeBinomialEps,
                    zeroInflatedPoissonLogRateInput,
                    zeroInflatedPoissonIncludeConstant,
                    zeroInflatedPoissonEps,
                    zeroInflatedNegativeBinomialLogInput,
                    zeroInflatedNegativeBinomialIncludeConstant,
                    zeroInflatedNegativeBinomialEps);
        }

    }

    // ══════════════════════════════════════════════════════════════════════
    // Traditional ML (Scikit-Learn Style)
    // ══════════════════════════════════════════════════════════════════════

    public static class ML extends GollekClassicML {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Selection & Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static class Selection extends GollekModelSelectionFacade {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Hub
    // ══════════════════════════════════════════════════════════════════════

    public static class Hub extends GollekModelHubFacade {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model Export
    // ══════════════════════════════════════════════════════════════════════

    public static class Export extends GollekModelExportFacade {
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Creation — delegates to GradTensor
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor tensor(float[] data, long... shape) {
        return GradTensor.of(data, shape);
    }

    public static GradTensor tensor(float... data) {
        return GradTensor.of(data);
    }

    public static GradTensor zeros(long... shape) {
        return GradTensor.zeros(shape);
    }

    public static GradTensor ones(long... shape) {
        return GradTensor.ones(shape);
    }

    public static GradTensor randn(long... shape) {
        return GradTensor.randn(shape);
    }

    public static GradTensor rand(long... shape) {
        return GradTensor.rand(shape);
    }

    public static GradTensor arange(float start, float end, float step) {
        return GradTensor.arange(start, end, step);
    }

    public static GradTensor arange(int end) {
        return GradTensor.arange(0, end, 1);
    }

    public static GradTensor scalar(float value) {
        return GradTensor.scalar(value);
    }

    public static GradTensor eye(int n) {
        return GradTensor.eye(n);
    }

    public static GradTensor full(float value, long... shape) {
        return GradTensor.full(value, shape);
    }

    public static GradTensor uniform(double lo, double hi, long... shape) {
        return GradTensor.uniform(lo, hi, shape);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Tensor Operations (static)
    // ══════════════════════════════════════════════════════════════════════

    public static GradTensor cat(GradTensor... tensors) {
        return GradTensor.cat(tensors);
    }

    public static GradTensor cat(int dim, GradTensor... tensors) {
        return GradTensor.cat(dim, tensors);
    }

    public static GradTensor stack(GradTensor... tensors) {
        return GradTensor.stack(tensors);
    }

    public static GradTensor stack(int dim, GradTensor... tensors) {
        return GradTensor.stack(dim, tensors);
    }

    public static GradTensor where(GradTensor condition, GradTensor x, GradTensor y) {
        return GradTensor.where(condition, x, y);
    }

    public static GradTensor einsum(String equation, GradTensor a, GradTensor b) {
        return a.einsum(equation, b);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Gradient Control
    // ══════════════════════════════════════════════════════════════════════

    public static NoGrad noGrad() {
        return NoGrad.enter();
    }

    private static GollekSdk _sdk;

    private static synchronized GollekSdk sdk() {
        if (_sdk == null) {
            _sdk = GollekSdk.builder().build();
        }
        return _sdk;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Multimodal Facade
    // ══════════════════════════════════════════════════════════════════════

    public static VisionBuilder vision(String model) {
        return new VisionBuilder(model, sdk());
    }

    public static MultimodalBuilder multimodal(String model) {
        return new MultimodalBuilder(model, sdk());
    }

    public static VideoBuilder video(String model) {
        return new VideoBuilder(model, sdk());
    }

    public static AudioBuilder audio(String model) {
        return new AudioBuilder(model, sdk());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device Utilities
    // ══════════════════════════════════════════════════════════════════════

    public static boolean isCudaAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        return (os.contains("linux") || os.contains("windows")) && System.getenv("CUDA_PATH") != null;
    }

    public static boolean isMetalAvailable() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        return os.contains("mac") && arch.equals("aarch64");
    }

    public static DeviceType defaultDevice() {
        if (isCudaAvailable())
            return DeviceType.CUDA;
        if (isMetalAvailable())
            return DeviceType.METAL;
        return DeviceType.CPU;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Info
    // ══════════════════════════════════════════════════════════════════════

    public static void printInfo() {
        var device = defaultDevice();
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek ML Framework                         ║");
        System.out.println("║          Version " + String.format("%-32s", VERSION) + "║");
        System.out.println("╠══════════════════════════════════════════════════════╣");
        System.out.println("║  Device:  " + String.format("%-32s", device) + "║");
        System.out.println("║  CUDA:    " + String.format("%-32s", isCudaAvailable()) + "║");
        System.out.println("║  Metal:   " + String.format("%-32s", isMetalAvailable()) + "║");
        System.out.println("║  Java:    " + String.format("%-32s", System.getProperty("java.version")) + "║");
        System.out.println("║  Vector:  "
                + String.format("%-32s", jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() + "-bit")
                + "║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
