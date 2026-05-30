package tech.kayys.gollek.ml;

import java.util.List;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.BinaryFocalWithLogitsLoss;
import tech.kayys.gollek.ml.nn.loss.CausalLanguageModelingLoss;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.FocalLoss;
import tech.kayys.gollek.ml.nn.loss.GaussianNllLoss;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.loss.NegativeBinomialNllLoss;
import tech.kayys.gollek.ml.nn.loss.PinballLoss;
import tech.kayys.gollek.ml.nn.loss.PoissonNllLoss;
import tech.kayys.gollek.ml.nn.loss.PredictionIntervalLoss;
import tech.kayys.gollek.ml.nn.loss.TweedieNllLoss;
import tech.kayys.gollek.ml.nn.loss.ZeroInflatedNegativeBinomialNllLoss;
import tech.kayys.gollek.ml.nn.loss.ZeroInflatedPoissonNllLoss;
import tech.kayys.gollek.ml.optim.AdamW;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.ml.train.TrainingMetric;
import tech.kayys.gollek.ml.train.TrainingMetrics;

/**
 * Resolves high-level Gollek training presets into optimizer/loss pairs.
 */
final class GollekTrainingPresetSupport {
    private GollekTrainingPresetSupport() {
    }

    static Optimizer presetOptimizer(
            NNModule model,
            float learningRate,
            Gollek.DL.TrainingPreset preset) {
        return switch (preset) {
            case REGRESSION_MSE_ADAMW,
                    REGRESSION_HUBER_ADAMW,
                    REGRESSION_PINBALL_ADAMW,
                    REGRESSION_INTERVAL_SCORE_ADAMW,
                    REGRESSION_GAUSSIAN_NLL_ADAMW,
                    REGRESSION_POISSON_NLL_ADAMW,
                    REGRESSION_TWEEDIE_NLL_ADAMW,
                    REGRESSION_NEGATIVE_BINOMIAL_NLL_ADAMW,
                    REGRESSION_ZERO_INFLATED_POISSON_NLL_ADAMW,
                    REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_ADAMW,
                    CLASSIFICATION_CROSS_ENTROPY_ADAMW,
                    CLASSIFICATION_FOCAL_ADAMW,
                    CAUSAL_LANGUAGE_MODELING_ADAMW,
                    CAUSAL_LM_ADAMW,
                    BINARY_FOCAL_WITH_LOGITS_ADAMW,
                    BINARY_BCE_WITH_LOGITS_ADAMW ->
                AdamW.builder(model.parameters(), learningRate).build();
            case REGRESSION_MSE_SGD,
                    REGRESSION_HUBER_SGD,
                    REGRESSION_PINBALL_SGD,
                    REGRESSION_INTERVAL_SCORE_SGD,
                    REGRESSION_GAUSSIAN_NLL_SGD,
                    REGRESSION_POISSON_NLL_SGD,
                    REGRESSION_TWEEDIE_NLL_SGD,
                    REGRESSION_NEGATIVE_BINOMIAL_NLL_SGD,
                    REGRESSION_ZERO_INFLATED_POISSON_NLL_SGD,
                    REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_SGD,
                    CLASSIFICATION_CROSS_ENTROPY_SGD,
                    CLASSIFICATION_FOCAL_SGD,
                    CAUSAL_LANGUAGE_MODELING_SGD,
                    CAUSAL_LM_SGD,
                    BINARY_FOCAL_WITH_LOGITS_SGD,
                    BINARY_BCE_WITH_LOGITS_SGD ->
                SGD.builder(model.parameters(), learningRate).momentum(0.9f).build();
        };
    }

    static TrainingLossFunction presetLoss(Gollek.DL.TrainingPreset preset) {
        return presetLoss(preset, null, null, null, null, null, null, null, null, null, null, null,
                null, null, null, null, null, null, null, null, null, null);
    }

    static List<Supplier<? extends TrainingMetric>> defaultMetricFactories(
            Gollek.DL.TrainingPreset preset,
            Float causalLanguageModelingIgnoreIndex) {
        if (isClassificationPreset(preset)) {
            return List.of(
                    TrainingMetrics.classificationAccuracy(),
                    TrainingMetrics.classificationLogLoss());
        }
        if (isBinaryPreset(preset)) {
            return List.of(
                    TrainingMetrics.binaryAccuracy(),
                    TrainingMetrics.binaryLogLoss());
        }
        if (isSimpleRegressionPreset(preset)) {
            return List.of(
                    TrainingMetrics.meanAbsoluteError(),
                    TrainingMetrics.meanSquaredError(),
                    TrainingMetrics.rootMeanSquaredError());
        }
        if (!isCausalLanguageModelingPreset(preset)) {
            return List.of();
        }

        float ignoreIndex = causalLanguageModelingIgnoreIndex == null
                ? CausalLanguageModelingLoss.DEFAULT_IGNORE_INDEX
                : causalLanguageModelingIgnoreIndex;
        return List.of(
                TrainingMetrics.causalLanguageModelingTokenAccuracy(ignoreIndex),
                TrainingMetrics.causalLanguageModelingLogLoss(ignoreIndex),
                TrainingMetrics.causalLanguageModelingPerplexity(ignoreIndex));
    }

    static boolean isCausalLanguageModelingPreset(Gollek.DL.TrainingPreset preset) {
        return switch (preset) {
            case CAUSAL_LANGUAGE_MODELING_ADAMW,
                    CAUSAL_LANGUAGE_MODELING_SGD,
                    CAUSAL_LM_ADAMW,
                    CAUSAL_LM_SGD -> true;
            default -> false;
        };
    }

    static boolean isClassificationPreset(Gollek.DL.TrainingPreset preset) {
        return switch (preset) {
            case CLASSIFICATION_CROSS_ENTROPY_ADAMW,
                    CLASSIFICATION_CROSS_ENTROPY_SGD,
                    CLASSIFICATION_FOCAL_ADAMW,
                    CLASSIFICATION_FOCAL_SGD -> true;
            default -> false;
        };
    }

    static boolean isBinaryPreset(Gollek.DL.TrainingPreset preset) {
        return switch (preset) {
            case BINARY_BCE_WITH_LOGITS_ADAMW,
                    BINARY_BCE_WITH_LOGITS_SGD,
                    BINARY_FOCAL_WITH_LOGITS_ADAMW,
                    BINARY_FOCAL_WITH_LOGITS_SGD -> true;
            default -> false;
        };
    }

    static boolean isSimpleRegressionPreset(Gollek.DL.TrainingPreset preset) {
        return switch (preset) {
            case REGRESSION_MSE_ADAMW,
                    REGRESSION_MSE_SGD,
                    REGRESSION_HUBER_ADAMW,
                    REGRESSION_HUBER_SGD -> true;
            default -> false;
        };
    }

    static TrainingLossFunction presetLoss(
            Gollek.DL.TrainingPreset preset,
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
        return switch (preset) {
            case REGRESSION_MSE_ADAMW, REGRESSION_MSE_SGD -> {
                MSELoss mse = new MSELoss();
                yield mse::compute;
            }
            case REGRESSION_HUBER_ADAMW, REGRESSION_HUBER_SGD -> {
                HuberLoss huber = new HuberLoss();
                yield huber::compute;
            }
            case REGRESSION_PINBALL_ADAMW, REGRESSION_PINBALL_SGD -> {
                PinballLoss pinball = pinballQuantiles == null
                        ? new PinballLoss()
                        : new PinballLoss(pinballQuantiles);
                yield pinball::compute;
            }
            case REGRESSION_INTERVAL_SCORE_ADAMW, REGRESSION_INTERVAL_SCORE_SGD -> {
                PredictionIntervalLoss interval = intervalAlpha == null
                        ? new PredictionIntervalLoss()
                        : intervalCrossingPenalty == null
                                ? new PredictionIntervalLoss(intervalAlpha)
                                : new PredictionIntervalLoss(intervalAlpha, intervalCrossingPenalty);
                yield interval::compute;
            }
            case REGRESSION_GAUSSIAN_NLL_ADAMW, REGRESSION_GAUSSIAN_NLL_SGD -> {
                GaussianNllLoss gaussian = new GaussianNllLoss();
                yield gaussian::compute;
            }
            case REGRESSION_POISSON_NLL_ADAMW, REGRESSION_POISSON_NLL_SGD -> {
                PoissonNllLoss poisson = new PoissonNllLoss();
                yield poisson::compute;
            }
            case REGRESSION_TWEEDIE_NLL_ADAMW, REGRESSION_TWEEDIE_NLL_SGD -> {
                double power = tweediePower == null ? TweedieNllLoss.DEFAULT_POWER : tweediePower;
                boolean logInput = tweedieLogInput == null || tweedieLogInput;
                TweedieNllLoss tweedie = tweedieEps == null
                        ? new TweedieNllLoss(power, logInput)
                        : new TweedieNllLoss(power, logInput, tweedieEps);
                yield tweedie::compute;
            }
            case REGRESSION_NEGATIVE_BINOMIAL_NLL_ADAMW, REGRESSION_NEGATIVE_BINOMIAL_NLL_SGD -> {
                boolean logInput = negativeBinomialLogInput == null || negativeBinomialLogInput;
                boolean includeConstant = negativeBinomialIncludeConstant != null
                        && negativeBinomialIncludeConstant;
                NegativeBinomialNllLoss negativeBinomial = negativeBinomialEps == null
                        ? new NegativeBinomialNllLoss(logInput, includeConstant)
                        : new NegativeBinomialNllLoss(logInput, includeConstant, negativeBinomialEps);
                yield negativeBinomial::compute;
            }
            case REGRESSION_ZERO_INFLATED_POISSON_NLL_ADAMW,
                    REGRESSION_ZERO_INFLATED_POISSON_NLL_SGD -> {
                boolean logRateInput = zeroInflatedPoissonLogRateInput == null
                        || zeroInflatedPoissonLogRateInput;
                boolean includeConstant = zeroInflatedPoissonIncludeConstant != null
                        && zeroInflatedPoissonIncludeConstant;
                ZeroInflatedPoissonNllLoss zeroInflatedPoisson = zeroInflatedPoissonEps == null
                        ? new ZeroInflatedPoissonNllLoss(logRateInput, includeConstant)
                        : new ZeroInflatedPoissonNllLoss(logRateInput, includeConstant, zeroInflatedPoissonEps);
                yield zeroInflatedPoisson::compute;
            }
            case REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_ADAMW,
                    REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_SGD -> {
                boolean logInput = zeroInflatedNegativeBinomialLogInput == null
                        || zeroInflatedNegativeBinomialLogInput;
                boolean includeConstant = zeroInflatedNegativeBinomialIncludeConstant != null
                        && zeroInflatedNegativeBinomialIncludeConstant;
                ZeroInflatedNegativeBinomialNllLoss zeroInflatedNegativeBinomial =
                        zeroInflatedNegativeBinomialEps == null
                                ? new ZeroInflatedNegativeBinomialNllLoss(logInput, includeConstant)
                                : new ZeroInflatedNegativeBinomialNllLoss(
                                        logInput,
                                        includeConstant,
                                        zeroInflatedNegativeBinomialEps);
                yield zeroInflatedNegativeBinomial::compute;
            }
            case CLASSIFICATION_CROSS_ENTROPY_ADAMW, CLASSIFICATION_CROSS_ENTROPY_SGD -> {
                CrossEntropyLoss crossEntropy = crossEntropyClassWeights == null
                        ? new CrossEntropyLoss()
                        : new CrossEntropyLoss(crossEntropyClassWeights);
                yield crossEntropy::compute;
            }
            case CLASSIFICATION_FOCAL_ADAMW, CLASSIFICATION_FOCAL_SGD -> {
                float gamma = focalGamma == null ? 2.0f : focalGamma;
                FocalLoss focal = focalClassWeights != null
                        ? new FocalLoss(gamma, focalClassWeights)
                        : new FocalLoss(gamma, focalAlpha == null ? 0.25f : focalAlpha);
                yield focal::compute;
            }
            case CAUSAL_LANGUAGE_MODELING_ADAMW,
                    CAUSAL_LANGUAGE_MODELING_SGD,
                    CAUSAL_LM_ADAMW,
                    CAUSAL_LM_SGD -> {
                CausalLanguageModelingLoss causalLm = causalLanguageModelingIgnoreIndex == null
                        ? new CausalLanguageModelingLoss()
                        : new CausalLanguageModelingLoss(causalLanguageModelingIgnoreIndex);
                yield causalLm::compute;
            }
            case BINARY_FOCAL_WITH_LOGITS_ADAMW, BINARY_FOCAL_WITH_LOGITS_SGD -> {
                float gamma = focalGamma == null ? 2.0f : focalGamma;
                float alpha = focalAlpha == null ? 0.25f : focalAlpha;
                BinaryFocalWithLogitsLoss focal = bcePositiveWeights == null
                        ? new BinaryFocalWithLogitsLoss(gamma, alpha)
                        : new BinaryFocalWithLogitsLoss(gamma, alpha, bcePositiveWeights);
                yield focal::compute;
            }
            case BINARY_BCE_WITH_LOGITS_ADAMW, BINARY_BCE_WITH_LOGITS_SGD -> {
                BCEWithLogitsLoss bce = bcePositiveWeights == null
                        ? new BCEWithLogitsLoss()
                        : new BCEWithLogitsLoss(bcePositiveWeights);
                yield bce::compute;
            }
        };
    }
}
