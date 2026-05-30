package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.train.data.PrefetchingIterable;
import tech.kayys.gollek.trainer.api.TrainingSummary;

@SuppressWarnings("deprecation")
class TrainingLossFunctionTest {

    @Test
    void canonicalTrainerAcceptsTopLevelTrainingLossFunction() {
        MSELoss mseLoss = new MSELoss();
        TrainingLossFunction loss = mseLoss::compute;
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1.0f, 3.0f}, 2, 1),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2, 1)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(loss)
                .epochs(1)
                .build()) {
            trainer.fit(train, null);

            TrainingSummary summary = trainer.summary();
            assertNotNull(summary.latestTrainLoss());
            assertEquals(2.5, summary.latestTrainLoss(), 1e-6);
        }
    }

    @Test
    void legacyNestedLossFunctionRemainsAssignableToTopLevelContract() {
        CanonicalTrainer.LossFunction legacyLoss = (predictions, targets) -> GradTensor.scalar(0.25f);

        TrainingLossFunction loss = legacyLoss;

        assertEquals(0.25f, loss.compute(
                GradTensor.of(new float[] {1.0f}, 1),
                GradTensor.of(new float[] {0.0f}, 1)).data()[0], 1e-6);
    }

    @Test
    void gollekDlPinballPresetUsesConfiguredQuantileLoss() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1.0f, 3.0f}, 2, 1),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2, 1)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_PINBALL_SGD,
                Gollek.DL.trainingOptions()
                        .pinballQuantile(0.9)
                        .device("cpu")
                        .build());

        assertEquals(0.55, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlIntervalScorePresetUsesConfiguredCoverageLoss() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(
                        new float[] {
                                0.0f, 2.0f,
                                2.0f, 3.0f,
                                5.0f, 4.0f
                        },
                        3, 2),
                GradTensor.of(new float[] {1.0f, 1.0f, 4.5f}, 3)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_INTERVAL_SCORE_SGD,
                Gollek.DL.trainingOptions()
                        .intervalScoreAlpha(0.2)
                        .intervalCrossingPenalty(12.0)
                        .device("cpu")
                        .build());

        assertEquals(34.0 / 3.0, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlGaussianNllPresetTrainsMeanAndLogVarianceOutputs() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(
                        new float[] {
                                1.0f, 0.0f,
                                3.0f, 0.6931472f
                        },
                        2, 2),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_GAUSSIAN_NLL_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(0.9232868, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlClassificationPresetAutoRegistersDefaultMetrics() {
        float[] logits = {
                3.0f, 0.0f,
                0.0f, 3.0f
        };
        List<Batch> train = List.of(new Batch(
                GradTensor.of(logits, 2, 2),
                GradTensor.of(new float[] {0.0f, 1.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        Map<String, Double> latestTrainMetrics = metricMap(summary, "latestTrainMetrics");
        assertEquals(2, latestTrainMetrics.size());
        assertEquals(1.0, latestTrainMetrics.get("accuracy"), 1e-6);
        assertEquals(nll(logits, 0, 2, 0), latestTrainMetrics.get("classification_log_loss"), 1e-6);
    }

    @Test
    void gollekDlBinaryPresetAutoRegistersDefaultMetrics() {
        float[] logits = {2.0f, -2.0f, 0.5f, -0.5f};
        float[] labels = {1.0f, 0.0f, 1.0f, 0.0f};
        List<Batch> train = List.of(new Batch(
                GradTensor.of(logits, 4),
                GradTensor.of(labels, 4)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        Map<String, Double> latestTrainMetrics = metricMap(summary, "latestTrainMetrics");
        assertEquals(2, latestTrainMetrics.size());
        assertEquals(1.0, latestTrainMetrics.get("binary_accuracy"), 1e-6);
        assertEquals(binaryLogLoss(logits, labels), latestTrainMetrics.get("binary_log_loss"), 1e-6);
    }

    @Test
    void gollekDlSimpleRegressionPresetAutoRegistersDefaultMetrics() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1.0f, 3.0f}, 2, 1),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2, 1)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        Map<String, Double> latestTrainMetrics = metricMap(summary, "latestTrainMetrics");
        assertEquals(3, latestTrainMetrics.size());
        assertEquals(1.5, latestTrainMetrics.get("mae"), 1e-6);
        assertEquals(2.5, latestTrainMetrics.get("mse"), 1e-6);
        assertEquals(Math.sqrt(2.5), latestTrainMetrics.get("rmse"), 1e-6);
    }

    @Test
    void gollekDlFitClosesPrefetchedTrainLoaderAfterTraining() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1.0f, 3.0f}, 2, 1),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2, 1)));
        PrefetchingIterable<Batch> prefetched = DataLoader.prefetch(train, 1);

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                prefetched,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(2.5, summary.latestTrainLoss(), 1e-6);
        assertThrows(IllegalStateException.class, prefetched::iterator);
    }

    @Test
    void gollekDlPoissonNllPresetTrainsLogRateOutputs() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {0.0f, 0.6931472f}, 2),
                GradTensor.of(new float[] {1.0f, 3.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_POISSON_NLL_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(0.46027923, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlTweedieNllPresetUsesConfiguredPower() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {0.0f, 1.3862944f}, 2),
                GradTensor.of(new float[] {1.0f, 2.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_TWEEDIE_NLL_SGD,
                Gollek.DL.trainingOptions()
                        .tweediePower(1.5)
                        .device("cpu")
                        .build());

        assertEquals(5.0, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlNegativeBinomialNllPresetSupportsFullCountLikelihood() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {(float) Math.log(2.0), 0.0f}, 1, 2),
                GradTensor.of(new float[] {2.0f}, 1)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_NEGATIVE_BINOMIAL_NLL_SGD,
                Gollek.DL.trainingOptions()
                        .negativeBinomialFullNll()
                        .device("cpu")
                        .build());

        assertEquals(1.9095426, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlZeroInflatedPoissonNllPresetSupportsExcessZeroCounts() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(
                        new float[] {
                                0.0f, 0.0f,
                                (float) Math.log(2.0), 0.0f
                        },
                        2, 2),
                GradTensor.of(new float[] {0.0f, 2.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_ZERO_INFLATED_POISSON_NLL_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .build());

        assertEquals(0.8433691, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlZeroInflatedNegativeBinomialNllPresetSupportsOverdispersedExcessZeroCounts() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(
                        new float[] {
                                (float) Math.log(2.0), 0.0f, 0.0f,
                                (float) Math.log(2.0), 0.0f, 0.0f
                        },
                        2, 3),
                GradTensor.of(new float[] {0.0f, 2.0f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_ZERO_INFLATED_NEGATIVE_BINOMIAL_NLL_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .build());

        assertEquals(1.1575038, summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlCausalLanguageModelingPresetUsesConfiguredIgnoreIndexAndMetrics() {
        float[] logits = {
                3.0f, 1.0f, 0.0f,
                -2.0f, 0.0f, 2.0f
        };
        List<Batch> train = List.of(new Batch(
                GradTensor.of(logits, 1, 2, 3),
                GradTensor.of(new float[] {0.0f, -1.0f}, 1, 2)));

        Gollek.DL.TrainingOptions options = Gollek.DL.causalLanguageModelingTrainingOptions(-1.0f)
                .device("cpu")
                .build();

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CAUSAL_LM_SGD,
                options);

        assertEquals(-1.0f, options.causalLanguageModelingIgnoreIndex());
        assertEquals(3, options.metricFactories().size());
        assertEquals("causal_lm_token_accuracy", options.metricFactories().get(0).get().name());
        assertEquals("causal_lm_log_loss", options.metricFactories().get(1).get().name());
        assertEquals("causal_lm_perplexity", options.metricFactories().get(2).get().name());
        assertEquals(nll(logits, 0, 3, 0), summary.latestTrainLoss(), 1e-6);
    }

    @Test
    void gollekDlCausalLanguageModelingPresetAutoRegistersDefaultMetrics() {
        float[] logits = {
                3.0f, 1.0f, 0.0f,
                -2.0f, 0.0f, 2.0f
        };
        List<Batch> train = List.of(new Batch(
                GradTensor.of(logits, 1, 2, 3),
                GradTensor.of(new float[] {0.0f, -1.0f}, 1, 2)));

        Gollek.DL.TrainingOptions options = Gollek.DL.trainingOptions()
                .causalLanguageModelingIgnoreIndex(-1.0f)
                .device("cpu")
                .build();

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CAUSAL_LANGUAGE_MODELING_SGD,
                options);

        assertEquals(0, options.metricFactories().size());
        double expectedLogLoss = nll(logits, 0, 3, 0);
        Map<String, Double> latestTrainMetrics = metricMap(summary, "latestTrainMetrics");
        assertEquals(3, latestTrainMetrics.size());
        assertEquals(1.0, latestTrainMetrics.get("causal_lm_token_accuracy"), 1e-6);
        assertEquals(expectedLogLoss, latestTrainMetrics.get("causal_lm_log_loss"), 1e-6);
        assertEquals(Math.exp(expectedLogLoss), latestTrainMetrics.get("causal_lm_perplexity"), 1e-6);
    }

    @Test
    void gollekDlCausalLanguageModelingPresetRejectsInvalidIgnoreIndex() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Gollek.DL.trainingOptions().causalLanguageModelingIgnoreIndex(Float.NaN));
    }

    private static final class IdentityModel extends NNModule {
        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }

    private static double nll(float[] logits, int offset, int length, int target) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            max = Math.max(max, logits[offset + i]);
        }
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += Math.exp(logits[offset + i] - max);
        }
        return max + Math.log(sum) - logits[offset + target];
    }

    private static double binaryLogLoss(float[] logits, float[] labels) {
        double total = 0.0;
        for (int i = 0; i < logits.length; i++) {
            double x = logits[i];
            total += Math.max(x, 0.0) - (x * labels[i]) + Math.log1p(Math.exp(-Math.abs(x)));
        }
        return total / logits.length;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Double> metricMap(TrainingSummary summary, String key) {
        return (Map<String, Double>) summary.metadata().get(key);
    }
}
