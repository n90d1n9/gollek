package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

@SuppressWarnings("deprecation")
class CausalLanguageModelingTrainingMetricsTest {

    @Test
    void tokenAccuracyIgnoresMaskedLanguageModelingLabels() {
        TrainingMetric metric = TrainingMetrics.causalLanguageModelingTokenAccuracy().get();

        metric.update(
                GradTensor.of(new float[] {
                        3.0f, 1.0f, 0.0f,
                        2.0f, 3.0f, 1.0f,
                        Float.NaN, Float.NaN, Float.NaN
                }, 1, 3, 3),
                GradTensor.of(new float[] {
                        0.0f,
                        2.0f,
                        -100.0f
                }, 1, 3));

        assertEquals("causal_lm_token_accuracy", metric.name());
        assertEquals(0.5, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("causal_lm_token_accuracy", details.get("type"));
        assertEquals("causal_language_modeling", details.get("task"));
        assertEquals(3, details.get("vocabularySize"));
        assertEquals(3L, details.get("totalPositions"));
        assertEquals(2L, details.get("validTokens"));
        assertEquals(1L, details.get("ignoredTokens"));
        assertEquals(1L, details.get("correctTokens"));
        assertEquals(0.5, (Double) details.get("accuracy"), 1e-6);
        assertEquals(-100.0f, (Float) details.get("ignoreIndex"));
        assertEquals("logits", details.get("input"));
        assertEquals("token_ids", details.get("targetEncoding"));
    }

    @Test
    void logLossStreamsStableMeanNegativeLogLikelihood() {
        TrainingMetric metric = TrainingMetrics.causalLanguageModelingLogLoss().get();
        float[] logits = {
                3.0f, 1.0f, 0.0f,
                2.0f, 3.0f, 1.0f,
                7.0f, -2.0f, 5.0f
        };

        metric.update(
                GradTensor.of(logits, 1, 3, 3),
                GradTensor.of(new float[] {
                        0.0f,
                        2.0f,
                        -100.0f
                }, 1, 3));

        double expected = (nll(logits, 0, 3, 0) + nll(logits, 3, 3, 2)) / 2.0;
        assertEquals("causal_lm_log_loss", metric.name());
        assertEquals(expected, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("causal_lm_log_loss", details.get("type"));
        assertEquals(2L, details.get("validTokens"));
        assertEquals(1L, details.get("ignoredTokens"));
        assertEquals(expected, (Double) details.get("logLoss"), 1e-6);
        assertEquals(expected, (Double) details.get("crossEntropy"), 1e-6);
        assertEquals(expected, (Double) details.get("negativeLogLikelihood"), 1e-6);
        assertEquals(Math.exp(expected), (Double) details.get("perplexity"), 1e-6);
        assertEquals(
                (Math.exp(-nll(logits, 0, 3, 0)) + Math.exp(-nll(logits, 3, 3, 2))) / 2.0,
                (Double) details.get("meanCorrectTokenProbability"),
                1e-6);
    }

    @Test
    void perplexityReturnsExponentiatedCausalLanguageModelingLoss() {
        TrainingMetric metric = TrainingMetrics.causalLanguageModelingPerplexity().get();
        float[] logits = {
                1.0f, 0.0f,
                -1.0f, 2.0f
        };

        metric.update(
                GradTensor.of(logits, 1, 2, 2),
                GradTensor.of(new float[] {0.0f, 1.0f}, 1, 2));

        double logLoss = (nll(logits, 0, 2, 0) + nll(logits, 2, 2, 1)) / 2.0;
        assertEquals("causal_lm_perplexity", metric.name());
        assertEquals(Math.exp(logLoss), metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        assertEquals("exp(logLoss)", detailed.details().get("valueMeaning"));
        assertEquals(logLoss, (Double) detailed.details().get("logLoss"), 1e-6);
    }

    @Test
    void customIgnoreIndexIsAvailableThroughGollekFacadeAndTrainingOptions() {
        CanonicalTrainer.Metric metric = Gollek.DL.nextTokenAccuracyMetric(-1.0f).get();
        metric.update(
                GradTensor.of(new float[] {
                        Float.NaN, Float.NaN,
                        0.0f, 4.0f
                }, 1, 2, 2),
                GradTensor.of(new float[] {-1.0f, 1.0f}, 1, 2));

        assertEquals("causal_lm_token_accuracy", metric.name());
        assertEquals(1.0, metric.value(), 1e-6);

        CanonicalTrainer.DetailedMetric detailed =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals(-1.0f, (Float) detailed.details().get("ignoreIndex"));
        assertEquals(1L, detailed.details().get("ignoredTokens"));

        Gollek.DL.TrainingOptions options = Gollek.DL.trainingOptions()
                .nextTokenMetrics(-1.0f)
                .build();
        assertEquals(3, options.metricFactories().size());
        assertEquals("causal_lm_token_accuracy", options.metricFactories().get(0).get().name());
        assertEquals("causal_lm_log_loss", options.metricFactories().get(1).get().name());
        assertEquals("causal_lm_perplexity", options.metricFactories().get(2).get().name());
    }

    @Test
    void allIgnoredLabelsReportNoValidTokens() {
        TrainingMetric metric = TrainingMetrics.causalLanguageModelingLogLoss().get();

        metric.update(
                GradTensor.of(new float[] {
                        Float.NaN, Float.NaN,
                        Float.NaN, Float.NaN
                }, 1, 2, 2),
                GradTensor.of(new float[] {-100.0f, -100.0f}, 1, 2));

        assertTrue(Double.isNaN(metric.value()));
        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        assertEquals(0L, detailed.details().get("validTokens"));
        assertEquals(2L, detailed.details().get("ignoredTokens"));
        assertNull(detailed.details().get("logLoss"));
        assertNull(detailed.details().get("perplexity"));
    }

    @Test
    void rejectsInvalidShapesTargetsAndLogits() {
        TrainingMetric metric = TrainingMetrics.nextTokenAccuracy().get();

        assertThrows(IllegalArgumentException.class, () -> metric.update(
                GradTensor.of(new float[] {1.0f, 2.0f}, 1, 2),
                GradTensor.of(new float[] {0.0f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> metric.update(
                GradTensor.of(new float[] {1.0f, 2.0f}, 1, 1, 2),
                GradTensor.of(new float[] {2.0f}, 1, 1)));
        assertThrows(IllegalArgumentException.class, () -> metric.update(
                GradTensor.of(new float[] {Float.POSITIVE_INFINITY, 0.0f}, 1, 1, 2),
                GradTensor.of(new float[] {0.0f}, 1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> TrainingMetrics.nextTokenAccuracy(Float.NaN));
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
}
