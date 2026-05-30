package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

@SuppressWarnings("deprecation")
class ClassificationTrainingMetricsTest {

    @Test
    void classificationLogLossStreamsStableCrossEntropyFromLogits() {
        TrainingMetric metric = TrainingMetrics.classificationLogLoss().get();

        metric.update(
                GradTensor.of(new float[] {
                        logProb(0.8), logProb(0.1), logProb(0.1),
                        logProb(0.4), logProb(0.5), logProb(0.1)
                }, 2, 3),
                GradTensor.of(new float[] {0.0f, 1.0f}, 2));
        metric.update(
                GradTensor.of(new float[] {
                        logProb(0.2), logProb(0.3), logProb(0.5)
                }, 1, 3),
                GradTensor.of(new float[] {2.0f}, 1));

        double expected = (-Math.log(0.8) - Math.log(0.5) - Math.log(0.5)) / 3.0;
        assertEquals("classification_log_loss", metric.name());
        assertEquals(expected, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_log_loss", details.get("type"));
        assertEquals(3, details.get("classes"));
        assertEquals(3L, details.get("samples"));
        assertEquals(expected, (Double) details.get("logLoss"), 1e-6);
        assertEquals(expected, (Double) details.get("crossEntropy"), 1e-6);
        assertEquals(expected, (Double) details.get("negativeLogLikelihood"), 1e-6);
        assertEquals(Math.exp(expected), (Double) details.get("perplexity"), 1e-6);
        assertEquals(0.6, (Double) details.get("meanCorrectClassProbability"), 1e-6);
        assertEquals(-Math.log(0.5), (Double) details.get("maximumSampleLogLoss"), 1e-6);
        assertEquals("logits", details.get("input"));
    }

    @Test
    void classificationCrossEntropyAliasAcceptsOneHotTargetsThroughGollekFacade() {
        CanonicalTrainer.Metric metric = Gollek.DL.classificationCrossEntropyMetric().get();

        metric.update(
                GradTensor.of(new float[] {
                        logProb(0.7), logProb(0.2), logProb(0.1),
                        logProb(0.1), logProb(0.2), logProb(0.7)
                }, 2, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f
                }, 2, 3));

        CanonicalTrainer.DetailedMetric detailed =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals("classification_log_loss", metric.name());
        assertEquals(-Math.log(0.7), metric.value(), 1e-6);
        assertEquals("classification_log_loss", detailed.details().get("type"));
        assertEquals(2L, detailed.details().get("samples"));
    }

    @Test
    void classificationLogLossRejectsInvalidLogits() {
        TrainingMetric metric = TrainingMetrics.classificationLogLoss().get();

        assertThrows(IllegalArgumentException.class, () -> metric.update(
                GradTensor.of(new float[] {1.0f, Float.POSITIVE_INFINITY}, 1, 2),
                GradTensor.of(new float[] {0.0f}, 1)));
    }

    private static float logProb(double probability) {
        return (float) Math.log(probability);
    }
}
