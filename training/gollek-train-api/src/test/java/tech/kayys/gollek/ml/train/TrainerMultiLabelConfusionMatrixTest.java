package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

class TrainerMultiLabelConfusionMatrixTest {

    @Test
    void evaluationCarriesMultilabelConfusionMatrixDetails() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        2.0f, -1.0f, 0.5f,
                        -2.0f, 3.0f, -0.5f,
                        1.0f, 2.0f, -3.0f
                }, 3, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 0, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelConfusionMatrixMetric());

        assertEquals(3, summary.sampleCount());
        assertEquals(7.0 / 9.0, summary.metric("multilabel_confusion_matrix_macro_f1"), 1e-6);

        Map<String, Object> details = metricDetails(summary.metricDetails(), "multilabel_confusion_matrix_macro_f1");
        assertEquals("multilabel_confusion_matrix", details.get("type"));
        assertEquals(0.8, (Double) details.get("microPrecision"), 1e-6);
        assertEquals(0.8, (Double) details.get("microRecall"), 1e-6);
        assertEquals(0.8, (Double) details.get("microF1"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("samplePrecision"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("sampleRecall"), 1e-6);
        assertEquals(7.0 / 9.0, (Double) details.get("sampleF1"), 1e-6);
        assertEquals(2.0 / 3.0, (Double) details.get("sampleJaccard"), 1e-6);
        assertEquals(List.of(
                List.of(List.of(1L, 0L), List.of(0L, 2L)),
                List.of(List.of(1L, 1L), List.of(0L, 1L)),
                List.of(List.of(1L, 0L), List.of(1L, 1L))),
                details.get("matrixByLabel"));
        assertEquals(details, metricDetails(summary.metadata(), "metricDetails", "multilabel_confusion_matrix_macro_f1"));
    }

    @Test
    void evaluationReportsMultilabelMicroMetricsFromDlFacade() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        2.0f, -1.0f, 0.5f,
                        -2.0f, 3.0f, -0.5f,
                        1.0f, 2.0f, -3.0f
                }, 3, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 0, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelMicroPrecisionMetric(),
                Gollek.DL.multiLabelMicroRecallMetric(),
                Gollek.DL.multiLabelMicroF1Metric());

        assertEquals(0.8, summary.metric("multilabel_micro_precision"), 1e-6);
        assertEquals(0.8, summary.metric("multilabel_micro_recall"), 1e-6);
        assertEquals(0.8, summary.metric("multilabel_micro_f1"), 1e-6);
    }

    @Test
    void evaluationReportsMultilabelSampleMetricsFromDlFacade() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        2.0f, -1.0f, 0.5f,
                        -2.0f, 3.0f, -0.5f,
                        1.0f, 2.0f, -3.0f
                }, 3, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 0, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelSamplePrecisionMetric(),
                Gollek.DL.multiLabelSampleRecallMetric(),
                Gollek.DL.multiLabelSampleF1Metric(),
                Gollek.DL.multiLabelSampleJaccardMetric());

        assertEquals(5.0 / 6.0, summary.metric("multilabel_sample_precision"), 1e-6);
        assertEquals(5.0 / 6.0, summary.metric("multilabel_sample_recall"), 1e-6);
        assertEquals(7.0 / 9.0, summary.metric("multilabel_sample_f1"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("multilabel_sample_jaccard"), 1e-6);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metricDetails(Map<String, Object> detailMap, String metricName) {
        Object value = detailMap.get(metricName);
        if (value instanceof Map<?, ?> details) {
            return (Map<String, Object>) details;
        }
        throw new AssertionError("missing metric details for " + metricName);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metricDetails(
            Map<String, Object> metadata,
            String detailsKey,
            String metricName) {
        Object value = metadata.get(detailsKey);
        if (value instanceof Map<?, ?> detailMap) {
            return metricDetails((Map<String, Object>) detailMap, metricName);
        }
        throw new AssertionError("missing metadata details map " + detailsKey);
    }

    private static final class IdentityModel extends NNModule {
        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }
}
