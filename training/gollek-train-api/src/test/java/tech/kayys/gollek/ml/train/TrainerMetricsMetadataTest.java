package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerMetricsMetadataTest {

    @Test
    @SuppressWarnings("unchecked")
    void publishesMetricMapsAndFlattenedEntries() {
        Map<String, Object> metadata = new LinkedHashMap<>();

        TrainerMetricsMetadata.putLatest(
                metadata,
                true,
                Map.of("mae", 0.5),
                Map.of("accuracy", 0.75),
                Map.of("confusion", Map.of("tp", 2, "fp", 1)),
                Map.of("roc", List.of(0.1, 0.9)));

        assertEquals(Boolean.TRUE, metadata.get("metricsEnabled"));
        assertEquals(0.5, ((Map<String, Double>) metadata.get("latestTrainMetrics")).get("mae"), 1e-6);
        assertEquals(0.75, ((Map<String, Double>) metadata.get("latestValidationMetrics")).get("accuracy"), 1e-6);
        assertEquals(0.5, (Double) metadata.get("trainMetric.mae"), 1e-6);
        assertEquals(0.75, (Double) metadata.get("validationMetric.accuracy"), 1e-6);
        assertEquals(Map.of("tp", 2, "fp", 1), metadata.get("trainMetricDetails.confusion"));
        assertEquals(List.of(0.1, 0.9), metadata.get("validationMetricDetails.roc"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void snapshotsInputsBeforePublishing() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        Map<String, Double> trainMetrics = new LinkedHashMap<>();
        Map<String, Object> trainDetails = new LinkedHashMap<>();
        List<Double> curve = new ArrayList<>(List.of(0.9, 0.8));
        trainMetrics.put("loss", 1.25);
        trainDetails.put("curve", curve);

        TrainerMetricsMetadata.putLatest(
                metadata,
                false,
                trainMetrics,
                Map.of(),
                trainDetails,
                Map.of());

        trainMetrics.put("loss", 99.0);
        trainDetails.put("after", "mutation");
        curve.add(0.7);

        Map<String, Double> publishedMetrics = (Map<String, Double>) metadata.get("latestTrainMetrics");
        Map<String, Object> publishedDetails = (Map<String, Object>) metadata.get("latestTrainMetricDetails");
        assertEquals(Boolean.FALSE, metadata.get("metricsEnabled"));
        assertEquals(1.25, publishedMetrics.get("loss"), 1e-6);
        assertEquals(List.of(0.9, 0.8), publishedDetails.get("curve"));
        assertFalse(publishedDetails.containsKey("after"));
        assertThrows(UnsupportedOperationException.class, () -> publishedMetrics.put("new", 1.0));
        assertThrows(UnsupportedOperationException.class,
                () -> ((List<Double>) publishedDetails.get("curve")).add(0.6));
    }
}
