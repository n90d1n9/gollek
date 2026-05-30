package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerLossImprovementMetadataTest {

    @Test
    void publishesBestLossAndNonImprovingStreaks() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerLossImprovementMetadata.putLatest(
                metadata,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.4),
                        Map.of("epoch", 2, "trainLoss", 0.51, "validationLoss", 0.45),
                        Map.of("epoch", 3, "trainLoss", 0.49, "validationLoss", 0.43)));

        assertEquals(Boolean.TRUE, metadata.get("latestTrainLossBestAvailable"));
        assertEquals(0.49, metadata.get("latestTrainLossBest"));
        assertEquals(3, metadata.get("latestTrainLossBestEpoch"));
        assertEquals(0, metadata.get("latestTrainLossNonImprovingStreak"));
        assertEquals(Boolean.TRUE, metadata.get("latestTrainLossBestAtLatestEpoch"));
        assertEquals(Boolean.TRUE, metadata.get("latestValidationLossBestAvailable"));
        assertEquals(0.4, metadata.get("latestValidationLossBest"));
        assertEquals(1, metadata.get("latestValidationLossBestEpoch"));
        assertEquals(2, metadata.get("latestValidationLossNonImprovingStreak"));
        assertEquals(Boolean.FALSE, metadata.get("latestValidationLossBestAtLatestEpoch"));
    }

    @Test
    void marksUnavailableWhenHistoryHasNoFiniteLosses() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerLossImprovementMetadata.putLatest(
                metadata,
                List.of(Map.of("epoch", 0, "trainLoss", Double.NaN)));

        assertEquals(Boolean.FALSE, metadata.get("latestTrainLossBestAvailable"));
        assertEquals(Boolean.FALSE, metadata.get("latestValidationLossBestAvailable"));
    }
}
