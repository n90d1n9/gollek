package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetMetadataSupportTest {
    @Test
    void metadataValueMatchesAllowsExtraMapKeysByDefault() {
        Map<String, Object> expected = Map.of(
                "status",
                "accepted",
                "details",
                Map.of("count", 1, "codes", List.of("continue")));
        Map<String, Object> actual = Map.of(
                "status",
                "accepted",
                "details",
                Map.of("count", 1L, "codes", List.of("continue"), "derived", true),
                "summary",
                "ignored by subset matching");

        assertTrue(DiscreteTokenDatasetMetadataSupport.metadataValueMatches(expected, actual));
    }

    @Test
    void metadataValueMatchesCanRequireExactMapKeys() {
        Map<String, Object> expected = Map.of(
                "status",
                "accepted",
                "details",
                Map.of("count", 1));
        Map<String, Object> extraActual = Map.of(
                "status",
                "accepted",
                "details",
                Map.of("count", 1L, "derived", true));
        Map<String, Object> exactActual = Map.of(
                "status",
                "accepted",
                "details",
                Map.of("count", 1L));

        assertFalse(DiscreteTokenDatasetMetadataSupport.metadataValueMatches(expected, extraActual, true));
        assertTrue(DiscreteTokenDatasetMetadataSupport.metadataValueMatches(expected, exactActual, true));
    }

    @Test
    void metadataValueMatchesCanUseMissingAndMismatchPolicies() {
        Map<String, Object> expected = Map.of(
                "action",
                Map.of(
                        "actionHint",
                        "Continue training from the selected checkpoint.",
                        "primaryActionHint",
                        "Continue training from the selected checkpoint."));
        Map<String, Object> legacyActual = Map.of(
                "action",
                Map.of("actionHint", "continue"));

        assertFalse(DiscreteTokenDatasetMetadataSupport.metadataValueMatches(expected, legacyActual, true));
        assertTrue(DiscreteTokenDatasetMetadataSupport.metadataValueMatches(
                expected,
                legacyActual,
                true,
                key -> "primaryActionHint".equals(key),
                (key, expectedValue, actualValue) ->
                        "actionHint".equals(key) && "continue".equals(actualValue)));
    }
}
