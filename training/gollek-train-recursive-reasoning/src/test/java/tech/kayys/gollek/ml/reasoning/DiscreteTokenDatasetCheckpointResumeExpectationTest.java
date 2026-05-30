package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumeExpectationTest {
    @Test
    void inactiveExpectationAcceptsAnyCheckpoint() {
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot = snapshot();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.none();

        assertFalse(expectation.active());
        assertTrue(expectation.accepts(snapshot));
        assertTrue(expectation.rejectionReasons(snapshot).isEmpty());
        assertEquals(Map.of("active", false), expectation.toMetadata());
    }

    @Test
    void builderAcceptsMatchingCheckpointIdentity() {
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot = snapshot();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("gram-nqueens")
                        .runId("run-001")
                        .modelFamily("gram")
                        .seed(2026L)
                        .checkpointStep(12L)
                        .minimumCheckpointStep(10L)
                        .build();

        assertTrue(expectation.active());
        assertTrue(expectation.accepts(snapshot));
        assertEquals("gram-nqueens", expectation.toMetadata().get("experimentName"));
        assertEquals(12L, expectation.toMetadata().get("checkpointStep"));
        assertEquals(10L, expectation.toMetadata().get("minimumCheckpointStep"));
    }

    @Test
    void roundTripsMetadataAndJsonLikeScalars() {
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("gram-nqueens")
                        .runId("run-001")
                        .modelFamily("gram")
                        .seed(2026L)
                        .checkpointStep(12L)
                        .minimumCheckpointStep(10L)
                        .build();

        DiscreteTokenDatasetCheckpointResumeExpectation parsed =
                DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(expectation.toMetadata());

        assertEquals(expectation, parsed);
        assertEquals(expectation.toMetadata(), parsed.toMetadata());

        Map<String, Object> jsonLike = new java.util.LinkedHashMap<>(expectation.toMetadata());
        jsonLike.put("seed", "2026");
        jsonLike.put("checkpointStep", "12");
        jsonLike.put("minimumCheckpointStep", "10");

        assertEquals(expectation, DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(jsonLike));
        assertThrows(UnsupportedOperationException.class, () -> expectation.toMetadata().put("bad", "value"));
    }

    @Test
    void reportsAllIdentityMismatches() {
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot = snapshot();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("other-experiment")
                        .runId("run-999")
                        .modelFamily("other-family")
                        .seed(7L)
                        .checkpointStep(99L)
                        .minimumCheckpointStep(20L)
                        .build();

        List<String> reasons = expectation.rejectionReasons(snapshot);

        assertFalse(expectation.accepts(snapshot));
        assertEquals(6, reasons.size());
        assertTrue(reasons.get(0).contains("experimentName expected other-experiment"));
        assertTrue(reasons.get(1).contains("runId expected run-999"));
        assertTrue(reasons.get(2).contains("modelFamily expected other-family"));
        assertTrue(reasons.get(3).contains("seed expected 7"));
        assertTrue(reasons.get(4).contains("checkpointStep expected 99"));
        assertTrue(reasons.get(5).contains("checkpointStep expected >= 20"));
    }

    @Test
    void createsExactExpectationsFromManifestAndSnapshot() {
        DiscreteTokenDatasetCheckpointManifest manifest = manifest();
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);

        assertTrue(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest).accepts(snapshot));
        assertTrue(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromSnapshot(snapshot).accepts(snapshot));
    }

    @Test
    void rejectsMalformedExpectationValues() {
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .runId(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .modelFamily(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .seed(-1L)
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .checkpointStep(4L)
                        .minimumCheckpointStep(5L)
                        .build());
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(
                        Map.of("experimentName", 7)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(
                        Map.of("seed", "not-a-number")));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExpectation.none().accepts(null));
    }

    private static DiscreteTokenDatasetCheckpointManifestSnapshot snapshot() {
        return DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest());
    }

    private static DiscreteTokenDatasetCheckpointManifest manifest() {
        return DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                .experimentName("gram-nqueens")
                .runId("run-001")
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(1_900_000_000_000L)
                .createdBy("trainer-test")
                .build();
    }

    private static DiscreteTokenDatasetPlan cleanPlan() {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("graph-coloring", 0, 1),
                        knownExample("graph-coloring", 1, 2),
                        knownExample("graph-coloring", 2, 1),
                        knownExample("graph-coloring", 3, 2),
                        knownExample("nqueens", 10, 1),
                        knownExample("nqueens", 11, 2),
                        knownExample("nqueens", 12, 1),
                        knownExample("nqueens", 13, 2)),
                new DiscreteTokenDatasetPlanConfig(
                        0.25d,
                        0.25d,
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        0L,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                        0L,
                        false));
    }

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                1,
                Map.of("inputLength", inputLength));
    }
}
