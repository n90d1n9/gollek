package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetFingerprintMatchTest {
    @Test
    void acceptsMatchingFingerprints() {
        DiscreteTokenDatasetPlan plan = plan(11L, 2);
        DiscreteTokenDatasetFingerprint expected = plan.fingerprint();

        DiscreteTokenDatasetFingerprintMatch match = plan.verifyFingerprint(expected);

        assertTrue(match.matches());
        assertEquals("matched", match.status());
        assertTrue(match.mismatchReasons().isEmpty());
        assertEquals("dataset fingerprint matched: " + expected.shortValue(), match.summary());
        match.requireMatch();
    }

    @Test
    void reportsMismatchedPlanFingerprints() {
        DiscreteTokenDatasetPlan original = plan(11L, 2);
        DiscreteTokenDatasetPlan changed = plan(12L, 2);

        DiscreteTokenDatasetFingerprintMatch match =
                changed.verifyFingerprint(original.fingerprint());

        assertFalse(match.matches());
        assertEquals("mismatched", match.status());
        assertTrue(match.mismatchReasons().contains("fingerprint value differs"));
        assertTrue(match.summary().contains(original.fingerprint().shortValue()));
        assertTrue(match.summary().contains(changed.fingerprint().shortValue()));
        assertThrows(IllegalStateException.class, match::requireMatch);
    }

    @Test
    void reportsAlgorithmAndExampleCountMismatches() {
        DiscreteTokenDatasetFingerprint actual =
                new DiscreteTokenDatasetFingerprint("SHA-256", "abcdef", 2);
        DiscreteTokenDatasetFingerprint expected =
                new DiscreteTokenDatasetFingerprint("OTHER", "123456", 3);

        DiscreteTokenDatasetFingerprintMatch match =
                DiscreteTokenDatasetFingerprintMatch.verify(expected, actual);

        assertFalse(match.matches());
        assertEquals(List.of(
                "fingerprint algorithm differs",
                "fingerprint value differs",
                "fingerprint example count differs"), match.mismatchReasons());
    }

    @Test
    void reportHelperVerifiesAgainstReportFingerprint() {
        DiscreteTokenDatasetPlanReport report = plan(11L, 2).report();

        DiscreteTokenDatasetFingerprintMatch match =
                report.verifyFingerprint(report.fingerprint());

        assertTrue(match.matches());
        assertEquals(report.fingerprint(), match.actual());
    }

    @Test
    void verifiesStoredFingerprintMetadataAgainstPlansAndReports() {
        DiscreteTokenDatasetPlan plan = plan(11L, 2);
        DiscreteTokenDatasetPlanReport report = plan.report();
        Map<String, Object> checkpointFingerprint = report.fingerprint().toMetadata();
        Map<String, Object> checkpointReport = report.toMetadata();
        Map<String, Object> customCheckpointReport = Map.of("datasetFingerprint", checkpointFingerprint);

        assertTrue(plan.verifyFingerprintMetadata(checkpointFingerprint).matches());
        assertTrue(report.verifyFingerprintMetadata(checkpointFingerprint).matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch.verifyMetadata(checkpointFingerprint, plan).matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch.verifyMetadata(checkpointFingerprint, report).matches());
        assertTrue(plan.verifyFingerprintMetadataSection(checkpointReport).matches());
        assertTrue(report.verifyFingerprintMetadataSection(checkpointReport).matches());
        assertTrue(plan.verifyFingerprintMetadataSection(customCheckpointReport, "datasetFingerprint").matches());
        assertTrue(report.verifyFingerprintMetadataSection(customCheckpointReport, "datasetFingerprint").matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(checkpointReport, plan).matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(checkpointReport, report).matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch
                .verifyMetadataSection(customCheckpointReport, "datasetFingerprint", plan)
                .matches());
        assertTrue(DiscreteTokenDatasetFingerprintMatch
                .verifyMetadataSection(customCheckpointReport, "datasetFingerprint", report)
                .matches());
        assertFalse(plan(12L, 2).verifyFingerprintMetadata(checkpointFingerprint).matches());
        assertFalse(plan(12L, 2).verifyFingerprintMetadataSection(checkpointReport).matches());
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetPlan original = plan(11L, 2);
        DiscreteTokenDatasetPlan changed = plan(12L, 2);

        Map<String, Object> metadata =
                changed.verifyFingerprint(original.fingerprint()).toMetadata();

        assertEquals("mismatched", metadata.get("status"));
        assertEquals(false, metadata.get("matches"));
        assertEquals(
                original.fingerprint().value(),
                ((Map<?, ?>) metadata.get("expected")).get("value"));
        assertEquals(
                changed.fingerprint().value(),
                ((Map<?, ?>) metadata.get("actual")).get("value"));
        assertTrue(((List<?>) metadata.get("mismatchReasons")).contains("fingerprint value differs"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void rehydratesFromMetadata() {
        DiscreteTokenDatasetPlan original = plan(11L, 2);
        DiscreteTokenDatasetPlan changed = plan(12L, 2);
        DiscreteTokenDatasetFingerprintMatch mismatch =
                changed.verifyFingerprint(original.fingerprint());

        DiscreteTokenDatasetFingerprintMatch rehydrated =
                DiscreteTokenDatasetFingerprintMatch.fromMetadata(mismatch.toMetadata());

        assertEquals(mismatch, rehydrated);
        assertEquals("mismatched", rehydrated.status());
        assertFalse(rehydrated.matches());
    }

    @Test
    void rejectsMalformedInputs() {
        DiscreteTokenDatasetFingerprint fingerprint = plan(11L, 2).fingerprint();
        DiscreteTokenDatasetPlan plan = plan(11L, 2);
        DiscreteTokenDatasetPlanReport report = plan.report();

        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetFingerprintMatch.verify(null, fingerprint));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verify(fingerprint, (DiscreteTokenDatasetFingerprint) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verify(fingerprint, (DiscreteTokenDatasetPlan) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verify(fingerprint, (DiscreteTokenDatasetPlanReport) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verifyMetadata(null, plan));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verifyMetadata(fingerprint.toMetadata(), (DiscreteTokenDatasetPlan) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.verifyMetadata(fingerprint.toMetadata(), (DiscreteTokenDatasetPlanReport) null));
        assertThrows(NullPointerException.class, () -> plan.verifyFingerprint(null));
        assertThrows(NullPointerException.class, () -> plan.verifyFingerprintMetadata(null));
        assertThrows(NullPointerException.class, () -> plan.verifyFingerprintMetadataSection(null));
        assertThrows(NullPointerException.class, () -> plan.verifyFingerprintMetadataSection(fingerprint.toMetadata(), null));
        assertThrows(NullPointerException.class, () -> report.verifyFingerprint(null));
        assertThrows(NullPointerException.class, () -> report.verifyFingerprintMetadata(null));
        assertThrows(NullPointerException.class, () -> report.verifyFingerprintMetadataSection(null));
        assertThrows(NullPointerException.class, () -> report.verifyFingerprintMetadataSection(fingerprint.toMetadata(), null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprintMatch.fromMetadata(Map.of(
                        "status",
                        "matched",
                        "matches",
                        true,
                        "expected",
                        fingerprint.toMetadata(),
                        "actual",
                        changedFingerprint().toMetadata())));
    }

    private static DiscreteTokenDatasetFingerprint changedFingerprint() {
        return plan(12L, 2).fingerprint();
    }

    private static DiscreteTokenDatasetPlan plan(long seed, int trainBatchSize) {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        example("alpha", 0),
                        example("alpha", 1),
                        example("alpha", 2),
                        example("beta", 10),
                        example("beta", 11),
                        example("beta", 12)),
                new DiscreteTokenDatasetPlanConfig(
                        0.25d,
                        0.25d,
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        seed,
                        trainBatchSize,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                        7L,
                        false));
    }

    private static DiscreteTokenDatasetExample example(String taskId, int index) {
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                new int[] {index + 1, index + 2},
                new int[] {index + 100},
                1,
                Map.of());
    }
}
