package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetFingerprintTest {
    @Test
    void fingerprintsExamplesDeterministicallyAcrossMetadataMapOrder() {
        DiscreteTokenDatasetFingerprint first = DiscreteTokenDatasetFingerprint.fromExamples(List.of(
                example("task", 0, new int[] {1, 2}, new int[] {3}, Map.of("b", 2, "a", List.of(1, 2)))));
        DiscreteTokenDatasetFingerprint second = DiscreteTokenDatasetFingerprint.fromExamples(List.of(
                example("task", 0, new int[] {1, 2}, new int[] {3}, Map.of("a", List.of(1, 2), "b", 2))));

        assertEquals(first, second);
        assertEquals(DiscreteTokenDatasetFingerprint.ALGORITHM, first.algorithm());
        assertEquals(64, first.value().length());
        assertEquals(12, first.shortValue().length());
        assertEquals(first.value().substring(0, 8), first.shortValue(8));
        assertEquals(1, first.exampleCount());
    }

    @Test
    void exampleFingerprintChangesWhenTokensOrMetadataChange() {
        DiscreteTokenDatasetFingerprint baseline = DiscreteTokenDatasetFingerprint.fromExamples(List.of(
                example("task", 0, new int[] {1, 2}, new int[] {3}, Map.of("source", "a"))));
        DiscreteTokenDatasetFingerprint tokenChanged = DiscreteTokenDatasetFingerprint.fromExamples(List.of(
                example("task", 0, new int[] {1, 9}, new int[] {3}, Map.of("source", "a"))));
        DiscreteTokenDatasetFingerprint metadataChanged = DiscreteTokenDatasetFingerprint.fromExamples(List.of(
                example("task", 0, new int[] {1, 2}, new int[] {3}, Map.of("source", "b"))));

        assertNotEquals(baseline, tokenChanged);
        assertNotEquals(baseline, metadataChanged);
    }

    @Test
    void planFingerprintTracksSplitAndEpochConfiguration() {
        DiscreteTokenDatasetPlan first = plan(11L, 2);
        DiscreteTokenDatasetPlan same = plan(11L, 2);
        DiscreteTokenDatasetPlan differentSeed = plan(12L, 2);
        DiscreteTokenDatasetPlan differentBatchSize = plan(11L, 3);

        assertEquals(first.fingerprint(), same.fingerprint());
        assertNotEquals(first.fingerprint(), differentSeed.fingerprint());
        assertNotEquals(first.fingerprint(), differentBatchSize.fingerprint());
        assertEquals(first.profile().exampleCount(), first.fingerprint().exampleCount());
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetFingerprint fingerprint =
                DiscreteTokenDatasetFingerprint.fromExamples(List.of(example("task", 0)));

        Map<String, Object> metadata = fingerprint.toMetadata();

        assertEquals(DiscreteTokenDatasetFingerprint.ALGORITHM, metadata.get("algorithm"));
        assertEquals(fingerprint.value(), metadata.get("value"));
        assertEquals(fingerprint.shortValue(), metadata.get("shortValue"));
        assertEquals(1, metadata.get("exampleCount"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void recreatesFingerprintFromMetadata() {
        DiscreteTokenDatasetFingerprint fingerprint =
                DiscreteTokenDatasetFingerprint.fromExamples(List.of(example("task", 0)));

        assertEquals(fingerprint, DiscreteTokenDatasetFingerprint.fromMetadata(fingerprint.toMetadata()));
        assertEquals(
                fingerprint,
                DiscreteTokenDatasetFingerprint.fromMetadataSection(Map.of(
                        DiscreteTokenDatasetFingerprint.METADATA_KEY,
                        fingerprint.toMetadata())));
        assertEquals(
                fingerprint,
                DiscreteTokenDatasetFingerprint.fromMetadataSection(
                        Map.of("datasetFingerprint", fingerprint.toMetadata()),
                        "datasetFingerprint"));
        assertEquals(
                new DiscreteTokenDatasetFingerprint("SHA-256", "abcdef", 7),
                DiscreteTokenDatasetFingerprint.fromMetadata(Map.of(
                        "algorithm", "SHA-256",
                        "value", "ABCDEF",
                        "exampleCount", "7")));
    }

    @Test
    void rejectsMalformedInputs() {
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetFingerprint.fromExamples(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetFingerprint.fromExamples(Arrays.asList(example("task", 0), null)));
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetFingerprint.fromPlan(null));
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetFingerprint.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetFingerprint(" ", "abc", 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetFingerprint("SHA-256", " ", 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetFingerprint("SHA-256", "abc", -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadata(Map.of(
                        "value", "abc",
                        "exampleCount", 1)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadataSection(Map.of("other", Map.of())));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadataSection(Map.of(
                        DiscreteTokenDatasetFingerprint.METADATA_KEY,
                        "not-a-map")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadataSection(Map.of(
                        DiscreteTokenDatasetFingerprint.METADATA_KEY,
                        Map.of()),
                        " "));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadata(Map.of(
                        "algorithm", "SHA-256",
                        "value", "abc",
                        "exampleCount", 1.5d)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromMetadata(Map.of(
                        "algorithm", "SHA-256",
                        "value", "abc",
                        "exampleCount", "-1")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetFingerprint.fromExamples(List.of(example("task", 0))).shortValue(0));
    }

    @Test
    void fingerprintLooksLikeLowercaseHex() {
        String value = DiscreteTokenDatasetFingerprint.fromExamples(List.of(example("task", 0))).value();

        assertTrue(value.matches("[0-9a-f]{64}"));
        assertFalse(value.matches(".*[A-F].*"));
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
        return example(taskId, index, new int[] {index + 1, index + 2}, new int[] {index + 100}, Map.of());
    }

    private static DiscreteTokenDatasetExample example(
            String taskId,
            int index,
            int[] inputTokens,
            int[] targetTokens,
            Map<String, Object> metadata) {
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                inputTokens,
                targetTokens,
                1,
                metadata);
    }
}
