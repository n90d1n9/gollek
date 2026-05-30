package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetCheckpointMetadataJsonTest {
    @TempDir
    Path tempDir;

    @Test
    void writesAndReadsCheckpointManifestMetadata() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReport planReport =
                plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        Map<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("note", "resume \"safe\"\nnext");
        attributes.put("weights", List.of(1, 2, 3));
        attributes.put("audit", Map.of("enabled", true));
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(planReport)
                        .attributes(attributes)
                        .build();
        Path path = tempDir.resolve("runs/run-001/checkpoint.json");

        DiscreteTokenDatasetCheckpointMetadataJson.write(path, manifest);

        assertTrue(Files.exists(path));
        Map<String, Object> loaded = DiscreteTokenDatasetCheckpointMetadataJson.read(path);
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointMetadataJson.readSnapshot(path);
        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumePolicy.training().requireReady(loaded, plan);
        Map<?, ?> loadedAttributes = (Map<?, ?>) loaded.get("attributes");

        assertEquals(manifest.summary(), snapshot.summary());
        assertTrue(resumeReport.ready());
        assertEquals("resume \"safe\"\nnext", loadedAttributes.get("note"));
        assertEquals(List.of(1L, 2L, 3L), loadedAttributes.get("weights"));
        assertEquals(true, ((Map<?, ?>) loadedAttributes.get("audit")).get("enabled"));
    }

    @Test
    void roundTripsResumePolicyAndReportMetadata() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withCurrentPlanGate(new DiscreteTokenDatasetPlanReadinessGate(
                                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                                        .withHighPaddingRateThreshold(0.33d),
                                true))
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                                .experimentName("gram-nqueens")
                                .runId("run-001")
                                .seed(2026L)
                                .minimumCheckpointStep(10L)
                                .build());

        DiscreteTokenDatasetCheckpointResumePolicy parsedPolicy =
                DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(
                        DiscreteTokenDatasetCheckpointMetadataJson.fromJson(
                                DiscreteTokenDatasetCheckpointMetadataJson.toJson(policy.toMetadata())));
        DiscreteTokenDatasetCheckpointResumeReport report = policy.evaluate(manifest, plan);
        Map<String, Object> reportMetadata =
                DiscreteTokenDatasetCheckpointMetadataJson.fromJson(
                        DiscreteTokenDatasetCheckpointMetadataJson.toJson(report.toMetadata()));

        assertEquals(policy, parsedPolicy);
        assertEquals(true, reportMetadata.get("ready"));
        assertEquals(true, reportMetadata.get("policyTracked"));
        assertEquals(
                policy,
                DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(
                        (Map<?, ?>) reportMetadata.get("policy")));
        assertEquals(
                manifest.runId(),
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                                (Map<?, ?>) reportMetadata.get("checkpoint"))
                        .runId());
    }

    @Test
    void emitsDeterministicCompactJsonWithSortedObjectKeys() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("b", 2);
        nested.put("a", 1);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("z", 1);
        metadata.put("a", nested);

        String json = DiscreteTokenDatasetCheckpointMetadataJson.toJson(metadata);

        assertEquals("{\"a\":{\"a\":1,\"b\":2},\"z\":1}", json);
    }

    @Test
    void escapesStringsAndParsesControlEscapes() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("text", "line\nquote\"slash\\");
        metadata.put("unicode", "\u0001");

        String json = DiscreteTokenDatasetCheckpointMetadataJson.toJson(metadata);
        Map<String, Object> parsed = DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json);

        assertEquals("{\"text\":\"line\\nquote\\\"slash\\\\\",\"unicode\":\"\\u0001\"}", json);
        assertEquals(metadata, parsed);
    }

    @Test
    @SuppressWarnings("unchecked")
    void parsesPrettyJsonAndImmutableResults() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("name", "run-001");
        metadata.put("values", List.of(1, 2, 3));

        String json = DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(metadata);
        Map<String, Object> parsed = DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json);

        assertTrue(json.contains(System.lineSeparator()) || json.contains("\n"));
        assertEquals("run-001", parsed.get("name"));
        assertEquals(List.of(1L, 2L, 3L), parsed.get("values"));
        assertThrows(UnsupportedOperationException.class, () -> parsed.put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> ((List<Object>) parsed.get("values")).add(4L));
    }

    @Test
    void rejectsMalformedJsonAndUnsupportedMetadataValues() {
        Map<Object, Object> nonStringKey = new LinkedHashMap<>();
        nonStringKey.put(1, "one");

        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.toJson((Map<?, ?>) null));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.fromJson("[1]"));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.fromJson("{\"a\":}"));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.fromJson("{\"a\":1,\"a\":2}"));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.toJson(Map.of("bad", Double.NaN)));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.toJson(Map.of("bad", new Object())));
        assertThrows(IllegalArgumentException.class, () ->
                DiscreteTokenDatasetCheckpointMetadataJson.toJson(nonStringKey));
    }

    @Test
    void loadedCheckpointStillBlocksMismatchedCurrentPlan() {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        Map<String, Object> loaded =
                DiscreteTokenDatasetCheckpointMetadataJson.fromJson(
                        DiscreteTokenDatasetCheckpointMetadataJson.toJson(manifest.toMetadata()));

        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(loaded, changed);

        assertFalse(report.ready());
        assertFalse(report.fingerprintMatched());
        assertTrue(report.rejectionReasons().stream().anyMatch(reason -> reason.contains("mismatch")));
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-nqueens")
                .runId("run-001")
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(1_900_000_000_000L)
                .createdBy("trainer-test");
    }

    private static DiscreteTokenDatasetPlan cleanPlan() {
        return plan(0L);
    }

    private static DiscreteTokenDatasetPlan changedPlan() {
        return plan(42L);
    }

    private static DiscreteTokenDatasetPlan plan(long seed) {
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
                        seed,
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
