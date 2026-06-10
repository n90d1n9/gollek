package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;

class TrainingReportRuntimeInputProfileGateTest {
    @TempDir
    Path tempDir;

    @Test
    void passesWhenRuntimeInputProfileIsMissing() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.of()));

        TrainingReportRuntimeInputProfileGate.Result result = report.runtimeInputProfileGate();

        assertFalse(result.available());
        assertTrue(result.passed());
        assertTrue(result.findings().isEmpty());
        assertEquals("Runtime input profile is not available.", result.message());
    }

    @Test
    void warnsWhenDominantInputStageExceedsPolicy() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 18.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 2.0))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(95.0, 70.0, 20.0);

        TrainingReportRuntimeInputProfileGate.Result result =
                Gollek.DL.trainingReportRuntimeInputProfileGate(report, policy);

        assertTrue(result.available());
        assertFalse(result.passed());
        assertEquals(1, result.findings().size());
        TrainingReportRuntimeInputProfileGate.Finding finding = result.findings().getFirst();
        assertEquals("runtime-input-dominant-stage", finding.code());
        assertTrue(finding.message().contains("train.next()"));
        assertEquals("train", finding.evidence().get("scope"));
        assertEquals("next", finding.evidence().get("stage"));
        assertEquals(90.0, finding.evidence().get("stagePercent"));
        assertEquals(result.toMap(), report.runtimeInputProfileGate(policy).toMap());

        String markdown = Gollek.DL.trainingReportRuntimeInputProfileGateMarkdown(report, policy);
        assertTrue(markdown.startsWith("# Runtime Input Profile Gate\n"));
        assertTrue(markdown.contains("- Passed: `false`"));
        assertTrue(markdown.contains("## Policy"));
        assertTrue(markdown.contains("| Dominant stage | `70.000%` |"));
        assertTrue(markdown.contains("## Input Summary"));
        assertTrue(markdown.contains("| Dominant scope | `train` |"));
        assertTrue(markdown.contains("## Findings"));
        assertTrue(markdown.contains("`runtime-input-dominant-stage`"));
        assertEquals(markdown, result.markdown());

        String junitXml = Gollek.DL.trainingReportRuntimeInputProfileGateJUnitXml(report, policy);
        assertTrue(junitXml.startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assertTrue(junitXml.contains("<testsuite name=\"gollek.training.runtime.input\" tests=\"1\" failures=\"1\""));
        assertTrue(junitXml.contains("name=\"gate.findingCodes\" value=\"runtime-input-dominant-stage\""));
        assertTrue(junitXml.contains("type=\"RUNTIME_INPUT_DOMINANT_STAGE\""));
        assertTrue(junitXml.contains("<failure type=\"RUNTIME_INPUT_DOMINANT_STAGE\""));
        assertTrue(junitXml.contains("<system-out># Runtime Input Profile Gate"));
        assertEquals(junitXml, result.junitXml());
    }

    @Test
    void recommendsPrefetchWhenTrainNextDominatesAndPrefetchIsDisabled() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 18.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 2.0),
                Map.entry("trainLoaderPlan.batchCount", 6),
                Map.entry("trainLoaderPlan.prefetch.enabled", false),
                Map.entry("trainLoaderPlan.prefetch.maxBufferedItems", 0),
                Map.entry("trainLoaderPlan.prefetch.summary", "prefetch[enabled=false]"))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(95.0, 70.0, 20.0);

        TrainingReportRuntimeInputProfileGate.Result result = report.runtimeInputProfileGate(policy);
        List<String> codes = result.findings().stream()
                .map(TrainingReportRuntimeInputProfileGate.Finding::code)
                .toList();

        assertEquals(List.of(
                "runtime-input-dominant-stage",
                "runtime-input-train-prefetch-disabled"), codes);
        TrainingReportRuntimeInputProfileGate.Finding prefetch = result.findings().get(1);
        assertTrue(prefetch.action().contains("DataLoader.prefetch(2)"));
        assertEquals(false, prefetch.evidence().get("trainLoaderPlan.prefetch.enabled"));
        assertEquals("next", prefetch.evidence().get("dominantStage"));
        assertEquals(2, prefetch.evidence().get("recommendedPrefetchBufferSize"));

        String markdown = result.markdown();
        assertTrue(markdown.contains("`runtime-input-train-prefetch-disabled`"));
        assertTrue(markdown.contains("Training input `next()` dominates while train loader prefetching is disabled."));
    }

    @Test
    void recommendsLargerPrefetchBufferWhenTrainNextDominatesAndBufferIsTooSmall() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 18.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 2.0),
                Map.entry("trainLoaderPlan.batchCount", 6),
                Map.entry("trainLoaderPlan.prefetch.enabled", true),
                Map.entry("trainLoaderPlan.prefetch.bufferSize", 1),
                Map.entry("trainLoaderPlan.prefetch.workerCount", 1),
                Map.entry("trainLoaderPlan.prefetch.maxBufferedItems", 1),
                Map.entry("trainLoaderPlan.prefetch.summary", "prefetch[enabled=true, bufferSize=1, workerCount=1]"))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(95.0, 70.0, 20.0);

        TrainingReportRuntimeInputProfileGate.Result result = report.runtimeInputProfileGate(policy);
        List<String> codes = result.findings().stream()
                .map(TrainingReportRuntimeInputProfileGate.Finding::code)
                .toList();

        assertEquals(List.of(
                "runtime-input-dominant-stage",
                "runtime-input-train-prefetch-buffer-too-small"), codes);
        TrainingReportRuntimeInputProfileGate.Finding prefetch = result.findings().get(1);
        assertTrue(prefetch.action().contains("at least 2 item(s)"));
        assertEquals(1, prefetch.evidence().get("trainLoaderPlan.prefetch.maxBufferedItems"));
        assertEquals(2, prefetch.evidence().get("recommendedPrefetchBufferSize"));
    }

    @Test
    void doesNotRecommendPrefetchWhenValidationInputDominates() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.next.totalMillis", 2.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 18.0),
                Map.entry("trainLoaderPlan.prefetch.enabled", false))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(95.0, 70.0, 20.0);

        List<String> codes = report.runtimeInputProfileGate(policy).findings().stream()
                .map(TrainingReportRuntimeInputProfileGate.Finding::code)
                .toList();

        assertEquals(List.of("runtime-input-dominant-stage"), codes);
    }

    @Test
    void buildsRuntimeInputGatePolicyFromMapAndPresets() {
        TrainingReportRuntimeInputProfileGate.Policy policy =
                TrainingReportRuntimeInputProfileGate.Policy.fromMap(Map.of(
                        "maxDominantScopePercent", "90.5",
                        "maxDominantStagePercent", 66.0,
                        "maxTrainToValidationTotalRatio", 3));

        assertEquals(90.5, policy.maxDominantScopePercent());
        assertEquals(66.0, policy.maxDominantStagePercent());
        assertEquals(3.0, policy.maxTrainToValidationTotalRatio());
        assertEquals(70.0, TrainingReportRuntimeInputProfileGate.Policy.strict().maxDominantStagePercent());
        assertEquals(20.0, TrainingReportRuntimeInputProfileGate.Policy.permissive()
                .maxTrainToValidationTotalRatio());
        assertEquals(42.0, policy.withMaxDominantStagePercent(42.0).maxDominantStagePercent());
    }

    @Test
    void warnsWhenTrainInputIsSkewedAgainstValidationInput() {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.next.totalMillis", 24.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 2.0))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(99.0, 101.0, 4.0);

        List<String> codes = TrainingReportRuntimeInputProfileGate.evaluate(report, policy)
                .findings()
                .stream()
                .map(TrainingReportRuntimeInputProfileGate.Finding::code)
                .toList();

        assertEquals(List.of("runtime-input-train-validation-skew"), codes);
        assertThrows(IllegalStateException.class, () -> report.runtimeInputProfileGate(policy).requirePassed());
    }

    @Test
    void writesVerifiesAndRefreshesRuntimeInputGateArtifacts() throws IOException {
        TrainingReport report = TrainingReport.of(Map.of("metadata", Map.ofEntries(
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 18.0),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 2.0))));
        TrainingReportRuntimeInputProfileGate.Policy policy =
                new TrainingReportRuntimeInputProfileGate.Policy(95.0, 70.0, 20.0);
        TrainingReportRuntimeInputProfileGate.Result result = report.runtimeInputProfileGate(policy);

        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportRuntimeInputProfileGateArtifacts(tempDir, result);
        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportRuntimeInputProfileGateArtifacts(tempDir);
        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportRuntimeInputProfileGateArtifacts(bundle);

        assertFalse(bundle.passed());
        assertFalse(inspection.passed());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertTrue(inspection.markdown().contains("# Runtime Input Profile Gate"));
        assertTrue(inspection.junitXml().contains("gollek.training.runtime.input"));
        assertTrue(verification.passed(), verification.message());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        verification.requirePassed();

        Files.writeString(bundle.markdownFile(), inspection.markdown() + "\n<!-- tampered -->\n");
        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification tampered =
                Gollek.DL.verifyTrainingReportRuntimeInputProfileGateArtifacts(tempDir);
        assertFalse(tampered.passed());
        assertFalse(tampered.markdownMatchesJson());
        assertThrows(IllegalStateException.class, tampered::requirePassed);

        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle refreshed =
                Gollek.DL.refreshTrainingReportRuntimeInputProfileGateArtifacts(tempDir);
        assertTrue(Gollek.DL.verifyTrainingReportRuntimeInputProfileGateArtifacts(refreshed).passed());
    }
}
