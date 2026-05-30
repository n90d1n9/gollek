package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerRuntimeSmokeTest {
    @TempDir
    Path tempDir;

    @Test
    void trainerRuntimeSmokeRunsCheckpointResumeThroughPublicFacade() throws Exception {
        TrainerRuntimeSmoke.Result result = Gollek.DL.trainerRuntimeSmoke(
                TrainerRuntimeSmoke.Options.builder()
                        .checkpointDir(tempDir)
                        .device("auto")
                        .build());

        assertTrue(result.passed(), () -> result.failures().toString());
        result.requirePassed();
        assertEquals(tempDir.toAbsolutePath().normalize(), result.checkpointDir());
        assertEquals(1, result.firstRun().epochCount());
        assertEquals(2, result.resumedRun().epochCount());
        assertTrue(result.check("resumedFromCheckpoint").orElseThrow().passed());
        assertTrue(result.check("deviceMetadataPresent").orElseThrow().passed());
        assertTrue(Files.isRegularFile(tempDir.resolve(TrainerCheckpointLayout.RUNTIME_FILE_NAME)));
        assertTrue(Files.isRegularFile(tempDir.resolve(TrainerCheckpointLayout.MANIFEST_FILE_NAME)));

        String junitXml = TrainerRuntimeSmoke.renderJUnitXml(result);
        assertTrue(TrainingReportXml.isWellFormed(junitXml));
        assertTrue(junitXml.contains("tests=\"10\""));
        assertTrue(junitXml.contains("failures=\"0\""));
        assertTrue(junitXml.contains("name=\"resumedFromCheckpoint\""));

        String markdown = TrainerRuntimeSmoke.renderMarkdown(result);
        assertTrue(markdown.contains("# Gollek Trainer Runtime Smoke"));
        assertTrue(markdown.contains("- Status: PASS"));
        assertTrue(markdown.contains("| resumedFromCheckpoint | PASS |"));
        assertTrue(markdown.contains("| Optimizer steps |"));
    }

    @Test
    void trainerRuntimeSmokeOptionsRejectInvalidResumeWindow() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> TrainerRuntimeSmoke.Options.builder()
                        .firstRunEpochs(2)
                        .resumedEpochs(2)
                        .build());

        assertTrue(error.getMessage().contains("resumedEpochs"));
    }

    @Test
    void trainerRuntimeSmokeJUnitXmlIncludesFailureDetails() {
        TrainingSummary summary = new TrainingSummary(1, 0.5, 1, 0.7, 0.5, 250, Map.of());
        TrainerRuntimeSmoke.Result result = new TrainerRuntimeSmoke.Result(
                tempDir,
                summary,
                summary,
                List.of(new TrainerRuntimeSmoke.Check("checkpointFilesPresent", false, "missing runtime state")));

        String junitXml = TrainerRuntimeSmoke.renderJUnitXml(result);

        assertTrue(TrainingReportXml.isWellFormed(junitXml));
        assertTrue(junitXml.contains("failures=\"1\""));
        assertTrue(junitXml.contains("<failure message=\"missing runtime state\">missing runtime state</failure>"));

        String markdown = TrainerRuntimeSmoke.renderMarkdown(result);
        assertTrue(markdown.contains("- Status: FAIL"));
        assertTrue(markdown.contains("## Failures"));
        assertTrue(markdown.contains("`checkpointFilesPresent`: missing runtime state"));
    }
}
