package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerRuntimeSmokeArtifactsTest {
    @TempDir
    Path tempDir;

    @Test
    void writesAndVerifiesTrainerRuntimeSmokeArtifactBundle() throws Exception {
        TrainerRuntimeSmoke.Result result = passingResult();

        TrainerRuntimeSmokeArtifacts.ArtifactBundle bundle =
                TrainerRuntimeSmokeArtifacts.write(tempDir, result);
        TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                TrainerRuntimeSmokeArtifacts.verify(bundle);

        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));
        assertTrue(Files.isRegularFile(bundle.junitXmlFile()));
        assertTrue(verification.passed(), () -> verification.failures().toString());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertTrue(verification.junitXmlWellFormed());
        assertTrue(verification.inspection().smokePassed());
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(bundle.artifactMap(), verification.inspection().artifactMap());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
    }

    @Test
    void verificationDetectsMarkdownDrift() throws Exception {
        TrainerRuntimeSmokeArtifacts.ArtifactBundle bundle =
                TrainerRuntimeSmokeArtifacts.write(tempDir, passingResult());
        Files.writeString(bundle.markdownFile(), "# stale smoke report\n");

        TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                TrainerRuntimeSmokeArtifacts.verify(tempDir);

        assertFalse(verification.passed());
        assertFalse(verification.markdownMatchesJson());
        assertTrue(verification.failures().stream().anyMatch(failure -> failure.contains("Markdown report")));
    }

    @Test
    void refreshRegeneratesDerivedReportsFromJson() throws Exception {
        TrainerRuntimeSmokeArtifacts.ArtifactBundle bundle =
                TrainerRuntimeSmokeArtifacts.write(tempDir, passingResult());
        Files.writeString(bundle.markdownFile(), "# stale smoke report\n");
        Files.writeString(bundle.junitXmlFile(), "<not-junit></not-junit>\n");

        TrainerRuntimeSmokeArtifacts.ArtifactBundle refreshed =
                TrainerRuntimeSmokeArtifacts.refresh(tempDir);
        TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                TrainerRuntimeSmokeArtifacts.verify(refreshed);

        assertEquals(bundle.jsonSha256(), refreshed.jsonSha256());
        assertTrue(verification.passed(), () -> verification.failures().toString());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertTrue(Files.readString(refreshed.markdownFile()).contains("# Gollek Trainer Runtime Smoke"));
        assertTrue(Files.readString(refreshed.junitXmlFile()).contains("gollek.trainer.runtime.smoke"));
    }

    private TrainerRuntimeSmoke.Result passingResult() {
        TrainingSummary firstRun = new TrainingSummary(
                1,
                0.25,
                1,
                0.5,
                0.25,
                120,
                Map.of(
                        "requestedDevice", "cpu",
                        "executionBackend", "cpu",
                        "executionAccelerated", false,
                        "executionFallback", false,
                        "optimizerStepCount", 2));
        TrainingSummary resumedRun = new TrainingSummary(
                2,
                0.125,
                2,
                0.2,
                0.125,
                180,
                Map.of(
                        "requestedDevice", "cpu",
                        "executionBackend", "cpu",
                        "executionAccelerated", false,
                        "executionFallback", false,
                        "optimizerStepCount", 4));
        return new TrainerRuntimeSmoke.Result(
                tempDir.resolve("checkpoints"),
                firstRun,
                resumedRun,
                List.of(
                        new TrainerRuntimeSmoke.Check("checkpointFilesPresent", true, "model/runtime present"),
                        new TrainerRuntimeSmoke.Check("resumedFromCheckpoint", true, "resume metadata true")));
    }
}
