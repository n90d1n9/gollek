package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerRuntimeArtifactWriterTest {
    @TempDir
    Path tempDir;

    @Test
    void writesHistoryCsvAtomically() throws Exception {
        Path history = tempDir.resolve("history.csv");
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("epoch", 1);
        row.put("trainLoss", 0.5);
        row.put("validationLoss", 0.4);

        TrainerRuntimeArtifactWriter.WriteResult result =
                TrainerRuntimeArtifactWriter.writeHistory(history, List.of(row));

        assertTrue(result.written());
        assertNull(result.error());
        String csv = Files.readString(history);
        assertTrue(csv.contains("epoch,trainLoss,validationLoss"));
        assertTrue(csv.contains("1,0.5,0.4"));
    }

    @Test
    void writesReportWithSchemaAndSummaryMetadata() throws Exception {
        Path report = tempDir.resolve("report.json");
        TrainingSummary summary = new TrainingSummary(
                3,
                0.25,
                2,
                0.3,
                0.25,
                1234,
                Map.of("device", "metal"));

        TrainerRuntimeArtifactWriter.WriteResult result = TrainerRuntimeArtifactWriter.writeReport(
                report,
                summary,
                Instant.parse("2026-05-18T00:00:00Z"));

        assertTrue(result.written());
        assertNull(result.error());
        String json = Files.readString(report);
        assertTrue(json.endsWith("\n"));
        assertTrue(json.contains("\"schema\":\"gollek.canonical-trainer.report.v1\""));
        assertTrue(json.contains("\"generatedAt\":\"2026-05-18T00:00:00Z\""));
        assertTrue(json.contains("\"device\":\"metal\""));
    }

    @Test
    void writesManifestForExistingArtifacts() throws Exception {
        Path model = tempDir.resolve("model.safetensors");
        Path manifest = tempDir.resolve("manifest.properties");
        Files.writeString(model, "weights");

        TrainerRuntimeArtifactWriter.WriteResult result = TrainerRuntimeArtifactWriter.writeManifest(
                manifest,
                Map.of("model", model),
                1,
                Instant.parse("2026-05-18T00:00:00Z"));

        assertTrue(result.written());
        assertNull(result.error());
        String properties = Files.readString(manifest);
        assertTrue(properties.contains("formatVersion=1"));
        assertTrue(properties.contains("artifact.model.file=model.safetensors"));
        assertTrue(properties.contains("artifact.model.sha256="));
    }

    @Test
    void skippedWhenTargetIsAbsent() {
        TrainerRuntimeArtifactWriter.WriteResult result =
                TrainerRuntimeArtifactWriter.writeHistory(null, List.of());

        assertFalse(result.written());
        assertNull(result.error());
    }
}
