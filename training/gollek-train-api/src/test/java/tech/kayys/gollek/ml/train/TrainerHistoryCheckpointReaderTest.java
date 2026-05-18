package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerHistoryCheckpointReaderTest {
    @TempDir
    Path tempDir;

    @Test
    void readsExistingHistoryRows() throws Exception {
        Path history = tempDir.resolve("history.csv");
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("epoch", 2);
        row.put("trainLoss", 0.25);
        row.put("validationLoss", 0.2);
        row.put("trainMetrics", Map.of("accuracy", 0.75));
        Files.writeString(history, TrainingHistoryCsv.write(List.of(row)));

        TrainerHistoryCheckpointReader.ReadResult result =
                TrainerHistoryCheckpointReader.readExisting(history);

        assertTrue(result.loaded());
        assertNull(result.error());
        assertNull(result.cause());
        assertEquals(1, result.rows().size());
        assertEquals(2, result.rows().getFirst().get("epoch"));
        assertEquals(0.25, result.rows().getFirst().get("trainLoss"));
        assertEquals(Map.of("accuracy", 0.75), result.rows().getFirst().get("trainMetrics"));
    }

    @Test
    void capturesCsvParseFailure() throws Exception {
        Path history = tempDir.resolve("history.csv");
        Files.writeString(history, "epoch,epoch\n1,2\n");

        TrainerHistoryCheckpointReader.ReadResult result =
                TrainerHistoryCheckpointReader.readExisting(history);

        assertFalse(result.loaded());
        assertEquals(List.of(), result.rows());
        assertNotNull(result.error());
        assertTrue(result.error().contains("duplicate column 'epoch'"));
        assertNotNull(result.cause());
    }

    @Test
    void capturesMissingFileFailureWhenCalledWithoutPresenceCheck() {
        Path history = tempDir.resolve("missing.csv");

        TrainerHistoryCheckpointReader.ReadResult result =
                TrainerHistoryCheckpointReader.readExisting(history);

        assertFalse(result.loaded());
        assertEquals(List.of(), result.rows());
        assertNotNull(result.error());
        assertNotNull(result.cause());
    }
}
