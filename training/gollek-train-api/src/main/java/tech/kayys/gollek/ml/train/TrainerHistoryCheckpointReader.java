package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Reads persisted epoch-history checkpoints after resume compatibility checks.
 */
final class TrainerHistoryCheckpointReader {
    private TrainerHistoryCheckpointReader() {
    }

    static ReadResult readExisting(Path historyFile) {
        try {
            return ReadResult.loaded(TrainingHistoryCsv.read(historyFile));
        } catch (Exception error) {
            return ReadResult.failed(error);
        }
    }

    record ReadResult(
            boolean loaded,
            List<Map<String, Object>> rows,
            String error,
            Exception cause) {
        static ReadResult loaded(List<Map<String, Object>> rows) {
            return new ReadResult(true, rows == null ? List.of() : List.copyOf(rows), null, null);
        }

        static ReadResult failed(Exception error) {
            return new ReadResult(false, List.of(), error.getMessage(), error);
        }
    }
}
