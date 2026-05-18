package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class TrainerCheckpointResumeDiagnosticsTest {

    @Test
    void recordCompatibilityMismatchKeepsFirstOccurrenceOrderAndDeduplicates() {
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();

        diagnostics.recordCompatibilityMismatch("model", "signature mismatch");
        diagnostics.recordCompatibilityMismatch("optimizer", "size mismatch");
        diagnostics.recordCompatibilityMismatch("model", "signature mismatch");

        assertEquals(
                List.of("model: signature mismatch", "optimizer: size mismatch"),
                diagnostics.compatibilityMismatches());
    }

    @Test
    void compatibilityMismatchesSnapshotIsImmutable() {
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();
        diagnostics.recordCompatibilityMismatch("model", "signature mismatch");

        List<String> snapshot = diagnostics.compatibilityMismatches();
        diagnostics.recordCompatibilityMismatch("history", "invalid csv");

        assertEquals(List.of("model: signature mismatch"), snapshot);
        assertThrows(UnsupportedOperationException.class, () -> snapshot.add("optimizer: size mismatch"));
    }

    @Test
    void missingArtifactsKeepSummaryOrder() {
        assertEquals(
                List.of("model", "optimizer", "scheduler", "gradScaler", "history"),
                TrainerCheckpointResumeDiagnostics.missingArtifacts(true, true, true, true, true));
        assertEquals(
                List.of("optimizer", "gradScaler"),
                TrainerCheckpointResumeDiagnostics.missingArtifacts(false, true, false, true, false));
        assertEquals(
                List.of(),
                TrainerCheckpointResumeDiagnostics.missingArtifacts(false, false, false, false, false));
    }

    @Test
    void missingArtifactExceptionUsesCanonicalMessage() {
        IllegalStateException error = TrainerCheckpointResumeDiagnostics.missingArtifactException(
                "optimizer",
                Path.of("/tmp/canonical-optimizer.state"));

        assertEquals(
                "Missing optimizer checkpoint artifact for resume: /tmp/canonical-optimizer.state",
                error.getMessage());
    }
}
