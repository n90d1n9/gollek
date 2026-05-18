package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerBestModelCheckpointMonitorTest {

    @Test
    void disabledMonitorNeverRequestsSave() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                false,
                null,
                CanonicalTrainer.BestModelMonitorMode.MIN,
                0.0);

        TrainerBestModelCheckpointMonitor.Decision decision =
                monitor.evaluate(1, 0.5, Map.of());

        assertFalse(decision.shouldSave());
        assertFalse(monitor.state().saved());
        assertEquals(-1, monitor.state().epoch());
    }

    @Test
    void validationLossMonitorSavesOnlyWhenLossImprovesBeyondMinDelta() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                true,
                null,
                CanonicalTrainer.BestModelMonitorMode.MIN,
                0.05);

        TrainerBestModelCheckpointMonitor.Decision first = monitor.evaluate(1, 1.0, Map.of());
        assertTrue(first.shouldSave());
        monitor.markSaved(first);

        assertFalse(monitor.evaluate(2, 0.96, Map.of()).shouldSave());
        TrainerBestModelCheckpointMonitor.Decision improved = monitor.evaluate(3, 0.90, Map.of());
        assertTrue(improved.shouldSave());
        monitor.markSaved(improved);

        assertEquals(3, monitor.state().epoch());
        assertEquals(0.90, monitor.state().validationLoss(), 1e-6);
        assertEquals(0.90, monitor.state().monitorValue(), 1e-6);
    }

    @Test
    void metricMonitorUsesConfiguredMaxMode() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                true,
                "accuracy",
                CanonicalTrainer.BestModelMonitorMode.MAX,
                0.01);

        TrainerBestModelCheckpointMonitor.Decision first =
                monitor.evaluate(1, 2.0, Map.of("accuracy", 0.80));
        monitor.markSaved(first);

        assertFalse(monitor.evaluate(2, 1.5, Map.of("accuracy", 0.805)).shouldSave());
        TrainerBestModelCheckpointMonitor.Decision improved =
                monitor.evaluate(3, 1.0, Map.of("accuracy", 0.82));

        assertTrue(improved.shouldSave());
        monitor.markSaved(improved);
        assertEquals(3, monitor.state().epoch());
        assertEquals(1.0, monitor.state().validationLoss(), 1e-6);
        assertEquals(0.82, monitor.state().monitorValue(), 1e-6);
    }

    @Test
    void nonFiniteLossOrMetricDoesNotRequestSave() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                true,
                "f1",
                CanonicalTrainer.BestModelMonitorMode.MAX,
                0.0);

        assertFalse(monitor.evaluate(1, Double.NaN, Map.of("f1", 0.5)).shouldSave());
        assertFalse(monitor.evaluate(2, 0.5, Map.of("f1", Double.NaN)).shouldSave());
        assertFalse(monitor.state().saved());
    }

    @Test
    void missingMetricFailsWithActionableMessage() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                true,
                "accuracy",
                CanonicalTrainer.BestModelMonitorMode.MAX,
                0.0);

        IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                monitor.evaluate(1, 0.5, Map.of()));

        assertEquals("Best model monitor metric 'accuracy' is not available. "
                + "Add it with .metric(...) or trainingOptions().*Metric().", error.getMessage());
    }

    @Test
    void markRestoredUpdatesMetadataStateWithoutChangingBestCandidate() {
        TrainerBestModelCheckpointMonitor monitor = new TrainerBestModelCheckpointMonitor(
                true,
                null,
                CanonicalTrainer.BestModelMonitorMode.MIN,
                0.0);
        TrainerBestModelCheckpointMonitor.Decision decision = monitor.evaluate(4, 0.25, Map.of());
        monitor.markSaved(decision);

        monitor.markRestored();

        assertTrue(monitor.state().saved());
        assertTrue(monitor.state().restored());
        assertEquals(4, monitor.state().epoch());
        assertEquals(0.25, monitor.state().validationLoss(), 1e-6);
    }
}
