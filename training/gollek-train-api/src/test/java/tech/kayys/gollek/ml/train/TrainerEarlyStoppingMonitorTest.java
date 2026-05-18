package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerEarlyStoppingMonitorTest {

    @Test
    void disabledMonitorKeepsInitialStateAndDoesNotStop() {
        TrainerEarlyStoppingMonitor monitor = new TrainerEarlyStoppingMonitor(
                null,
                2,
                0.0,
                CanonicalTrainer.BestModelMonitorMode.MAX);

        TrainerEarlyStoppingMonitor.UpdateResult result =
                monitor.update(1, 5.0, Map.of("accuracy", 0.9));

        assertFalse(result.shouldStop());
        assertFalse(result.improved());
        assertFalse(monitor.triggered());
        assertEquals(-1, monitor.stopEpoch());
        assertEquals(-1, monitor.state().bestEpoch());
        assertTrue(Double.isNaN(monitor.state().bestValue()));
    }

    @Test
    void missingConfiguredMetricFailsWithActionableMessage() {
        TrainerEarlyStoppingMonitor monitor = new TrainerEarlyStoppingMonitor(
                "accuracy",
                2,
                0.0,
                CanonicalTrainer.BestModelMonitorMode.MAX);

        IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                monitor.update(1, 5.0, Map.of()));

        assertEquals("Early stopping monitor metric 'accuracy' is not available. "
                + "Add it with .metric(...) or trainingOptions().*Metric().", error.getMessage());
    }

    @Test
    void maxModeTracksImprovementAndStopsAfterPatience() {
        TrainerEarlyStoppingMonitor monitor = new TrainerEarlyStoppingMonitor(
                "accuracy",
                2,
                0.01,
                CanonicalTrainer.BestModelMonitorMode.MAX);

        assertTrue(monitor.update(1, 0.0, Map.of("accuracy", 0.80)).improved());
        assertFalse(monitor.update(2, 0.0, Map.of("accuracy", 0.805)).shouldStop());
        TrainerEarlyStoppingMonitor.UpdateResult stop =
                monitor.update(3, 0.0, Map.of("accuracy", 0.81));

        assertTrue(stop.shouldStop());
        assertTrue(monitor.triggered());
        assertEquals(3, monitor.stopEpoch());
        assertEquals(1, monitor.state().bestEpoch());
        assertEquals(0.80, monitor.state().bestValue(), 1e-6);
        assertEquals(0.81, monitor.state().latestValue(), 1e-6);
        assertEquals(2, monitor.state().epochsWithoutImprovement());
    }

    @Test
    void minModeCountsLowerValuesAsImprovement() {
        TrainerEarlyStoppingMonitor monitor = new TrainerEarlyStoppingMonitor(
                "mae",
                1,
                0.05,
                CanonicalTrainer.BestModelMonitorMode.MIN);

        assertTrue(monitor.update(1, 0.0, Map.of("mae", 0.50)).improved());
        assertFalse(monitor.update(2, 0.0, Map.of("mae", 0.40)).shouldStop());

        assertEquals(2, monitor.state().bestEpoch());
        assertEquals(0.40, monitor.state().bestValue(), 1e-6);
        assertEquals(0, monitor.state().epochsWithoutImprovement());
    }

    @Test
    void nonFiniteMonitorValueUpdatesLatestButDoesNotAdvancePatience() {
        TrainerEarlyStoppingMonitor monitor = new TrainerEarlyStoppingMonitor(
                "f1",
                1,
                0.0,
                CanonicalTrainer.BestModelMonitorMode.MAX);

        TrainerEarlyStoppingMonitor.UpdateResult result =
                monitor.update(1, 0.0, Map.of("f1", Double.NaN));

        assertFalse(result.shouldStop());
        assertFalse(monitor.triggered());
        assertTrue(Double.isNaN(monitor.state().latestValue()));
        assertEquals(0, monitor.state().epochsWithoutImprovement());
    }
}
