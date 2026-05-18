package tech.kayys.gollek.train.diffusion.api;

import java.util.Map;

/**
 * Typed run-summary section for the normalized DiffusionOPD report.
 */
public record DiffusionOpdRunReport(
        int epochCount,
        Double latestTrainLoss,
        long durationMs,
        Object samplerType,
        Object taskCount,
        Object optimizationSteps,
        Object roundsCompleted,
        Object stopped) {

    public Map<String, Object> asMap() {
        return Map.of(
                "epochCount", epochCount,
                "latestTrainLoss", latestTrainLoss,
                "durationMs", durationMs,
                "samplerType", samplerType,
                "taskCount", taskCount,
                "optimizationSteps", optimizationSteps,
                "roundsCompleted", roundsCompleted,
                "stopped", stopped);
    }
}
