package tech.kayys.gollek.train.diffusion.api;

import java.util.Map;

/**
 * Typed artifact-location section for the normalized DiffusionOPD report.
 */
public record DiffusionOpdArtifactsReport(
        String summaryFile,
        String historyFile,
        String reportFile,
        String checkpointDir) {

    public Map<String, Object> asMap() {
        return Map.of(
                "summaryFile", summaryFile,
                "historyFile", historyFile,
                "reportFile", reportFile,
                "checkpointDir", checkpointDir);
    }
}
