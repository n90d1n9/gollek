package tech.kayys.gollek.train.diffusion.api;

import java.util.List;
import java.util.Map;

/**
 * Typed top-level schema for the normalized DiffusionOPD report artifact.
 */
public record DiffusionOpdReport(
        DiffusionOpdRunReport run,
        DiffusionOpdArtifactsReport artifacts,
        Map<String, Object> teachers,
        Map<String, Object> stages,
        Map<String, Object> tasks,
        Map<String, Object> conditioning,
        Map<String, Object> adaptive,
        Map<String, Object> bindings,
        List<Map<String, Object>> roundHistory) {

    public Map<String, Object> asMap() {
        return Map.of(
                "run", run.asMap(),
                "artifacts", artifacts.asMap(),
                "teachers", teachers,
                "stages", stages,
                "tasks", tasks,
                "conditioning", conditioning,
                "adaptive", adaptive,
                "bindings", bindings,
                "roundHistory", roundHistory);
    }
}
