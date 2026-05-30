package tech.kayys.gollek.spi.pipeline;

import java.util.List;
import java.util.Optional;

/**
 * Registry for feature pipelines contributed by core modules, internal feature
 * modules, or external projects.
 */
public interface ModelPipelineRegistry {

    /**
     * All known pipelines ordered by descending priority.
     */
    List<ModelPipeline> all();

    /**
     * Select the highest-priority pipeline that supports the request.
     */
    Optional<ModelPipeline> select(ModelPipelineRequest request);
}
