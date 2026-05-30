package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Input-conditioned context shared across recursive reasoning transitions.
 */
public record RecursiveReasoningContext(
        String taskId,
        Tensor inputEmbedding,
        RecursiveReasoningConfig config) {

    public RecursiveReasoningContext {
        taskId = Objects.requireNonNullElse(taskId, "default");
        inputEmbedding = Objects.requireNonNull(inputEmbedding, "inputEmbedding must not be null");
        config = Objects.requireNonNull(config, "config must not be null");
    }
}
