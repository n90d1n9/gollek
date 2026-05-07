package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;
import tech.kayys.gollek.plugin.runner.RunnerSession;

/**
 * Backend interface for GGUF execution.
 */
public interface GgufBackend {
    <T> RunnerResult<T> execute(RunnerRequest request);
    void close();
}
