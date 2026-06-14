package tech.kayys.gollek.plugin.cache;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.execution.ExecutionToken;
import tech.kayys.gollek.spi.execution.ExecutionStatus;
import tech.kayys.gollek.spi.exception.PluginException;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

class SemanticCachePluginTest {

    @Test
    void testPluginPhaseAndOrder() {
        SemanticCachePlugin plugin = new SemanticCachePlugin();
        assertEquals(InferencePhase.PRE_PROCESSING, plugin.phase());
        assertEquals(10, plugin.order());
    }

    @Test
    void testExecutionShortCircuit() throws PluginException {
        SemanticCachePlugin plugin = new SemanticCachePlugin();
        MockExecutionContext context = new MockExecutionContext();
        
        // This is a skeleton test. We just ensure it runs without throwing exceptions.
        plugin.execute(context, null);
        
        // For the skeleton, checkVectorStore returns false
        assertNull(context.metadata().get("shortCircuit"));
    }

    private static class MockExecutionContext implements ExecutionContext {
        private final Map<String, Object> metadata = new HashMap<>();
        private final Map<String, Object> variables = new HashMap<>();

        @Override public EngineContext engine() { return null; }
        @Override public ExecutionToken token() { return null; }
        @Override public RequestContext requestContext() { return null; }
        @Override public void updateStatus(ExecutionStatus status) {}
        @Override public void updatePhase(InferencePhase phase) {}
        @Override public void incrementAttempt() {}
        @Override public Map<String, Object> variables() { return variables; }
        @Override public void putVariable(String key, Object value) { variables.put(key, value); }
        @SuppressWarnings("unchecked")
        @Override public <T> Optional<T> getVariable(String key, Class<T> type) {
            return Optional.ofNullable((T) variables.get(key));
        }
        @Override public void removeVariable(String key) { variables.remove(key); }
        @Override public Map<String, Object> metadata() { return metadata; }
        @Override public void putMetadata(String key, Object value) { metadata.put(key, value); }
        @Override public void replaceToken(ExecutionToken newToken) {}
        @Override public boolean hasError() { return false; }
        @Override public Optional<Throwable> getError() { return Optional.empty(); }
        @Override public void setError(Throwable error) {}
        @Override public void clearError() {}
    }
}
