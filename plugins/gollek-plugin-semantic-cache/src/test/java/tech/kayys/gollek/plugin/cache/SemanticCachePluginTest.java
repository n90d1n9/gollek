package tech.kayys.gollek.plugin.cache;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.exception.PluginException;

import java.util.HashMap;
import java.util.Map;

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
        
        // In a full implementation, we would mock the vector store and test that:
        // 1. If hit, shortCircuit = true
        // 2. If miss, shortCircuit = null/false
        
        // For the skeleton, checkVectorStore returns false
        assertNull(context.getMetadata().get("shortCircuit"));
    }

    private static class MockExecutionContext implements ExecutionContext {
        private final Map<String, Object> metadata = new HashMap<>();

        @Override
        public String getRequestId() { return "req-1"; }

        @Override
        public String getTenantId() { return "tenant-1"; }

        @Override
        public tech.kayys.gollek.spi.inference.InferenceRequest getRequest() { return null; }

        @Override
        public Map<String, Object> getMetadata() { return metadata; }

        @Override
        public void cancel() {}

        @Override
        public boolean isCancelled() { return false; }
    }
}
