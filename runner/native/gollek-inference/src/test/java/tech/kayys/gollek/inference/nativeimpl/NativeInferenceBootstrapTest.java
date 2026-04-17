package tech.kayys.gollek.inference.nativeimpl;

import io.quarkus.runtime.StartupEvent;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.tensor.ComputeKernel;
import tech.kayys.gollek.spi.tensor.ComputeKernelRegistry;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for NativeInferenceBootstrap.
 */
class NativeInferenceBootstrapTest {

    @BeforeEach
    @AfterEach
    void clearRegistry() {
        ComputeKernelRegistry.get().clear();
    }

    @Test
    @DisplayName("onStart selects and initializes a kernel without throwing")
    void testOnStartInitializesKernel() {
        NativeInferenceBootstrap bootstrap = new NativeInferenceBootstrap();

        // Should not throw
        assertDoesNotThrow(() -> bootstrap.onStart(new StartupEvent()));

        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        assertNotNull(kernel, "Expected a kernel to be selected");
        assertTrue(kernel.isAvailable(), "Selected kernel should be available");

        // Basic sanity about device name
        String name = kernel.deviceName();
        assertNotNull(name);
        assertTrue(name.length() > 0);
    }
}
