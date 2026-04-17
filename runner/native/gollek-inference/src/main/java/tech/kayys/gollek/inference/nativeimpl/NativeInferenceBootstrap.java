package tech.kayys.gollek.inference.nativeimpl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import io.quarkus.runtime.StartupEvent;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.tensor.ComputeKernelRegistry;
import tech.kayys.gollek.spi.tensor.ComputeKernel;

/**
 * Bootstrap for native inference runner. Registers providers and performs lightweight
 * initialization at application startup.
 */
@ApplicationScoped
public class NativeInferenceBootstrap {
    private static final Logger log = Logger.getLogger(NativeInferenceBootstrap.class);

    void onStart(@Observes StartupEvent ev) {
        log.info("Native inference bootstrap starting");

        ComputeKernelRegistry registry = ComputeKernelRegistry.get();

        // Enumerate and log available kernels (if any)
        try {
            var available = registry.getAllAvailable();
            if (available == null || available.isEmpty()) {
                log.warn("No compute kernels detected; CPU fallback will be used.");
            } else {
                for (ComputeKernel k : available) {
                    log.infof("Available kernel: %s (%s) total=%d avail=%d",
                            k.deviceType(), k.deviceName(), k.totalMemory(), k.availableMemory());
                }
            }
        } catch (Exception e) {
            log.debugf(e, "Failed to enumerate kernels: %s", e.getMessage());
        }

        ComputeKernel kernel = registry.getBestAvailable();
        if (kernel == null) {
            log.error("Failed to select a compute kernel. Native inference disabled.");
            return;
        }

        try {
            // Initialize kernel (may load native libraries / drivers)
            kernel.initialize();
            log.infof("Initialized kernel: %s (%s)", kernel.deviceType(), kernel.deviceName());

            // Lightweight warm-up: small allocation to ensure native bindings are ready
            try {
                var tmp = kernel.allocate(1024);
                kernel.free(tmp);
                log.debug("Performed native kernel warm-up allocation.");
            } catch (Throwable t) {
                log.debugf(t, "Kernel warm-up allocation failed: %s", t.getMessage());
            }

            // Register JVM shutdown hook to cleanly shutdown kernel
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    log.infof("Shutting down kernel: %s (%s)", kernel.deviceType(), kernel.deviceName());
                    kernel.shutdown();
                } catch (Exception e) {
                    log.warnf(e, "Error shutting down kernel: %s", e.getMessage());
                }
            }, "gollek-native-kernel-shutdown"));

        } catch (Exception e) {
            log.errorf(e, "Failed to initialize selected kernel: %s", e.getMessage());
        }

        // Ensure useful adapters are loaded so static initializers run
        try {
            Class.forName("tech.kayys.gollek.inference.nativeimpl.FlashAttentionAdapter");
            log.debug("Loaded FlashAttentionAdapter");
        } catch (ClassNotFoundException ignored) {
            // ignore
        }

        log.info("Native inference bootstrap complete");
    }
}
