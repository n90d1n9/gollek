package tech.kayys.gollek.engine;

import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeReflection;
import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;

/**
 * Feature to configure GraalVM Native Image generation for the Gollek Engine.
 * Registers necessary classes for reflection to support AOT compilation.
 */
@RegisterForReflection(targets = {
    InferenceRequest.class,
    ProviderConfig.class,
    Uni.class,
    Multi.class
})
public class NativeImageFeature implements Feature {

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        // Register additional classes or specific Mutiny internals if needed
        RuntimeReflection.register(InferenceRequest.class);
        RuntimeReflection.register(ProviderConfig.class);
        
        System.out.println("Gollek NativeImageFeature: Registered SPI classes for reflection.");
    }
}
