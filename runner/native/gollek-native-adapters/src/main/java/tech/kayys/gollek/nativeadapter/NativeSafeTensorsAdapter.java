package tech.kayys.gollek.nativeadapter;

import java.nio.file.Path;
import tech.kayys.gollek.safetensor.loader.SafetensorLoadResult;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

/**
 * Thin adapter intended for native callers. Construct with a SafetensorLoaderFacade
 * instance (preferred when running inside a CDI container). This adapter avoids
 * duplicating SafeTensors logic — it simply delegates to the existing loader.
 */
public final class NativeSafeTensorsAdapter {

    private final SafetensorLoaderFacade facade;

    public NativeSafeTensorsAdapter(SafetensorLoaderFacade facade) {
        this.facade = facade;
    }

    public SafetensorLoadResult loadCached(Path path) {
        return facade.loadCached(path);
    }

    public SafetensorShardSession open(Path path) {
        return facade.open(path);
    }

    /**
     * Convenience factory when the caller already has the facade instance.
     */
    public static NativeSafeTensorsAdapter of(SafetensorLoaderFacade facade) {
        return new NativeSafeTensorsAdapter(facade);
    }
}
