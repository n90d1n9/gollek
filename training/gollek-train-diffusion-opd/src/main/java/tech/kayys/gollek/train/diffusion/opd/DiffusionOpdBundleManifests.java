package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed loader utilities for DiffusionOPD bundle manifests.
 */
public final class DiffusionOpdBundleManifests {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private DiffusionOpdBundleManifests() {
    }

    public static DiffusionOpdBundleManifest load(Path manifestPath) {
        try {
            return fromMap(OBJECT_MAPPER.readValue(
                    manifestPath.toFile(),
                    new TypeReference<LinkedHashMap<String, Object>>() {
                    }));
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load bundle manifest from " + manifestPath + ".", exception);
        }
    }

    public static DiffusionOpdBundleManifest fromMap(Map<String, Object> raw) {
        return DiffusionOpdBundleManifest.fromMap(raw);
    }
}
