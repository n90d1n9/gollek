package tech.kayys.gollek.spi.model;

import java.util.Locale;
import java.util.Map;

/**
 * Packaging profile for deciding which model-family plugins belong in a build.
 */
public enum ModelFamilyBundleProfile {
    CORE("core", true),
    OPTIONAL("optional", false),
    METADATA_ONLY("metadata_only", false),
    EXPERIMENTAL("experimental", false);

    public static final String METADATA_KEY = "bundle_profile";

    private final String key;
    private final boolean defaultBundle;

    ModelFamilyBundleProfile(String key, boolean defaultBundle) {
        this.key = key;
        this.defaultBundle = defaultBundle;
    }

    public String key() {
        return key;
    }

    public boolean defaultBundle() {
        return defaultBundle;
    }

    public static ModelFamilyBundleProfile fromMetadata(Map<String, String> metadata) {
        String value = metadata == null ? null : metadata.get(METADATA_KEY);
        if (value == null || value.isBlank()) {
            return OPTIONAL;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        for (ModelFamilyBundleProfile profile : values()) {
            if (profile.key.equals(normalized)) {
                return profile;
            }
        }
        return OPTIONAL;
    }
}
