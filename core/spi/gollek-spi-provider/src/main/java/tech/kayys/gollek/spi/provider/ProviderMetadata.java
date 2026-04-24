package tech.kayys.gollek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Provider metadata and identification
 */
public final class ProviderMetadata {

    private final String providerId;
    private final String name;
    private final String version;
    private final String description;
    private final String vendor;
    private final String homepage;
    private final String defaultModel;
    private final Map<String, String> capabilities;

    @JsonCreator
    public ProviderMetadata(
            @JsonProperty("providerId") String providerId,
            @JsonProperty("name") String name,
            @JsonProperty("version") String version,
            @JsonProperty("description") String description,
            @JsonProperty("vendor") String vendor,
            @JsonProperty("homepage") String homepage,
            @JsonProperty("defaultModel") String defaultModel,
            @JsonProperty("capabilities") Map<String, String> capabilities) {
        this.providerId = Objects.requireNonNull(providerId, "providerId");
        this.name = Objects.requireNonNull(name, "name");
        this.version = Objects.requireNonNull(version, "version");
        this.description = description;
        this.vendor = vendor;
        this.homepage = homepage;
        this.defaultModel = defaultModel;
        this.capabilities = capabilities != null ? Map.copyOf(capabilities) : Collections.emptyMap();
    }

    public ProviderMetadata(
            String providerId,
            String name,
            String version,
            String description,
            String vendor,
            String homepage,
            String defaultModel) {
        this(providerId, name, version, description, vendor, homepage, defaultModel, null);
    }

    // Getters
    public String getProviderId() {
        return providerId;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getDefaultModel() {
        return defaultModel;
    }

    public String getDescription() {
        return description;
    }

    public String getVendor() {
        return vendor;
    }

    public String getHomepage() {
        return homepage;
    }

    public Map<String, String> getCapabilities() {
        return capabilities;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String providerId;
        private String name;
        private String version;
        private String description;
        private String vendor;
        private String homepage;
        private String defaultModel;
        private final Map<String, String> capabilities = new HashMap<>();

        public Builder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder vendor(String vendor) {
            this.vendor = vendor;
            return this;
        }

        public Builder homepage(String homepage) {
            this.homepage = homepage;
            return this;
        }

        public Builder defaultModel(String defaultModel) {
            this.defaultModel = defaultModel;
            return this;
        }

        public Builder capability(String key, String value) {
            this.capabilities.put(key, value);
            return this;
        }

        public ProviderMetadata build() {
            return new ProviderMetadata(
                    providerId, name, version, description, vendor, homepage, defaultModel, capabilities);
        }
    }

    @Override
    public String toString() {
        return "ProviderMetadata{" +
                "providerId='" + providerId + '\'' +
                ", name='" + name + '\'' +
                ", version='" + version + '\'' +
                '}';
    }
}