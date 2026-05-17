package tech.kayys.gollek.hub;

import java.nio.file.Path;

/**
 * @deprecated Use {@link tech.kayys.gollek.ml.hub.HubConfig} instead.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public record HubConfig(
        String revision,
        Path cacheDir,
        String token,
        boolean forceDownload,
        int timeoutSeconds) {

    public static final HubConfig DEFAULT = fromCanonical(tech.kayys.gollek.ml.hub.HubConfig.DEFAULT);

    public tech.kayys.gollek.ml.hub.HubConfig toCanonical() {
        return new tech.kayys.gollek.ml.hub.HubConfig(revision, cacheDir, token, forceDownload, timeoutSeconds);
    }

    public static HubConfig fromCanonical(tech.kayys.gollek.ml.hub.HubConfig config) {
        return new HubConfig(
                config.revision(),
                config.cacheDir(),
                config.token(),
                config.forceDownload(),
                config.timeoutSeconds());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final tech.kayys.gollek.ml.hub.HubConfig.Builder delegate =
                tech.kayys.gollek.ml.hub.HubConfig.builder();

        public Builder revision(String value) {
            delegate.revision(value);
            return this;
        }

        public Builder cacheDir(Path value) {
            delegate.cacheDir(value);
            return this;
        }

        public Builder cacheDir(String value) {
            delegate.cacheDir(value);
            return this;
        }

        public Builder token(String value) {
            delegate.token(value);
            return this;
        }

        public Builder forceDownload(boolean value) {
            delegate.forceDownload(value);
            return this;
        }

        public Builder timeoutSeconds(int value) {
            delegate.timeoutSeconds(value);
            return this;
        }

        public HubConfig build() {
            return fromCanonical(delegate.build());
        }
    }
}
