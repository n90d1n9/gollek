package tech.kayys.gollek.ml.hub;

import java.nio.file.Path;

/**
 * Configuration for model hub operations.
 */
public record HubConfig(
        String revision,
        Path cacheDir,
        String token,
        boolean forceDownload,
        int timeoutSeconds) {

    public static final HubConfig DEFAULT = new HubConfig(
            "main",
            Path.of(System.getProperty("user.home"), ".gollek", "models"),
            null,
            false,
            300);

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String revision = "main";
        private Path cacheDir = Path.of(System.getProperty("user.home"), ".gollek", "models");
        private String token;
        private boolean forceDownload;
        private int timeoutSeconds = 300;

        public Builder revision(String value) {
            this.revision = value;
            return this;
        }

        public Builder cacheDir(Path value) {
            this.cacheDir = value;
            return this;
        }

        public Builder cacheDir(String value) {
            this.cacheDir = Path.of(value);
            return this;
        }

        public Builder token(String value) {
            this.token = value;
            return this;
        }

        public Builder forceDownload(boolean value) {
            this.forceDownload = value;
            return this;
        }

        public Builder timeoutSeconds(int value) {
            this.timeoutSeconds = value;
            return this;
        }

        public HubConfig build() {
            return new HubConfig(revision, cacheDir, token, forceDownload, timeoutSeconds);
        }
    }
}
