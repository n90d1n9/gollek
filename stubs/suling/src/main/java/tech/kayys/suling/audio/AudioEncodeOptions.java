package tech.kayys.suling.audio;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public record AudioEncodeOptions(
        String format,
        int compressionLevel,
        int bitrateKbps,
        boolean verify,
        Map<String, String> metadata) {
    public AudioEncodeOptions {
        format = format == null || format.isBlank() ? "wav" : format.trim().toLowerCase(Locale.ROOT);
        compressionLevel = Math.max(0, Math.min(8, compressionLevel));
        bitrateKbps = Math.max(1, bitrateKbps);
        metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                metadata == null ? Map.of() : metadata));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String format = "wav";
        private int compressionLevel = 5;
        private int bitrateKbps = 192;
        private boolean verify;
        private Map<String, String> metadata = Map.of();

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder compressionLevel(int compressionLevel) {
            this.compressionLevel = compressionLevel;
            return this;
        }

        public Builder bitrateKbps(int bitrateKbps) {
            this.bitrateKbps = bitrateKbps;
            return this;
        }

        public Builder verify(boolean verify) {
            this.verify = verify;
            return this;
        }

        public Builder metadata(Map<String, String> metadata) {
            this.metadata = metadata;
            return this;
        }

        public AudioEncodeOptions build() {
            return new AudioEncodeOptions(format, compressionLevel, bitrateKbps, verify, metadata);
        }
    }
}
