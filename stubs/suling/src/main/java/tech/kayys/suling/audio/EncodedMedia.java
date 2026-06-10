package tech.kayys.suling.audio;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public record EncodedMedia(byte[] bytes, String format, String mimeType, Map<String, String> metadata) {
    public EncodedMedia {
        Objects.requireNonNull(bytes, "bytes must not be null");
        if (bytes.length == 0) {
            throw new IllegalArgumentException("bytes must not be empty");
        }
        format = Objects.requireNonNull(format, "format must not be null");
        mimeType = Objects.requireNonNull(mimeType, "mimeType must not be null");
        bytes = bytes.clone();
        metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                metadata == null ? Map.of() : metadata));
    }

    @Override
    public byte[] bytes() {
        return bytes.clone();
    }
}
