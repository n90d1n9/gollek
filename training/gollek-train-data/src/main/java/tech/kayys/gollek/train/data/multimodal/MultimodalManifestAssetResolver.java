package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

final class MultimodalManifestAssetResolver {
    record Asset(Path path, String uri, byte[] bytes, String mimeType, String documentFormat) {
        Asset {
            uri = Objects.requireNonNull(uri, "uri must not be null");
            mimeType = Objects.requireNonNull(mimeType, "mimeType must not be null");
            if (path != null) {
                path = path.toAbsolutePath().normalize();
            }
            if (bytes != null) {
                bytes = bytes.clone();
            }
        }

        @Override
        public byte[] bytes() {
            return bytes == null ? null : bytes.clone();
        }
    }

    private final MultimodalManifestDataset.Options options;

    MultimodalManifestAssetResolver(MultimodalManifestDataset.Options options) {
        this.options = Objects.requireNonNull(options, "options must not be null");
    }

    Asset resolvePath(String rawPath, ModalityType modality, String mimeType, String documentFormat) throws IOException {
        Objects.requireNonNull(modality, "modality must not be null");
        String value = requireText(rawPath, "asset path");
        Path path = Path.of(value);
        if (path.isAbsolute() && !options.allowAbsolutePaths()) {
            throw new IllegalArgumentException("absolute asset paths are disabled: " + value);
        }
        Path normalized = (path.isAbsolute() ? path : options.assetRoot().resolve(path)).toAbsolutePath().normalize();
        if (!path.isAbsolute() && !normalized.startsWith(options.assetRoot())) {
            throw new IllegalArgumentException("relative asset path escapes asset root: " + value);
        }
        if (options.requireExistingFiles() && !Files.isRegularFile(normalized)) {
            throw new IllegalArgumentException("asset path must be an existing file: " + normalized);
        }
        String resolvedMime = resolveMimeType(normalized, modality, mimeType);
        String resolvedDocumentFormat = modality == ModalityType.DOCUMENT
                ? resolveDocumentFormat(normalized, documentFormat)
                : documentFormat;
        byte[] bytes = options.inlineBinaryAssets() && Files.isRegularFile(normalized)
                ? Files.readAllBytes(normalized)
                : null;
        return new Asset(normalized, normalized.toUri().toString(), bytes, resolvedMime, resolvedDocumentFormat);
    }

    Asset resolveUri(String rawUri, ModalityType modality, String mimeType, String documentFormat) {
        Objects.requireNonNull(modality, "modality must not be null");
        String uri = requireText(rawUri, "asset uri");
        URI.create(uri);
        return new Asset(null, uri, null, mimeOrDefault(mimeType), documentFormat);
    }

    private static String resolveMimeType(Path path, ModalityType modality, String explicit) {
        if (explicit != null && !explicit.isBlank()) {
            return explicit.trim();
        }
        return switch (modality) {
            case IMAGE -> MultimodalFileSupport.imageMimeType(path);
            case AUDIO -> MultimodalFileSupport.audioMimeType(path);
            case DOCUMENT -> MultimodalFileSupport.documentMimeType(path);
            case VIDEO -> MultimodalFileSupport.videoMimeType(path);
            default -> "application/octet-stream";
        };
    }

    private static String resolveDocumentFormat(Path path, String explicit) {
        if (explicit != null && !explicit.isBlank()) {
            return explicit.trim();
        }
        return MultimodalFileSupport.extension(path).orElse("binary");
    }

    private static String mimeOrDefault(String value) {
        return value == null || value.isBlank() ? "application/octet-stream" : value.trim();
    }

    private static String requireText(String value, String name) {
        String text = Objects.requireNonNull(value, name + " must not be null").trim();
        if (text.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return text;
    }
}
