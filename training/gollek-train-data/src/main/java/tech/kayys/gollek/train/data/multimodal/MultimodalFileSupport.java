package tech.kayys.gollek.train.data.multimodal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

final class MultimodalFileSupport {
    static final Set<String> IMAGE_EXTENSIONS = Set.of("jpg", "jpeg", "png");
    static final Set<String> AUDIO_EXTENSIONS = Set.of("wav", "mp3", "flac", "ogg", "m4a");

    private MultimodalFileSupport() {
    }

    static List<Path> regularFilesWithExtensions(Path directory, Set<String> extensions) throws IOException {
        Path root = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            throw new IllegalArgumentException("directory must be an existing directory: " + root);
        }
        try (var stream = Files.walk(root)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path -> extension(path).filter(extensions::contains).isPresent())
                    .map(path -> path.toAbsolutePath().normalize())
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
    }

    static Optional<String> extension(Path path) {
        Objects.requireNonNull(path, "path must not be null");
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) {
            return Optional.empty();
        }
        return Optional.of(name.substring(dot + 1).toLowerCase(Locale.ROOT));
    }

    static Path sidecarTextPath(Path mediaPath) {
        Objects.requireNonNull(mediaPath, "mediaPath must not be null");
        String name = mediaPath.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String baseName = dot < 0 ? name : name.substring(0, dot);
        return mediaPath.getParent().resolve(baseName + ".txt").toAbsolutePath().normalize();
    }

    static String audioMimeType(Path path) {
        return extension(path)
                .map(ext -> switch (ext) {
                    case "wav" -> "audio/wav";
                    case "mp3" -> "audio/mpeg";
                    case "flac" -> "audio/flac";
                    case "ogg" -> "audio/ogg";
                    case "m4a" -> "audio/mp4";
                    default -> "application/octet-stream";
                })
                .orElse("application/octet-stream");
    }

    static String imageMimeType(Path path) {
        return extension(path)
                .map(ext -> switch (ext) {
                    case "jpg", "jpeg" -> "image/jpeg";
                    case "png" -> "image/png";
                    case "webp" -> "image/webp";
                    case "gif" -> "image/gif";
                    default -> "application/octet-stream";
                })
                .orElse("application/octet-stream");
    }

    static String documentMimeType(Path path) {
        return extension(path)
                .map(ext -> switch (ext) {
                    case "pdf" -> "application/pdf";
                    case "docx" -> "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
                    case "html", "htm" -> "text/html";
                    case "md", "markdown" -> "text/markdown";
                    case "txt" -> "text/plain";
                    case "json", "jsonl" -> "application/json";
                    case "csv" -> "text/csv";
                    default -> "application/octet-stream";
                })
                .orElse("application/octet-stream");
    }

    static String videoMimeType(Path path) {
        return extension(path)
                .map(ext -> switch (ext) {
                    case "mp4", "m4v" -> "video/mp4";
                    case "mov" -> "video/quicktime";
                    case "webm" -> "video/webm";
                    default -> "application/octet-stream";
                })
                .orElse("application/octet-stream");
    }
}
