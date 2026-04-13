package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import tech.kayys.gollek.cli.GollekHome;

final class LocalModelIndex {

    static final class Entry {
        public String id;
        public String name;
        public String format;
        public boolean runnable;
        public long sizeBytes;
        public String path;
        public String updatedAt;
        public String source;
    }

    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private static final Path INDEX_PATH = GollekHome.path("models", "index.json");

    private LocalModelIndex() {
    }

    static synchronized List<Entry> refreshFromDisk() {
        List<Entry> entries = scanDiskEntries();
        write(entries);
        return entries;
    }

    static synchronized Optional<Entry> find(String ref) {
        if (ref == null || ref.isBlank()) {
            return Optional.empty();
        }
        String needle = ref.trim();
        List<Entry> entries = readOrRefresh();
        return entries.stream().filter(e -> matches(e, needle)).findFirst();
    }

    private static boolean matches(Entry e, String ref) {
        if (e == null) {
            return false;
        }
        if (ref.equals(e.id) || ref.equals(e.name) || ref.equals(e.path)) {
            return true;
        }
        String normalized = ref.replace("\\", "/").toLowerCase(Locale.ROOT);
        String path = e.path != null ? e.path.replace("\\", "/").toLowerCase(Locale.ROOT) : "";
        return path.endsWith(normalized);
    }

    private static List<Entry> readOrRefresh() {
        try {
            if (Files.exists(INDEX_PATH)) {
                byte[] bytes = Files.readAllBytes(INDEX_PATH);
                if (bytes.length > 0) {
                    Entry[] parsed = JSON.readValue(bytes, Entry[].class);
                    return new ArrayList<>(List.of(parsed));
                }
            }
        } catch (Exception ignored) {
            // fallback to refresh
        }
        return refreshFromDisk();
    }

    private static void write(List<Entry> entries) {
        try {
            Files.createDirectories(INDEX_PATH.getParent());
            JSON.writeValue(INDEX_PATH.toFile(), entries);
        } catch (Exception ignored) {
            // best effort cache/index only
        }
    }

    private static List<Entry> scanDiskEntries() {
        List<Entry> out = new ArrayList<>();
        Path root = GollekHome.path("models");
        if (!Files.isDirectory(root)) {
            return out;
        }
        scanFlat(root.resolve("gguf"), "gguf", true, out);
        scanFlat(root.resolve("libtorchscript"), "libtorchscript", true, out);
        scanFlat(root.resolve("safetensors"), "safetensors", false, out);
        scanFlat(root.resolve("litert"), "litert", true, out);

        out.sort(Comparator.comparing((Entry e) -> parseInstant(e.updatedAt)).reversed());
        return out;
    }

    private static void scanFlat(Path base, String fallbackFormat, boolean runnable, List<Entry> out) {
        if (!Files.isDirectory(base)) {
            return;
        }
        try (var files = Files.walk(base, 4)) {
            files.filter(Files::isRegularFile)
                    .filter(p -> !p.getFileName().toString().startsWith("."))
                    .filter(LocalModelIndex::isLikelyWeightFile)
                    .forEach(p -> out.add(toEntry(base, p, fallbackFormat, runnable)));
        } catch (Exception ignored) {
            // best effort
        }
    }

    private static Entry toEntry(Path base, Path file, String fallbackFormat, boolean runnable) {
        Entry e = new Entry();
        e.id = base.relativize(file).toString().replace("\\", "/");
        e.name = file.getFileName().toString();
        e.format = detectFormat(file, fallbackFormat);
        e.runnable = runnable && !e.format.equalsIgnoreCase("safetensors") && !e.format.equalsIgnoreCase("bin");
        e.path = file.toAbsolutePath().toString();
        e.source = "local";
        try {
            e.sizeBytes = Files.size(file);
            e.updatedAt = Files.getLastModifiedTime(file).toInstant().toString();
        } catch (Exception ignored) {
            e.sizeBytes = 0L;
        }
        return e;
    }

    private static boolean isLikelyWeightFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".gguf")
                || name.endsWith(".safetensors")
                || name.endsWith(".safetensor")
                || name.endsWith(".pt")
                || name.endsWith(".pth")
                || name.endsWith(".bin")
                || name.endsWith(".litertlm")
                || name.endsWith(".task")
                || !name.contains(".");
    }

    private static String detectFormat(Path file, String fallback) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return "safetensors";
        }
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return "libtorchscript";
        }
        if (name.endsWith(".bin")) {
            return "bin";
        }
        if (name.endsWith(".litertlm") || name.endsWith(".task")) {
            return "litert";
        }
        return fallback;
    }

    static Instant parseInstant(String value) {
        if (value == null || value.isBlank()) {
            return Instant.EPOCH;
        }
        try {
            return Instant.parse(value);
        } catch (Exception ignored) {
            return Instant.EPOCH;
        }
    }
}
