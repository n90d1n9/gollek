package tech.kayys.gollek.jupyter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

final class NotebookFiles {

    private NotebookFiles() {
    }

    record DiskUsageStats(long files, long directories, long bytes, long entriesScanned, boolean truncated) {}

    record TextPreview(long bytes, boolean truncated, String body) {}

    static DiskUsageStats computeDiskUsage(Path target, int maxEntries) throws IOException {
        if (Files.isRegularFile(target)) {
            return new DiskUsageStats(1, 0, Files.size(target), 1, false);
        }
        long files = 0;
        long directories = 0;
        long bytes = 0;
        long entriesScanned = 0;
        boolean truncated = false;
        try (Stream<Path> stream = Files.walk(target)) {
            java.util.Iterator<Path> entries = stream.iterator();
            while (entries.hasNext()) {
                Path entry = entries.next();
                if (entry.equals(target)) {
                    continue;
                }
                if (entriesScanned >= maxEntries) {
                    truncated = true;
                    break;
                }
                entriesScanned++;
                if (Files.isDirectory(entry)) {
                    directories++;
                } else if (Files.isRegularFile(entry)) {
                    files++;
                    bytes += Files.size(entry);
                }
            }
        }
        return new DiskUsageStats(files, directories, bytes, entriesScanned, truncated);
    }

    static String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        String[] units = {"KiB", "MiB", "GiB", "TiB"};
        double value = bytes;
        int unitIndex = -1;
        do {
            value /= 1024.0;
            unitIndex++;
        } while (value >= 1024.0 && unitIndex < units.length - 1);
        return String.format(Locale.ROOT, "%.1f %s", value, units[unitIndex]);
    }

    static TextPreview readUtf8Preview(Path target, int maxBytes) throws IOException {
        byte[] bytes = Files.readAllBytes(target);
        boolean truncated = bytes.length > maxBytes;
        byte[] previewBytes = truncated ? Arrays.copyOf(bytes, maxBytes) : bytes;
        return new TextPreview(bytes.length, truncated, new String(previewBytes, StandardCharsets.UTF_8));
    }

    static void collectGrepMatches(Path file, String label, String pattern, List<String> matches, int maxMatches) {
        try {
            List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            for (int i = 0; i < lines.size(); i++) {
                if (lines.get(i).contains(pattern)) {
                    matches.add(label + ":" + (i + 1) + ": " + lines.get(i));
                    if (matches.size() >= maxMatches) {
                        return;
                    }
                }
            }
        } catch (Exception ignored) {
        }
    }

    static String formatPathEntry(Path entry) {
        String name = entry.getFileName().toString();
        return Files.isDirectory(entry) ? name + "/" : name;
    }
}
