package tech.kayys.gollek.jupyter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Provides reusable filesystem operations for notebook file magics.
 */
final class NotebookFiles {

    private NotebookFiles() {
    }

    record DiskUsageStats(long files, long directories, long bytes, long entriesScanned, boolean truncated) {}

    record TextPreview(long bytes, boolean truncated, String body) {}

    record LineWindow(List<String> lines, int totalLines) {}

    record FileStats(long lines, long words, int chars, long bytes) {}

    record Sha256Digest(long bytes, String hash) {}

    record DiffResult(int changed, int added, int removed, List<String> lines) {}

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

    static List<String> listDirectoryEntries(Path directory, int maxEntries) throws IOException {
        try (Stream<Path> stream = Files.list(directory)) {
            return stream
                    .sorted(Comparator.comparing(path -> path.getFileName().toString().toLowerCase(Locale.ROOT)))
                    .limit(maxEntries)
                    .map(NotebookFiles::formatPathEntry)
                    .toList();
        }
    }

    static List<String> treeLines(Path root, int maxEntries, int maxDepth) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("Tree(" + root + ")");
        try (Stream<Path> stream = Files.walk(root, maxDepth)) {
            stream
                    .filter(path -> !path.equals(root))
                    .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                    .limit(maxEntries)
                    .forEach(path -> {
                        Path relative = root.relativize(path);
                        int depth = relative.getNameCount();
                        String indent = "  ".repeat(Math.max(0, depth - 1));
                        lines.add(indent + formatPathEntry(path));
                    });
        }
        return List.copyOf(lines);
    }

    static LineWindow readHead(Path target, int lines) throws IOException {
        List<String> allLines = Files.readAllLines(target, StandardCharsets.UTF_8);
        return new LineWindow(
                allLines.stream().limit(lines).toList(),
                allLines.size()
        );
    }

    static LineWindow readTail(Path target, int lines) throws IOException {
        List<String> allLines = Files.readAllLines(target, StandardCharsets.UTF_8);
        int fromIndex = Math.max(0, allLines.size() - lines);
        return new LineWindow(
                List.copyOf(allLines.subList(fromIndex, allLines.size())),
                allLines.size()
        );
    }

    static FileStats computeFileStats(Path target) throws IOException {
        String text = Files.readString(target, StandardCharsets.UTF_8);
        long lines = text.isEmpty() ? 0 : text.lines().count();
        long words = text.isBlank() ? 0 : Arrays.stream(text.trim().split("\\s+"))
                .filter(token -> !token.isBlank())
                .count();
        return new FileStats(lines, words, text.length(), Files.size(target));
    }

    static Sha256Digest computeSha256(Path target) throws IOException, NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(target)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                if (read > 0) {
                    digest.update(buffer, 0, read);
                }
            }
        }
        return new Sha256Digest(Files.size(target), toHex(digest.digest()));
    }

    static DiffResult computeLineDiff(Path left, Path right) throws IOException {
        List<String> leftLines = Files.readAllLines(left, StandardCharsets.UTF_8);
        List<String> rightLines = Files.readAllLines(right, StandardCharsets.UTF_8);
        int max = Math.max(leftLines.size(), rightLines.size());
        List<String> diffLines = new ArrayList<>();
        int changed = 0;
        int added = 0;
        int removed = 0;
        for (int i = 0; i < max; i++) {
            String leftLine = i < leftLines.size() ? leftLines.get(i) : null;
            String rightLine = i < rightLines.size() ? rightLines.get(i) : null;
            if (Objects.equals(leftLine, rightLine)) {
                continue;
            }
            changed++;
            if (leftLine != null) {
                diffLines.add("- " + (i + 1) + ": " + leftLine);
                removed++;
            }
            if (rightLine != null) {
                diffLines.add("+ " + (i + 1) + ": " + rightLine);
                added++;
            }
        }
        return new DiffResult(changed, added, removed, List.copyOf(diffLines));
    }

    static List<String> collectTextMatches(Path target, String pattern, int maxFiles, int maxMatches, int maxDepth)
            throws IOException {
        List<String> matches = new ArrayList<>();
        if (Files.isRegularFile(target)) {
            collectGrepMatches(target, target.getFileName().toString(), pattern, matches, maxMatches);
            return matches;
        }
        try (Stream<Path> stream = Files.walk(target, maxDepth)) {
            stream.filter(Files::isRegularFile)
                    .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                    .limit(maxFiles)
                    .forEach(path -> collectGrepMatches(path, target.relativize(path).toString(), pattern, matches, maxMatches));
        }
        return matches;
    }

    static List<String> collectFileNameMatches(Path target, String pattern, int maxEntries, int maxDepth)
            throws IOException {
        List<String> matches = new ArrayList<>();
        if (Files.isRegularFile(target)) {
            String name = target.getFileName().toString();
            if (name.contains(pattern)) {
                matches.add(name);
            }
            return matches;
        }
        try (Stream<Path> stream = Files.walk(target, maxDepth)) {
            stream.filter(path -> !path.equals(target))
                    .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                    .limit(maxEntries)
                    .forEach(path -> {
                        String relative = target.relativize(path).toString();
                        if (relative.contains(pattern)) {
                            matches.add(Files.isDirectory(path) ? relative + "/" : relative);
                        }
                    });
        }
        return matches;
    }

    private static void collectGrepMatches(Path file, String label, String pattern, List<String> matches, int maxMatches) {
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

    private static String toHex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
    }
}
