package tech.kayys.gollek.jupyter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class NotebookArchives {

    private NotebookArchives() {
    }

    record ArchiveListing(List<String> entries, long bytes) {}

    record ArchiveEntry(boolean directory, long size, byte[] bytes) {}

    static ArchiveListing listZip(Path archive) throws IOException {
        List<String> entries = new ArrayList<>();
        long totalUncompressed = 0L;
        try (ZipFile zipFile = new ZipFile(archive.toFile())) {
            java.util.Enumeration<? extends ZipEntry> zipEntries = zipFile.entries();
            while (zipEntries.hasMoreElements()) {
                ZipEntry entry = zipEntries.nextElement();
                String label = entry.getName() + (entry.isDirectory() ? "/" : "");
                long size = entry.getSize();
                if (size > 0) {
                    totalUncompressed += size;
                }
                entries.add(label + (size >= 0 ? " (" + size + " bytes)" : ""));
            }
        }
        return new ArchiveListing(entries, totalUncompressed);
    }

    static ArchiveEntry readZipEntry(Path archive, String entryPart) throws IOException {
        try (ZipFile zipFile = new ZipFile(archive.toFile())) {
            ZipEntry entry = zipFile.getEntry(entryPart);
            if (entry == null) {
                return null;
            }
            if (entry.isDirectory()) {
                return new ArchiveEntry(true, 0L, new byte[0]);
            }
            try (InputStream in = zipFile.getInputStream(entry)) {
                byte[] bytes = in.readAllBytes();
                return new ArchiveEntry(false, bytes.length, bytes);
            }
        }
    }

    static ArchiveListing listTar(Path archive) throws IOException {
        List<String> entries = new ArrayList<>();
        long totalBytes = 0L;
        try (InputStream rawIn = Files.newInputStream(archive);
             InputStream in = isGzipArchive(archive) ? new GZIPInputStream(rawIn) : rawIn) {
            byte[] header = new byte[512];
            while (readFully(in, header, 0, header.length) == header.length) {
                if (isAllZeros(header)) {
                    break;
                }
                String name = readTarEntryName(header);
                long size = parseTarOctal(header, 124, 12);
                int type = header[156] & 0xff;
                boolean directory = type == '5' || name.endsWith("/");
                String label = directory ? name + (name.endsWith("/") ? "" : "/") : name;
                if (size > 0) {
                    totalBytes += size;
                }
                entries.add(label + (size >= 0 ? " (" + size + " bytes)" : ""));

                skipTarPayload(in, size);
            }
        }
        return new ArchiveListing(entries, totalBytes);
    }

    static ArchiveEntry readTarEntryPreview(Path archive, String entryPart, int maxBytes) throws IOException {
        try (InputStream rawIn = Files.newInputStream(archive);
             InputStream in = isGzipArchive(archive) ? new GZIPInputStream(rawIn) : rawIn) {
            byte[] header = new byte[512];
            while (readFully(in, header, 0, header.length) == header.length) {
                if (isAllZeros(header)) {
                    break;
                }
                String name = readTarEntryName(header);
                long size = parseTarOctal(header, 124, 12);
                int type = header[156] & 0xff;
                boolean directory = type == '5' || name.endsWith("/");
                if (!name.equals(entryPart)) {
                    skipTarPayload(in, size);
                    continue;
                }
                if (directory) {
                    return new ArchiveEntry(true, size, new byte[0]);
                }

                int previewLength = (int) Math.min(size, maxBytes);
                byte[] previewBytes = new byte[previewLength];
                int read = readFully(in, previewBytes, 0, previewLength);
                if (read < previewLength) {
                    previewBytes = Arrays.copyOf(previewBytes, Math.max(0, read));
                }
                return new ArchiveEntry(false, size, previewBytes);
            }
        }
        return null;
    }

    static ArchiveEntry readArchiveEntryBytes(Path archive, String entryPart, long maxBytes) throws IOException {
        return isTarArchive(archive)
                ? readTarEntryBytes(archive, entryPart, maxBytes)
                : readZipEntryBytes(archive, entryPart, maxBytes);
    }

    private static ArchiveEntry readZipEntryBytes(Path archive, String entryPart, long maxBytes) throws IOException {
        try (ZipFile zipFile = new ZipFile(archive.toFile())) {
            ZipEntry entry = zipFile.getEntry(entryPart);
            if (entry == null) {
                return null;
            }
            if (entry.isDirectory()) {
                return new ArchiveEntry(true, 0L, new byte[0]);
            }
            long size = entry.getSize();
            if (size > maxBytes) {
                return new ArchiveEntry(false, size, overflowSentinel(maxBytes));
            }
            try (InputStream in = zipFile.getInputStream(entry)) {
                byte[] bytes = in.readAllBytes();
                return new ArchiveEntry(false, bytes.length, bytes);
            }
        }
    }

    private static ArchiveEntry readTarEntryBytes(Path archive, String entryPart, long maxBytes) throws IOException {
        try (InputStream rawIn = Files.newInputStream(archive);
             InputStream in = isGzipArchive(archive) ? new GZIPInputStream(rawIn) : rawIn) {
            byte[] header = new byte[512];
            while (readFully(in, header, 0, header.length) == header.length) {
                if (isAllZeros(header)) {
                    break;
                }
                String name = readTarEntryName(header);
                long size = parseTarOctal(header, 124, 12);
                int type = header[156] & 0xff;
                boolean directory = type == '5' || name.endsWith("/");
                if (!name.equals(entryPart)) {
                    skipTarPayload(in, size);
                    continue;
                }
                if (directory) {
                    return new ArchiveEntry(true, size, new byte[0]);
                }
                if (size > maxBytes) {
                    return new ArchiveEntry(false, size, overflowSentinel(maxBytes));
                }
                byte[] bytes = new byte[(int) size];
                int read = readFully(in, bytes, 0, bytes.length);
                if (read < bytes.length) {
                    bytes = Arrays.copyOf(bytes, Math.max(0, read));
                }
                return new ArchiveEntry(false, size, bytes);
            }
        }
        return null;
    }

    private static boolean isGzipArchive(Path target) {
        String name = target.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".tgz") || name.endsWith(".tar.gz") || name.endsWith(".gz");
    }

    private static boolean isTarArchive(Path target) {
        String name = target.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".tar") || name.endsWith(".tar.gz") || name.endsWith(".tgz") || name.endsWith(".gz");
    }

    private static int readFully(InputStream in, byte[] buffer, int offset, int length) throws IOException {
        int total = 0;
        while (total < length) {
            int read = in.read(buffer, offset + total, length - total);
            if (read < 0) {
                break;
            }
            total += read;
        }
        return total;
    }

    private static void skipTarPayload(InputStream in, long size) throws IOException {
        skipFully(in, size);
        long padding = (512 - (size % 512)) % 512;
        skipFully(in, padding);
    }

    private static void skipFully(InputStream in, long bytes) throws IOException {
        long remaining = bytes;
        while (remaining > 0) {
            long skipped = in.skip(remaining);
            if (skipped <= 0) {
                if (in.read() == -1) {
                    break;
                }
                skipped = 1;
            }
            remaining -= skipped;
        }
    }

    private static boolean isAllZeros(byte[] buffer) {
        for (byte value : buffer) {
            if (value != 0) {
                return false;
            }
        }
        return true;
    }

    private static String readTarString(byte[] buffer, int offset, int length) {
        int end = offset;
        int limit = Math.min(buffer.length, offset + length);
        while (end < limit && buffer[end] != 0) {
            end++;
        }
        return new String(buffer, offset, Math.max(0, end - offset), StandardCharsets.UTF_8).trim();
    }

    private static long parseTarOctal(byte[] buffer, int offset, int length) {
        String text = readTarString(buffer, offset, length).trim();
        if (text.isEmpty()) {
            return 0L;
        }
        try {
            return Long.parseLong(text.replaceAll("[^0-7]", ""), 8);
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private static String readTarEntryName(byte[] header) {
        String name = readTarString(header, 0, 100);
        String prefix = readTarString(header, 345, 155);
        return prefix.isBlank() ? name : prefix + "/" + name;
    }

    private static byte[] overflowSentinel(long maxBytes) {
        return new byte[Math.toIntExact(maxBytes + 1)];
    }
}
