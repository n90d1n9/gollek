package tech.kayys.gollek.jupyter;

import java.util.List;

final class NotebookFileMagicOptions {

    private NotebookFileMagicOptions() {
    }

    record LinePreviewOptions(int lines, String path) {}

    record SearchOptions(String pattern, String path) {}

    record ExtractOptions(boolean dryRun, String archivePath, String entryPath, String outputPath) {}

    static LinePreviewOptions parseLinePreviewOptions(String magicName, String raw) {
        String usage = "Usage: %" + magicName + " [-n N] <PATH>";
        String pathPart = raw == null ? "" : raw.trim();
        int lineCount = 10;
        if (pathPart.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        if (pathPart.startsWith("-n ")) {
            String remainder = pathPart.substring(3).trim();
            int split = remainder.indexOf(' ');
            if (split <= 0) {
                throw new IllegalArgumentException(usage);
            }
            String countToken = remainder.substring(0, split).trim();
            try {
                lineCount = Integer.parseInt(countToken);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid line count for %" + magicName + ": " + countToken);
            }
            pathPart = remainder.substring(split + 1).trim();
        }
        if (lineCount <= 0) {
            throw new IllegalArgumentException("Line count for %" + magicName + " must be > 0");
        }
        if (pathPart.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        return new LinePreviewOptions(lineCount, pathPart);
    }

    static SearchOptions parseSearchOptions(String magicName, String raw) {
        String usage = "Usage: %" + magicName + " <TEXT> [PATH]";
        String trimmed = raw == null ? "" : raw.trim();
        if (trimmed.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        String pattern;
        String pathPart = null;
        int split = trimmed.indexOf(' ');
        if (split < 0) {
            pattern = trimmed;
        } else {
            pattern = trimmed.substring(0, split).trim();
            pathPart = trimmed.substring(split + 1).trim();
            if (pathPart.isBlank()) {
                pathPart = null;
            }
        }
        if (pattern.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        return new SearchOptions(pattern, pathPart);
    }

    static ExtractOptions parseExtractOptions(String raw) {
        String usage = "Usage: %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]";
        String args = raw == null ? "" : raw.trim();
        if (args.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        boolean dryRun = false;
        if (args.startsWith("--dry-run ")) {
            dryRun = true;
            args = args.substring("--dry-run".length()).trim();
        } else if (args.startsWith("-n ")) {
            dryRun = true;
            args = args.substring("-n".length()).trim();
        }
        List<String> parts = List.of(args.split("\\s+"));
        if (parts.size() < 2 || parts.size() > 3 || parts.stream().anyMatch(String::isBlank)) {
            throw new IllegalArgumentException(usage);
        }
        return new ExtractOptions(dryRun, parts.get(0), parts.get(1), parts.size() == 3 ? parts.get(2) : null);
    }
}
