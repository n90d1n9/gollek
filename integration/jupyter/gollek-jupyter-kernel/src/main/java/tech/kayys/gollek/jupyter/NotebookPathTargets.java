package tech.kayys.gollek.jupyter;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Validates resolved notebook paths while preserving magic-specific error wording.
 */
final class NotebookPathTargets {

    private NotebookPathTargets() {
    }

    static String directoryTargetError(Path target, String missingPrefix, String notDirectoryPrefix) {
        if (!Files.exists(target)) {
            return missingPrefix + target;
        }
        if (!Files.isDirectory(target)) {
            return notDirectoryPrefix + target;
        }
        return null;
    }

    static String fileOrDirectoryTargetError(Path target, String missingPrefix, String invalidPrefix) {
        if (!Files.exists(target)) {
            return missingPrefix + target;
        }
        if (!Files.isRegularFile(target) && !Files.isDirectory(target)) {
            return invalidPrefix + target;
        }
        return null;
    }

    static String regularFileTargetError(Path target, String missingPrefix, String notFilePrefix) {
        if (!Files.exists(target)) {
            return missingPrefix + target;
        }
        if (!Files.isRegularFile(target)) {
            return notFilePrefix + target;
        }
        return null;
    }
}
