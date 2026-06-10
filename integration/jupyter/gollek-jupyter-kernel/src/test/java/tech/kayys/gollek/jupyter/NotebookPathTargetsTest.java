package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies reusable path-target validation for notebook filesystem magics.
 */
class NotebookPathTargetsTest {

    @TempDir
    Path tempDir;

    @Test
    void regularFileTargetErrorReportsMissingAndDirectoryTargets() throws Exception {
        Path missing = tempDir.resolve("missing.txt");
        Path file = tempDir.resolve("notes.txt");
        Files.writeString(file, "ok", StandardCharsets.UTF_8);

        assertEquals(
                "File not found: " + missing,
                NotebookPathTargets.regularFileTargetError(missing, "File not found: ", "Not a file: ")
        );
        assertEquals(
                "Not a file: " + tempDir,
                NotebookPathTargets.regularFileTargetError(tempDir, "File not found: ", "Not a file: ")
        );
        assertNull(NotebookPathTargets.regularFileTargetError(file, "File not found: ", "Not a file: "));
    }

    @Test
    void directoryTargetErrorReportsMissingAndFileTargets() throws Exception {
        Path missing = tempDir.resolve("missing-dir");
        Path file = tempDir.resolve("payload.txt");
        Files.writeString(file, "ok", StandardCharsets.UTF_8);

        assertEquals(
                "Directory not found: " + missing,
                NotebookPathTargets.directoryTargetError(missing, "Directory not found: ", "Not a directory: ")
        );
        assertEquals(
                "Not a directory: " + file,
                NotebookPathTargets.directoryTargetError(file, "Directory not found: ", "Not a directory: ")
        );
        assertNull(NotebookPathTargets.directoryTargetError(tempDir, "Directory not found: ", "Not a directory: "));
    }

    @Test
    void fileOrDirectoryTargetErrorAcceptsFilesAndDirectoriesAndReportsMissingTargets() throws Exception {
        Path missing = tempDir.resolve("missing-target");
        Path file = tempDir.resolve("payload.txt");
        Files.writeString(file, "ok", StandardCharsets.UTF_8);

        assertEquals(
                "Target not found: " + missing,
                NotebookPathTargets.fileOrDirectoryTargetError(missing, "Target not found: ", "Invalid target: ")
        );
        assertNull(NotebookPathTargets.fileOrDirectoryTargetError(file, "Target not found: ", "Invalid target: "));
        assertNull(NotebookPathTargets.fileOrDirectoryTargetError(tempDir, "Target not found: ", "Invalid target: "));
    }
}
