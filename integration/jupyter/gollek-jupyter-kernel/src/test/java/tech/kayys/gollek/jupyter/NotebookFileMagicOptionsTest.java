package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NotebookFileMagicOptionsTest {

    @Test
    void parseLinePreviewOptionsUsesDefaultLineCount() {
        NotebookFileMagicOptions.LinePreviewOptions options =
                NotebookFileMagicOptions.parseLinePreviewOptions("head", "notes.txt");

        assertEquals(10, options.lines());
        assertEquals("notes.txt", options.path());
    }

    @Test
    void parseLinePreviewOptionsReadsExplicitLineCount() {
        NotebookFileMagicOptions.LinePreviewOptions options =
                NotebookFileMagicOptions.parseLinePreviewOptions("tail", "-n 5 logs.txt");

        assertEquals(5, options.lines());
        assertEquals("logs.txt", options.path());
    }

    @Test
    void parseLinePreviewOptionsPreservesUsageAndValidationMessages() {
        assertError("Usage: %head [-n N] <PATH>",
                () -> NotebookFileMagicOptions.parseLinePreviewOptions("head", ""));
        assertError("Usage: %head [-n N] <PATH>",
                () -> NotebookFileMagicOptions.parseLinePreviewOptions("head", "-n 5"));
        assertError("Invalid line count for %tail: many",
                () -> NotebookFileMagicOptions.parseLinePreviewOptions("tail", "-n many logs.txt"));
        assertError("Line count for %tail must be > 0",
                () -> NotebookFileMagicOptions.parseLinePreviewOptions("tail", "-n 0 logs.txt"));
    }

    @Test
    void parseSearchOptionsReadsPatternAndOptionalPath() {
        NotebookFileMagicOptions.SearchOptions currentDirectory =
                NotebookFileMagicOptions.parseSearchOptions("grep", "needle");
        NotebookFileMagicOptions.SearchOptions explicitPath =
                NotebookFileMagicOptions.parseSearchOptions("findfile", "needle ./src");

        assertEquals("needle", currentDirectory.pattern());
        assertNull(currentDirectory.path());
        assertEquals("needle", explicitPath.pattern());
        assertEquals("./src", explicitPath.path());
    }

    @Test
    void parseSearchOptionsRejectsBlankInputWithMagicSpecificUsage() {
        assertError("Usage: %grep <TEXT> [PATH]",
                () -> NotebookFileMagicOptions.parseSearchOptions("grep", " "));
        assertError("Usage: %findfile <TEXT> [PATH]",
                () -> NotebookFileMagicOptions.parseSearchOptions("findfile", null));
    }

    @Test
    void parseExtractOptionsReadsRequiredAndOptionalPaths() {
        NotebookFileMagicOptions.ExtractOptions defaultOutput =
                NotebookFileMagicOptions.parseExtractOptions("archive.zip nested/file.txt");
        NotebookFileMagicOptions.ExtractOptions customOutput =
                NotebookFileMagicOptions.parseExtractOptions("--dry-run archive.zip nested/file.txt out/file.txt");
        NotebookFileMagicOptions.ExtractOptions shortDryRun =
                NotebookFileMagicOptions.parseExtractOptions("-n archive.zip nested/file.txt");

        assertEquals(false, defaultOutput.dryRun());
        assertEquals("archive.zip", defaultOutput.archivePath());
        assertEquals("nested/file.txt", defaultOutput.entryPath());
        assertNull(defaultOutput.outputPath());
        assertTrue(customOutput.dryRun());
        assertEquals("out/file.txt", customOutput.outputPath());
        assertTrue(shortDryRun.dryRun());
    }

    @Test
    void parseExtractOptionsPreservesUsageMessage() {
        assertError("Usage: %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]",
                () -> NotebookFileMagicOptions.parseExtractOptions(""));
        assertError("Usage: %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]",
                () -> NotebookFileMagicOptions.parseExtractOptions("archive.zip"));
        assertError("Usage: %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]",
                () -> NotebookFileMagicOptions.parseExtractOptions("archive.zip entry.txt out.txt extra.txt"));
    }

    private static void assertError(String expected, Runnable action) {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(expected, error.getMessage());
    }
}
