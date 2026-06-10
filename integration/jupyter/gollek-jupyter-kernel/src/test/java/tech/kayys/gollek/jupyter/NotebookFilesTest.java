package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies reusable filesystem helpers used by notebook file magics.
 */
class NotebookFilesTest {

    @TempDir
    Path tempDir;

    @Test
    void listDirectoryEntriesSortsAndMarksDirectories() throws Exception {
        Files.createDirectories(tempDir.resolve("Beta"));
        Files.writeString(tempDir.resolve("alpha.txt"), "ok", StandardCharsets.UTF_8);
        Files.writeString(tempDir.resolve("zeta.txt"), "ok", StandardCharsets.UTF_8);

        List<String> entries = NotebookFiles.listDirectoryEntries(tempDir, 50);

        assertEquals(List.of("alpha.txt", "Beta/", "zeta.txt"), entries);
    }

    @Test
    void treeLinesIncludeRootAndIndentedChildren() throws Exception {
        Files.createDirectories(tempDir.resolve("nested"));
        Files.writeString(tempDir.resolve("root.txt"), "ok", StandardCharsets.UTF_8);
        Files.writeString(tempDir.resolve("nested/child.txt"), "ok", StandardCharsets.UTF_8);

        List<String> lines = NotebookFiles.treeLines(tempDir, 60, 3);

        assertEquals("Tree(" + tempDir + ")", lines.get(0));
        assertEquals(List.of("nested/", "  child.txt", "root.txt"), lines.subList(1, lines.size()));
    }

    @Test
    void readHeadReturnsFirstLinesAndTotalLineCount() throws Exception {
        Path file = tempDir.resolve("lines.txt");
        Files.writeString(file, "line1\nline2\nline3\nline4\n", StandardCharsets.UTF_8);

        NotebookFiles.LineWindow window = NotebookFiles.readHead(file, 2);

        assertEquals(List.of("line1", "line2"), window.lines());
        assertEquals(4, window.totalLines());
    }

    @Test
    void readTailReturnsLastLinesAndCapsAtFileLength() throws Exception {
        Path file = tempDir.resolve("lines.txt");
        Files.writeString(file, "line1\nline2\nline3\n", StandardCharsets.UTF_8);

        NotebookFiles.LineWindow tail = NotebookFiles.readTail(file, 2);
        NotebookFiles.LineWindow oversized = NotebookFiles.readTail(file, 10);

        assertEquals(List.of("line2", "line3"), tail.lines());
        assertEquals(3, tail.totalLines());
        assertEquals(List.of("line1", "line2", "line3"), oversized.lines());
        assertEquals(3, oversized.totalLines());
    }

    @Test
    void computeFileStatsCountsLinesWordsCharsAndBytes() throws Exception {
        Path file = tempDir.resolve("notes.txt");
        Files.writeString(file, "one two\nthree\n", StandardCharsets.UTF_8);

        NotebookFiles.FileStats stats = NotebookFiles.computeFileStats(file);

        assertEquals(2, stats.lines());
        assertEquals(3, stats.words());
        assertEquals(14, stats.chars());
        assertEquals(14, stats.bytes());
    }

    @Test
    void computeSha256ReportsBytesAndHexDigest() throws Exception {
        Path file = tempDir.resolve("payload.txt");
        Files.writeString(file, "abc", StandardCharsets.UTF_8);

        NotebookFiles.Sha256Digest digest = NotebookFiles.computeSha256(file);

        assertEquals(3, digest.bytes());
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", digest.hash());
    }

    @Test
    void collectTextMatchesSearchesSingleFileAndDirectoryTrees() throws Exception {
        Path file = tempDir.resolve("notes.txt");
        Path nested = tempDir.resolve("nested");
        Files.createDirectories(nested);
        Files.writeString(file, "hello\nneedle here\n", StandardCharsets.UTF_8);
        Files.writeString(nested.resolve("more.txt"), "other\nneedle there\n", StandardCharsets.UTF_8);

        assertEquals(
                List.of("notes.txt:2: needle here"),
                NotebookFiles.collectTextMatches(file, "needle", 80, 50, 3)
        );
        assertEquals(
                List.of("nested/more.txt:2: needle there", "notes.txt:2: needle here"),
                NotebookFiles.collectTextMatches(tempDir, "needle", 80, 50, 3)
        );
    }

    @Test
    void collectFileNameMatchesSearchesSingleFileAndDirectoryTrees() throws Exception {
        Path matchingFile = tempDir.resolve("alpha-notes.txt");
        Path nested = tempDir.resolve("nested-beta");
        Files.createDirectories(nested);
        Files.writeString(matchingFile, "ok", StandardCharsets.UTF_8);
        Files.writeString(nested.resolve("gamma.txt"), "ok", StandardCharsets.UTF_8);

        assertEquals(
                List.of("alpha-notes.txt"),
                NotebookFiles.collectFileNameMatches(matchingFile, "alpha", 120, 3)
        );
        assertEquals(
                List.of("nested-beta/", "nested-beta/gamma.txt"),
                NotebookFiles.collectFileNameMatches(tempDir, "beta", 120, 3)
        );
    }

    @Test
    void computeLineDiffReportsChangedAddedAndRemovedLines() throws Exception {
        Path left = tempDir.resolve("left.txt");
        Path right = tempDir.resolve("right.txt");
        Files.writeString(left, "alpha\nbeta\nsame\n", StandardCharsets.UTF_8);
        Files.writeString(right, "alpha\ngamma\nsame\nextra\n", StandardCharsets.UTF_8);

        NotebookFiles.DiffResult diff = NotebookFiles.computeLineDiff(left, right);

        assertEquals(2, diff.changed());
        assertEquals(2, diff.added());
        assertEquals(1, diff.removed());
        assertEquals(List.of("- 2: beta", "+ 2: gamma", "+ 4: extra"), diff.lines());
    }

    @Test
    void computeLineDiffReportsNoChangesForEqualFiles() throws Exception {
        Path left = tempDir.resolve("left.txt");
        Path right = tempDir.resolve("right.txt");
        Files.writeString(left, "alpha\nbeta\n", StandardCharsets.UTF_8);
        Files.writeString(right, "alpha\nbeta\n", StandardCharsets.UTF_8);

        NotebookFiles.DiffResult diff = NotebookFiles.computeLineDiff(left, right);

        assertEquals(0, diff.changed());
        assertEquals(0, diff.added());
        assertEquals(0, diff.removed());
        assertEquals(List.of(), diff.lines());
    }
}
