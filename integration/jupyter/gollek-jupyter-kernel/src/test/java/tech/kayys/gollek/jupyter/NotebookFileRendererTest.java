package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookFileRendererTest {

    @Test
    void workingDirectoryPreviewsEscapeHtml() {
        NotebookPreview current = NotebookFileRenderer.workingDirectoryPreview("/tmp/<work>");
        NotebookPreview changed = NotebookFileRenderer.workingDirectoryChangedPreview("/tmp/<next>");

        assertTrue(current.plain().contains("/tmp/<work>"));
        assertTrue(current.html().contains("/tmp/&lt;work&gt;"));
        assertTrue(changed.plain().contains("Working directory changed to /tmp/<next>"));
        assertTrue(changed.html().contains("/tmp/&lt;next&gt;"));
    }

    @Test
    void directoryAndTreePreviewsFormatEntriesAndEscapeHtml() {
        NotebookPreview directory = NotebookFileRenderer.directoryPreview(
                "/tmp/<work>",
                List.of("alpha.txt", "<nested>/")
        );
        NotebookPreview tree = NotebookFileRenderer.treePreview(
                "/tmp/<work>",
                List.of("Tree(/tmp/<work>)", "alpha.txt", "  <nested>/")
        );

        assertTrue(directory.plain().contains("Directory(/tmp/<work>)"));
        assertTrue(directory.plain().contains("<nested>/"));
        assertTrue(directory.html().contains("/tmp/&lt;work&gt;"));
        assertTrue(directory.html().contains("&lt;nested&gt;/"));
        assertTrue(tree.plain().contains("Tree(/tmp/<work>)"));
        assertTrue(tree.html().contains("Tree(/tmp/&lt;work&gt;)"));
        assertFalse(tree.html().contains("Tree(/tmp/<work>)"));
    }

    @Test
    void diskUsagePreviewFormatsSummaryAndTruncation() {
        NotebookPreview preview = NotebookFileRenderer.diskUsagePreview(
                "/tmp/data",
                "directory",
                new NotebookFiles.DiskUsageStats(3, 2, 2048, 5, true)
        );

        assertTrue(preview.plain().contains("DU(/tmp/data, type=directory)"));
        assertTrue(preview.plain().contains("files=3 directories=2 bytes=2048 human=2.0 KiB entriesScanned=5 truncated=true"));
        assertTrue(preview.html().contains("Disk Usage"));
        assertTrue(preview.html().contains("human=2.0 KiB"));
    }

    @Test
    void fileAndLinePreviewsEscapeBodies() {
        NotebookPreview file = NotebookFileRenderer.filePreview(
                "/tmp/<notes>.txt",
                new NotebookFiles.TextPreview(32, true, "hello <world>")
        );
        NotebookPreview head = NotebookFileRenderer.linePreview(
                "Head",
                "/tmp/<notes>.txt",
                List.of("one", "<two>"),
                10
        );

        assertTrue(file.plain().contains("File(/tmp/<notes>.txt, bytes=32, truncated=true)"));
        assertTrue(file.html().contains("/tmp/&lt;notes&gt;.txt"));
        assertTrue(file.html().contains("hello &lt;world&gt;"));
        assertTrue(head.plain().contains("Head(/tmp/<notes>.txt, lines=2/10)"));
        assertTrue(head.html().contains("&lt;two&gt;"));
    }

    @Test
    void matchesPreviewHandlesHitsAndMisses() {
        NotebookPreview hits = NotebookFileRenderer.matchesPreview(
                "Grep",
                "Grep",
                "<needle>",
                List.of("a.txt:1: has <needle>")
        );
        NotebookPreview miss = NotebookFileRenderer.matchesPreview("FindFile", "Find File", "missing", List.of());

        assertTrue(hits.plain().contains("Grep(pattern=<needle>, matches=1)"));
        assertTrue(hits.html().contains("&lt;needle&gt;"));
        assertTrue(hits.html().contains("has &lt;needle&gt;"));
        assertTrue(miss.plain().contains("FindFile(pattern=missing, matches=0)"));
        assertTrue(miss.html().contains("No matches found."));
    }

    @Test
    void wordCountAndShaPreviewsKeepPlainSummaries() {
        NotebookPreview wc = NotebookFileRenderer.wordCountPreview("/tmp/<notes>.txt", 2, 4, 20, 21);
        NotebookPreview sha = NotebookFileRenderer.sha256Preview("/tmp/<notes>.txt", 21, "abc123");

        assertTrue(wc.plain().contains("WC(/tmp/<notes>.txt)"));
        assertTrue(wc.plain().contains("lines=2 words=4 chars=20 bytes=21"));
        assertTrue(wc.html().contains("/tmp/&lt;notes&gt;.txt"));
        assertTrue(sha.plain().contains("SHA256(/tmp/<notes>.txt, bytes=21)"));
        assertTrue(sha.html().contains("abc123"));
    }

    @Test
    void diffPreviewColorsChangesAndEscapesLines() {
        NotebookPreview changed = NotebookFileRenderer.diffPreview(
                "/tmp/<left>.txt",
                "/tmp/<right>.txt",
                1,
                1,
                1,
                List.of("- 1: old <value>", "+ 1: new <value>")
        );
        NotebookPreview same = NotebookFileRenderer.diffPreview(
                "/tmp/left.txt",
                "/tmp/right.txt",
                0,
                0,
                0,
                List.of()
        );

        assertTrue(changed.plain().contains("Diff(left=/tmp/<left>.txt, right=/tmp/<right>.txt, changed=1, added=1, removed=1)"));
        assertTrue(changed.html().contains("/tmp/&lt;left&gt;.txt"));
        assertTrue(changed.html().contains("color:#b31d28"));
        assertTrue(changed.html().contains("old &lt;value&gt;"));
        assertTrue(changed.html().contains("color:#22863a"));
        assertTrue(same.plain().contains("No line differences found."));
        assertTrue(same.html().contains("No line differences found."));
    }
}
