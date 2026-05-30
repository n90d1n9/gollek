package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookArchiveRendererTest {

    @Test
    void archiveListingPreviewFormatsEntriesAndEscapesTarget() {
        NotebookPreview preview = NotebookArchiveRenderer.archiveListingPreview(
                "Zip",
                "/tmp/<archive>.zip",
                new NotebookArchives.ArchiveListing(List.of("nested/report.txt (12 bytes)"), 12)
        );

        assertTrue(preview.plain().contains("Zip(/tmp/<archive>.zip, entries=1, bytes=12)"));
        assertTrue(preview.plain().contains("nested/report.txt (12 bytes)"));
        assertTrue(preview.html().contains("/tmp/&lt;archive&gt;.zip"));
        assertTrue(preview.html().contains("entries=1 bytes=12"));
        assertTrue(preview.html().contains("nested/report.txt (12 bytes)"));
    }

    @Test
    void archiveListingPreviewHandlesEmptyArchives() {
        NotebookPreview preview = NotebookArchiveRenderer.archiveListingPreview(
                "Tar",
                "/tmp/empty.tar",
                new NotebookArchives.ArchiveListing(List.of(), 0)
        );

        assertTrue(preview.plain().contains("Tar(/tmp/empty.tar, entries=0)"));
        assertTrue(preview.html().contains("No entries found."));
    }

    @Test
    void archiveEntryPreviewFormatsTruncationAndEscapesHtml() {
        NotebookPreview preview = NotebookArchiveRenderer.archiveEntryPreview(
                "ZipEntry",
                "Zip Entry",
                "/tmp/<archive>.zip",
                "nested/<report>.txt",
                100,
                "hello <world>".getBytes(StandardCharsets.UTF_8),
                true
        );

        assertTrue(preview.plain().contains("ZipEntry(/tmp/<archive>.zip!nested/<report>.txt, bytes=100, truncated=true)"));
        assertTrue(preview.plain().contains("hello <world>"));
        assertTrue(preview.html().contains("/tmp/&lt;archive&gt;.zip!nested/&lt;report&gt;.txt"));
        assertTrue(preview.html().contains("hello &lt;world&gt;"));
        assertTrue(preview.html().contains("truncated=true"));
    }

    @Test
    void extractPreviewFormatsDryRunAndWrittenStates() {
        NotebookPreview dryRun = NotebookArchiveRenderer.extractPreview(
                true,
                "/tmp/archive.zip",
                "nested/report.txt",
                "/tmp/out/report.txt",
                12
        );
        NotebookPreview written = NotebookArchiveRenderer.extractPreview(
                false,
                "/tmp/archive.zip",
                "nested/report.txt",
                "/tmp/out/report.txt",
                12
        );

        assertTrue(dryRun.plain().contains("Extract(dryRun=true, archive=/tmp/archive.zip, entry=nested/report.txt, output=/tmp/out/report.txt, bytes=12)"));
        assertTrue(dryRun.html().contains("<b>Extract</b> dry-run"));
        assertTrue(written.plain().contains("Extract(dryRun=false"));
        assertTrue(written.plain().contains("Wrote /tmp/out/report.txt"));
        assertTrue(written.html().contains("<b>Extract</b> written"));
    }
}
