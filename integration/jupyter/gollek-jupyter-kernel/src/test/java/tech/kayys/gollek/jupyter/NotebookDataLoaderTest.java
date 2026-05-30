package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NotebookDataLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void readDelimitedTableParsesCsvRowsAndQuotedCells() throws Exception {
        Path csv = tempDir.resolve("metrics.csv");
        Files.writeString(csv, "name,score\n\"Ada, Lovelace\",99\nBob,88\n");

        NotebookDataLoader.DelimitedTable table = NotebookDataLoader.readDelimitedTable(csv, false);

        assertEquals("CSV", table.label());
        assertEquals(",", table.delimiter());
        assertEquals(List.of("name", "score"), table.header());
        assertEquals(2, table.rowCount());
        assertEquals("Ada, Lovelace", table.rows().getFirst().getFirst());
        assertEquals("99", table.rows().getFirst().get(1));
    }

    @Test
    void readDelimitedTableParsesTsvRowsAndPreservesEmptyCells() throws Exception {
        Path tsv = tempDir.resolve("metrics.tsv");
        Files.writeString(tsv, "name\tscore\tnote\nAda\t\tok\n");

        NotebookDataLoader.DelimitedTable table = NotebookDataLoader.readDelimitedTable(tsv, true);

        assertEquals("TSV", table.label());
        assertEquals("\t", table.delimiter());
        assertEquals(List.of("name", "score", "note"), table.header());
        assertEquals(1, table.rowCount());
        assertEquals("", table.rows().getFirst().get(1));
        assertEquals("ok", table.rows().getFirst().get(2));
    }

    @Test
    void regularFileErrorReportsMissingAndDirectoryTargets() {
        Path missing = tempDir.resolve("missing.csv");

        assertEquals("File not found: " + missing, NotebookDataLoader.regularFileError(missing));
        assertEquals("Not a file: " + tempDir, NotebookDataLoader.regularFileError(tempDir));
    }
}
