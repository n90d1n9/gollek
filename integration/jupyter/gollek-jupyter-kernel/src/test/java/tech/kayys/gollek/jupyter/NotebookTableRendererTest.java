package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookTableRendererTest {

    @Test
    void plainRowsCanRenderNumberedPreviewRows() {
        String plain = NotebookTableRenderer.plainRows(
                List.of("name", "score"),
                List.of(List.of("Ada", "99"), List.of("Bob", "88")),
                true
        );

        assertTrue(plain.contains("# | name | score"));
        assertTrue(plain.contains("1 | Ada | 99"));
        assertTrue(plain.contains("2 | Bob | 88"));
    }

    @Test
    void htmlRowsEscapesCellsAndCanOmitRowNumbers() {
        String html = NotebookTableRenderer.htmlRows(
                List.of("name"),
                List.of(List.of("<Ada>")),
                false
        );

        assertTrue(html.contains("<table"));
        assertTrue(html.contains("&lt;Ada&gt;"));
        assertFalse(html.contains(">#</th>"));
    }

    @Test
    void scrollableHtmlRowsUsesStickyHeaders() {
        String html = NotebookTableRenderer.scrollableHtmlRows(
                List.of("name"),
                List.of(List.of("Ada")),
                true
        );

        assertTrue(html.contains("max-height:420px"));
        assertTrue(html.contains("position:sticky"));
        assertTrue(html.contains(">#</th>"));
    }

    @Test
    void profileRenderersFormatAndEscapeColumnProfiles() {
        List<NotebookStats.ColumnProfile> profiles = List.of(
                new NotebookStats.ColumnProfile("<score>", 2, 1, 2, 1.0, 3.5, 2.25)
        );

        String plain = NotebookTableRenderer.plainColumnProfile(profiles);
        String html = NotebookTableRenderer.htmlColumnProfile(profiles);

        assertEquals("Profile\n"
                + "column | nonEmpty | missing | numeric | min | max | mean\n"
                + "<score> | 2 | 1 | 2 | 1 | 3.5000 | 2.2500", plain);
        assertTrue(html.contains("<b>Column Profile</b>"));
        assertTrue(html.contains("&lt;score&gt;"));
        assertFalse(html.contains("<score>"));
    }

    @Test
    void delimitedTablePreviewIncludesTruncationAndProfileSections() {
        List<NotebookStats.ColumnProfile> profiles = List.of(
                new NotebookStats.ColumnProfile("score", 2, 1, 2, 1.0, 2.0, 1.5)
        );

        NotebookPreview preview = NotebookTableRenderer.delimitedTablePreview(
                "CSV",
                "/tmp/<scores>.csv",
                List.of("name", "score"),
                List.of(List.of("Ada", "1"), List.of("Bob", "2"), List.of("Cleo", "")),
                List.of(List.of("Ada", "1"), List.of("Bob", "2")),
                true,
                profiles
        );

        assertTrue(preview.plain().contains("CSV(/tmp/<scores>.csv, rows=3, columns=2, previewRows=2, truncated=true, profile=true)"));
        assertTrue(preview.plain().contains("... 1 more rows"));
        assertTrue(preview.plain().contains("Profile\ncolumn | nonEmpty | missing | numeric | min | max | mean"));
        assertTrue(preview.html().contains("<b>CSV</b>"));
        assertTrue(preview.html().contains("/tmp/&lt;scores&gt;.csv"));
        assertTrue(preview.html().contains("Column Profile"));
    }

    @Test
    void samplePreviewIncludesSeedAndEscapesTarget() {
        NotebookPreview preview = NotebookTableRenderer.samplePreview(
                "/tmp/<sample>.csv",
                "CSV",
                List.of("name"),
                5,
                List.of(List.of("Ada"), List.of("Bob")),
                "42"
        );

        assertTrue(preview.plain().contains("Sample(/tmp/<sample>.csv, format=CSV, rows=5, sample=2, columns=1, seed=42)"));
        assertTrue(preview.plain().contains("name\nAda\nBob"));
        assertTrue(preview.html().contains("/tmp/&lt;sample&gt;.csv"));
        assertTrue(preview.html().contains("seed=42"));
    }

    @Test
    void sortPreviewIncludesModeAndTruncation() {
        NotebookPreview preview = NotebookTableRenderer.sortPreview(
                "/tmp/scores.csv",
                "CSV",
                "<score>",
                "desc",
                "numeric",
                List.of("name", "score"),
                3,
                List.of(List.of("Ada", "99"), List.of("Bob", "88"))
        );

        assertTrue(preview.plain().contains("Sort(/tmp/scores.csv, format=CSV, column=<score>, order=desc, mode=numeric"));
        assertTrue(preview.plain().contains("... 1 more rows"));
        assertTrue(preview.html().contains("column=&lt;score&gt;"));
        assertTrue(preview.html().contains("1 more rows not shown"));
    }

    @Test
    void filterPreviewIncludesPredicateValueAndTruncation() {
        NotebookPreview preview = NotebookTableRenderer.filterPreview(
                "/tmp/scores.csv",
                "CSV",
                "score",
                ">=",
                "90",
                "score >= 90",
                List.of("name", "score"),
                5,
                3,
                List.of(List.of("Ada", "99"), List.of("Cleo", "91"))
        );

        assertTrue(preview.plain().contains("Filter(/tmp/scores.csv, format=CSV, column=score, op=>=, value=90"));
        assertTrue(preview.plain().contains("... 1 more matched rows"));
        assertTrue(preview.html().contains("predicate=score &gt;= 90"));
        assertTrue(preview.html().contains("1 more matched rows not shown"));
    }

}
