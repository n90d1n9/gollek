package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookSummaryRendererTest {

    @Test
    void valueCountsPreviewIncludesPercentagesOtherRowsAndChart() {
        NotebookPreview preview = NotebookSummaryRenderer.valueCountsPreview(
                "/tmp/<labels>.csv",
                "CSV",
                "<label>",
                10,
                3,
                List.of(
                        new NotebookStats.ValueCount("<A>", 6),
                        new NotebookStats.ValueCount("B", 3)
                ),
                1,
                "<svg id='value-counts'></svg>"
        );

        assertTrue(preview.plain().contains("ValueCounts(/tmp/<labels>.csv, format=CSV, column=<label>, rows=10, unique=3, top=2, other=1)"));
        assertTrue(preview.plain().contains("<A> | 6 | 60.00%"));
        assertTrue(preview.plain().contains("(other) | 1 | 10.00%"));
        assertTrue(preview.html().contains("/tmp/&lt;labels&gt;.csv"));
        assertTrue(preview.html().contains("column=&lt;label&gt;"));
        assertTrue(preview.html().contains("&lt;A&gt;"));
        assertTrue(preview.html().contains("<svg id='value-counts'></svg>"));
    }

    @Test
    void groupByPreviewIncludesAggregateRowsHiddenGroupsAndChart() {
        NotebookPreview preview = NotebookSummaryRenderer.groupByPreview(
                "/tmp/groups.csv",
                "CSV",
                "<dept>",
                "salary",
                "mean",
                10,
                3,
                1,
                2,
                List.of(
                        new NotebookStats.GroupResult("<eng>", 5, 4, 400.0, 100.0, 80.0, 120.0),
                        new NotebookStats.GroupResult("ops", 3, 2, 120.0, 60.0, 50.0, 70.0)
                ),
                "<svg id='groupby'></svg>"
        );

        assertTrue(preview.plain().contains("GroupBy(/tmp/groups.csv, format=CSV, group=<dept>, value=salary, agg=mean, rows=10, groups=3, shown=2, hidden=1, skipped=2)"));
        assertTrue(preview.plain().contains("group | rows | numeric | mean"));
        assertTrue(preview.plain().contains("<eng> | 5 | 4 | 100"));
        assertTrue(preview.plain().contains("... 1 more groups"));
        assertTrue(preview.html().contains("group=&lt;dept&gt;"));
        assertTrue(preview.html().contains("&lt;eng&gt;"));
        assertTrue(preview.html().contains("<svg id='groupby'></svg>"));
        assertTrue(preview.html().contains("1 more groups not shown"));
    }

    @Test
    void schemaPreviewIncludesTypesExamplesAndEscaping() {
        NotebookPreview preview = NotebookSummaryRenderer.schemaPreview(
                "/tmp/<schema>.csv",
                "CSV",
                3,
                List.of(new NotebookStats.SchemaColumn("<name>", "text", 2, 1, List.of("<Ada>", "Bob")))
        );

        assertTrue(preview.plain().contains("Schema(/tmp/<schema>.csv, format=CSV, rows=3, columns=1)"));
        assertTrue(preview.plain().contains("<name> | text | 2 | 1 | 33.33% | <Ada>, Bob"));
        assertTrue(preview.html().contains("/tmp/&lt;schema&gt;.csv"));
        assertTrue(preview.html().contains("&lt;name&gt;"));
        assertTrue(preview.html().contains("&lt;Ada&gt;, Bob"));
    }

    @Test
    void missingPreviewIncludesPercentagesChartAndEscaping() {
        NotebookPreview preview = NotebookSummaryRenderer.missingPreview(
                "/tmp/missing.csv",
                "CSV",
                5,
                List.of(new NotebookStats.MissingColumn("<score>", 2, 3)),
                1,
                "<svg id='missing'></svg>"
        );

        assertTrue(preview.plain().contains("Missing(/tmp/missing.csv, format=CSV, rows=5, columns=1, columnsWithMissing=1)"));
        assertTrue(preview.plain().contains("<score> | 2 | 3 | 40.00%"));
        assertTrue(preview.html().contains("&lt;score&gt;"));
        assertTrue(preview.html().contains("<svg id='missing'></svg>"));
        assertTrue(preview.html().contains("40.00%"));
    }
}
