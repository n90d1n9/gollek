package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NotebookTableMagicOptionsTest {

    @Test
    void tableOptionsParseProfileRowsAndClampPreviewSize() {
        NotebookTableMagicOptions.TableOptions options =
                NotebookTableMagicOptions.parseTableOptions("--profile --rows 999 data.csv", "CSV");

        assertEquals("data.csv", options.path());
        assertEquals(200, options.previewRows());
        assertTrue(options.profile());
    }

    @Test
    void sampleOptionsParseTsvRowsAndSeed() {
        NotebookTableMagicOptions.SampleOptions options =
                NotebookTableMagicOptions.parseSampleOptions("--tsv -n 25 --seed 42 data.tsv");

        assertEquals("data.tsv", options.path());
        assertEquals(25, options.rows());
        assertEquals(42L, options.seed());
        assertTrue(options.tsv());
    }

    @Test
    void sortOptionsRespectLastAscDescFlag() {
        NotebookTableMagicOptions.SortOptions options =
                NotebookTableMagicOptions.parseSortOptions("--desc --asc --rows 300 data.csv score");

        assertEquals("data.csv", options.path());
        assertEquals("score", options.column());
        assertEquals(200, options.rows());
        assertFalse(options.descending());
    }

    @Test
    void filterOptionsNormalizeOperatorsAndBlankDropsValue() {
        NotebookTableMagicOptions.FilterOptions comparison =
                NotebookTableMagicOptions.parseFilterOptions("--tsv -n 10 data.tsv score gte 90");
        NotebookTableMagicOptions.FilterOptions blank =
                NotebookTableMagicOptions.parseFilterOptions("data.csv name is_blank ignored");

        assertEquals(">=", comparison.operator());
        assertEquals("90", comparison.value());
        assertEquals(10, comparison.rows());
        assertTrue(comparison.tsv());
        assertEquals("blank", blank.operator());
        assertNull(blank.value());
    }

    @Test
    void valueCountsAndHistogramOptionsClampLimits() {
        NotebookTableMagicOptions.ValueCountsOptions counts =
                NotebookTableMagicOptions.parseValueCountsOptions("--top 999 data.csv label");
        NotebookTableMagicOptions.HistogramOptions histogram =
                NotebookTableMagicOptions.parseHistogramOptions("--bins 999 data.csv score");

        assertEquals(100, counts.top());
        assertEquals("label", counts.column());
        assertEquals(100, histogram.bins());
        assertEquals("score", histogram.column());
    }

    @Test
    void groupByOptionsDefaultValueColumnToMeanAndValidateAggregates() {
        NotebookTableMagicOptions.GroupByOptions count =
                NotebookTableMagicOptions.parseGroupByOptions("data.csv department count");
        NotebookTableMagicOptions.GroupByOptions mean =
                NotebookTableMagicOptions.parseGroupByOptions("data.csv department salary");
        NotebookTableMagicOptions.GroupByOptions sum =
                NotebookTableMagicOptions.parseGroupByOptions("--tsv data.tsv department salary sum");

        assertEquals("count", count.aggregate());
        assertNull(count.valueColumn());
        assertEquals("salary", mean.valueColumn());
        assertEquals("mean", mean.aggregate());
        assertEquals("sum", sum.aggregate());
        assertTrue(sum.tsv());
        assertThrows(IllegalArgumentException.class,
                () -> NotebookTableMagicOptions.parseGroupByOptions("data.csv department sum"));
        assertFalse(NotebookTableMagicOptions.isGroupAggregate("median"));
    }

    @Test
    void invalidPositiveIntegerOptionsRaiseClearErrors() {
        assertThrows(IllegalArgumentException.class,
                () -> NotebookTableMagicOptions.parseSampleOptions("-n 0 data.csv"));
        assertThrows(IllegalArgumentException.class,
                () -> NotebookTableMagicOptions.parseFilterOptions("data.csv score >"));
        assertThrows(IllegalArgumentException.class,
                () -> NotebookTableMagicOptions.parseHistogramOptions("--bins nope data.csv score"));
    }
}
