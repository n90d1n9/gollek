package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookTableOpsTest {

    @Test
    void sampleRowsReturnsOriginalWhenRequestCoversData() {
        List<List<String>> rows = List.of(List.of("Ada"), List.of("Bob"));

        assertSame(rows, NotebookTableOps.sampleRows(rows, 2, 7L));
        assertSame(rows, NotebookTableOps.sampleRows(rows, 3, null));
    }

    @Test
    void sampleRowsUsesSeededDeterministicShuffle() {
        List<List<String>> rows = List.of(
                List.of("Ada"),
                List.of("Bob"),
                List.of("Cleo"),
                List.of("Dia")
        );

        List<List<String>> first = NotebookTableOps.sampleRows(rows, 2, 42L);
        List<List<String>> second = NotebookTableOps.sampleRows(rows, 2, 42L);

        assertEquals(first, second);
        assertEquals(2, first.size());
    }

    @Test
    void isNumericSortColumnRequiresNonBlankNumericValues() {
        assertTrue(NotebookTableOps.isNumericSortColumn(
                List.of(List.of("1"), List.of(""), List.of("2.5")),
                0
        ));
        assertFalse(NotebookTableOps.isNumericSortColumn(
                List.of(List.of("1"), List.of("nope")),
                0
        ));
        assertFalse(NotebookTableOps.isNumericSortColumn(
                List.of(List.of(""), List.of("")),
                0
        ));
    }

    @Test
    void compareSortRowsKeepsBlankKeysLast() {
        NotebookTableOps.SortRow blank = new NotebookTableOps.SortRow(List.of(""), "", null, 0);
        NotebookTableOps.SortRow high = new NotebookTableOps.SortRow(List.of("9"), "9", 9.0, 1);
        NotebookTableOps.SortRow low = new NotebookTableOps.SortRow(List.of("3"), "3", 3.0, 2);

        assertTrue(NotebookTableOps.compareSortRows(blank, high, true, false) > 0);
        assertTrue(NotebookTableOps.compareSortRows(blank, high, true, true) > 0);
        assertTrue(NotebookTableOps.compareSortRows(high, low, true, false) > 0);
        assertTrue(NotebookTableOps.compareSortRows(high, low, true, true) < 0);
    }

    @Test
    void normalizeFilterOperatorMapsAliasesAndValueRequirement() {
        assertEquals("==", NotebookTableOps.normalizeFilterOperator("eq"));
        assertEquals("!=", NotebookTableOps.normalizeFilterOperator("<>"));
        assertEquals(">=", NotebookTableOps.normalizeFilterOperator("gte"));
        assertEquals("!contains", NotebookTableOps.normalizeFilterOperator("not_contains"));
        assertEquals("blank", NotebookTableOps.normalizeFilterOperator("is_blank"));
        assertFalse(NotebookTableOps.filterRequiresValue("blank"));
        assertTrue(NotebookTableOps.filterRequiresValue("contains"));
    }

    @Test
    void matchesFilterSupportsTextNumericAndBlankPredicates() {
        assertTrue(NotebookTableOps.matchesFilter("Ada Lovelace", "contains", "Love"));
        assertTrue(NotebookTableOps.matchesFilter("42", ">", "10"));
        assertTrue(NotebookTableOps.matchesFilter("   ", "blank", null));
        assertFalse(NotebookTableOps.matchesFilter("N/A", ">", "10"));
        assertFalse(NotebookTableOps.matchesFilter("Ada", "ends", "Bob"));
    }

    @Test
    void filterPredicateLabelOmitsNullValues() {
        assertEquals("name blank", NotebookTableOps.filterPredicateLabel("name", "blank", null));
        assertEquals("score > 10", NotebookTableOps.filterPredicateLabel("score", ">", "10"));
    }
}
