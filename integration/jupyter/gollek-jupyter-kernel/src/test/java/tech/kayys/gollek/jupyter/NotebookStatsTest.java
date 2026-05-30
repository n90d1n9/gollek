package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookStatsTest {

    @Test
    void profileColumnsSummarizesMissingAndNumericValues() {
        List<NotebookStats.ColumnProfile> profiles = NotebookStats.profileColumns(
                List.of("name", "score"),
                List.of(
                        List.of("Ada", "1"),
                        List.of("Bob", ""),
                        List.of("", "3.5")
                )
        );

        NotebookStats.ColumnProfile name = profiles.get(0);
        assertEquals("name", name.name());
        assertEquals(2, name.nonEmpty());
        assertEquals(1, name.missing());
        assertEquals(0, name.numeric());
        assertNull(name.min());
        assertNull(name.max());
        assertNull(name.mean());

        NotebookStats.ColumnProfile score = profiles.get(1);
        assertEquals("score", score.name());
        assertEquals(2, score.nonEmpty());
        assertEquals(1, score.missing());
        assertEquals(2, score.numeric());
        assertEquals(1.0, score.min());
        assertEquals(3.5, score.max());
        assertEquals(2.25, score.mean());
    }

    @Test
    void profileColumnsTreatsNonFiniteNumbersAsNonNumericValues() {
        List<NotebookStats.ColumnProfile> profiles = NotebookStats.profileColumns(
                List.of("value"),
                List.of(List.of("NaN"), List.of("Infinity"), List.of("-Infinity"))
        );

        NotebookStats.ColumnProfile profile = profiles.getFirst();
        assertEquals(3, profile.nonEmpty());
        assertEquals(0, profile.missing());
        assertEquals(0, profile.numeric());
        assertNull(profile.min());
        assertNull(profile.max());
        assertNull(profile.mean());
    }
}
