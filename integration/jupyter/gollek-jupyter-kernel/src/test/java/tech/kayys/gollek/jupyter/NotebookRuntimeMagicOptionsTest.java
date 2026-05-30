package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class NotebookRuntimeMagicOptionsTest {

    @Test
    void parseTimeitOptionsUsesDefaultRunCount() {
        NotebookRuntimeMagicOptions.TimeitOptions options =
                NotebookRuntimeMagicOptions.parseTimeitOptions("1 + 2");

        assertEquals(5, options.runs());
        assertEquals("1 + 2", options.expression());
    }

    @Test
    void parseTimeitOptionsReadsExplicitRunCount() {
        NotebookRuntimeMagicOptions.TimeitOptions options =
                NotebookRuntimeMagicOptions.parseTimeitOptions("-n 3 gollekLossCurveDemo()");

        assertEquals(3, options.runs());
        assertEquals("gollekLossCurveDemo()", options.expression());
    }

    @Test
    void parseTimeitOptionsRejectsBlankInputWithUsage() {
        assertError("Usage: %timeit [-n N] <java-expression-or-cell>",
                () -> NotebookRuntimeMagicOptions.parseTimeitOptions(""));
        assertError("Usage: %timeit [-n N] <java-expression-or-cell>",
                () -> NotebookRuntimeMagicOptions.parseTimeitOptions(null));
    }

    @Test
    void parseTimeitOptionsPreservesValidationMessages() {
        assertError("Usage: %timeit [-n N] <java-expression-or-cell>",
                () -> NotebookRuntimeMagicOptions.parseTimeitOptions("-n 3"));
        assertError("Invalid run count for %timeit: many",
                () -> NotebookRuntimeMagicOptions.parseTimeitOptions("-n many 1 + 2"));
        assertError("Run count for %timeit must be > 0",
                () -> NotebookRuntimeMagicOptions.parseTimeitOptions("-n 0 1 + 2"));
    }

    private static void assertError(String expected, Runnable action) {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(expected, error.getMessage());
    }
}
