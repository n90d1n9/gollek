package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookMagicArgsTest {

    @Test
    void parseLeadingOptionsSeparatesFlagsValuesAndPositionals() {
        NotebookMagicArgs.ParsedOptions args = NotebookMagicArgs.parseLeadingOptions(
                "--tsv -n 12 --seed 99 data.tsv",
                "usage",
                "Unknown option: ",
                NotebookMagicArgs.flag("tsv", "--tsv"),
                NotebookMagicArgs.value("rows", "-n", "--rows"),
                NotebookMagicArgs.value("seed", "--seed")
        );

        assertTrue(args.has("tsv"));
        assertEquals("12", args.value("rows"));
        assertEquals("99", args.value("seed"));
        assertEquals(List.of("data.tsv"), args.positionals());
    }

    @Test
    void parseLeadingOptionsTracksLastFlagAmongMutuallyExclusiveOptions() {
        NotebookMagicArgs.ParsedOptions args = NotebookMagicArgs.parseLeadingOptions(
                "--desc --asc --desc data.csv score",
                "usage",
                "Unknown option: ",
                NotebookMagicArgs.flag("desc", "--desc"),
                NotebookMagicArgs.flag("asc", "--asc")
        );

        assertEquals("desc", args.lastPresent("asc", "desc"));
    }

    @Test
    void parseLeadingOptionsRejectsUnknownLongOptions() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                NotebookMagicArgs.parseLeadingOptions(
                        "--wat data.csv",
                        "usage",
                        "Unknown option: ",
                        NotebookMagicArgs.flag("tsv", "--tsv")
                )
        );

        assertEquals("Unknown option: --wat", error.getMessage());
    }

    @Test
    void parseLeadingOptionsAllowsSingleDashPositionalsThatAreNotKnownOptions() {
        NotebookMagicArgs.ParsedOptions args = NotebookMagicArgs.parseLeadingOptions(
                "-dataset.csv",
                "usage",
                "Unknown option: ",
                NotebookMagicArgs.value("rows", "-n")
        );

        assertFalse(args.has("rows"));
        assertEquals(List.of("-dataset.csv"), args.positionals());
    }

    @Test
    void requirePositionalsReportsUsageOnWrongArity() {
        NotebookMagicArgs.ParsedOptions args = NotebookMagicArgs.parseLeadingOptions(
                "data.csv column extra",
                "usage",
                "Unknown option: "
        );

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                args.requirePositionals(2, "Usage: demo")
        );
        assertEquals("Usage: demo", error.getMessage());
    }
}
