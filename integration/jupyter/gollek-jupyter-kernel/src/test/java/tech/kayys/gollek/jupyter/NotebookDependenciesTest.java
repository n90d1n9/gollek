package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NotebookDependenciesTest {

    @Test
    void parseMavenMagicArgsAcceptsPlainCoordinates() {
        NotebookDependencies.MavenMagicArgs args = NotebookDependencies.parseMavenMagicArgs("dev.gollek:kernel:1.0.0");

        assertTrue(args.valid());
        assertFalse(args.allowRemote());
        assertFalse(args.fetch());
        assertFalse(args.explain());
        assertEquals("dev.gollek:kernel:1.0.0", args.coordinates());
        assertEquals("dev.gollek", args.groupId());
        assertEquals("kernel", args.artifactId());
        assertEquals("1.0.0", args.version());
    }

    @Test
    void parseMavenMagicArgsAcceptsFlagsInAnyOrder() {
        NotebookDependencies.MavenMagicArgs args = NotebookDependencies.parseMavenMagicArgs(
                "--fetch --explain --allow-remote dev.gollek:kernel:1.0.0"
        );

        assertTrue(args.valid());
        assertTrue(args.allowRemote());
        assertTrue(args.fetch());
        assertTrue(args.explain());
        assertEquals("dev.gollek:kernel:1.0.0", args.coordinates());
    }

    @Test
    void parseMavenMagicArgsRejectsInvalidCoordinates() {
        NotebookDependencies.MavenMagicArgs args = NotebookDependencies.parseMavenMagicArgs("dev.gollek:kernel");

        assertFalse(args.valid());
        assertEquals("Invalid coordinates. Expected: groupId:artifactId:version", args.error());
    }

    @Test
    void parseMavenMagicArgsRejectsBlankCoordinateParts() {
        NotebookDependencies.MavenMagicArgs args = NotebookDependencies.parseMavenMagicArgs("dev.gollek::1.0.0");

        assertFalse(args.valid());
        assertEquals("Invalid coordinates. Expected: groupId:artifactId:version", args.error());
    }

    @Test
    void parseMavenMagicArgsRejectsUnknownOptionsWithUsage() {
        NotebookDependencies.MavenMagicArgs args = NotebookDependencies.parseMavenMagicArgs(
                "--offline dev.gollek:kernel:1.0.0"
        );

        assertFalse(args.valid());
        assertTrue(args.error().contains("Unknown %maven option: --offline"));
        assertTrue(args.error().contains("Usage: %maven"));
    }
}
