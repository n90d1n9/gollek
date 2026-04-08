package tech.kayys.gollek.runtime.unified;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.annotations.QuarkusMain;
import picocli.CommandLine;
import tech.kayys.gollek.cli.GollekCommand;

/**
 * Main entry point for the Gollek Unified Runtime.
 * 
 * <p>This class provides a dual-mode execution strategy:
 * <ul>
 *   <li><b>CLI Mode:</b> Triggered when arguments are provided. It delegates execution to the Picocli-based 
 *   {@link GollekCommand}.</li>
 *   <li><b>Server Mode:</b> Triggered when no arguments are provided. It starts the Quarkus application 
 *   container, enabling the REST API and Web UI.</li>
 * </ul>
 * 
 * <p>Usage:
 * <pre>
 *   # Start Server Mode
 *   java -jar gollek-runtime-unified.jar
 * 
 *   # Run CLI Command
 *   java -jar gollek-runtime-unified.jar chat --model llama3
 * </pre>
 * 
 * @author Bhangun
 * @since 1.0.0
 */
@QuarkusMain
public class UnifiedRuntimeMain {

    /**
     * Bootstraps the unified runtime.
     *
     * @param args Command line arguments. If empty, starts in server mode.
     */
    public static void main(String[] args) {
        if (isCliMode(args)) {
            // Run CLI command and exit with the appropriate status code
            System.exit(runCli(args));
        } else {
            // Start the Quarkus server (REST API + Web UI)
            Quarkus.run(args);
        }
    }

    /**
     * Determines if the runtime should start in CLI mode based on provided arguments.
     *
     * @param args The launch arguments.
     * @return {@code true} if arguments suggest a CLI command, {@code false} for server mode.
     */
    private static boolean isCliMode(String[] args) {
        return args.length > 0 && !args[0].startsWith("-");
    }

    /**
     * Initializes and executes the Gollek CLI.
     *
     * @param args CLI arguments.
     * @return Exit code returned by the command execution.
     */
    private static int runCli(String[] args) {
        return new CommandLine(new GollekCommand()).execute(args);
    }
}