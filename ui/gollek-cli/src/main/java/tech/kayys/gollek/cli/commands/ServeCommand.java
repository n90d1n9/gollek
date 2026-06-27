package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import io.quarkus.arc.Unremovable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Convenience command to serve Gollek API server (alias for "gollek server start")
 */
@Dependent
@Unremovable
@Command(name = "serve",
        description = "Start the Gollek API server (shorthand for 'gollek server start')")
public class ServeCommand implements Runnable {

    @Option(names = {"--debug"}, description = "Run server in foreground with TTY output (not backgrounded)")
    boolean debug = false;

    @Option(names = {"--port"}, description = "Port to run the server on (default: 9131)")
    int port = 9131;

    @Override
    public void run() {
        try {
            ServerCommand.ServeSubcommand subcommand = new ServerCommand.ServeSubcommand();
            subcommand.debug = this.debug;
            subcommand.port = this.port;
            subcommand.run();
        } catch (Exception e) {
            System.err.println("Error starting server: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
