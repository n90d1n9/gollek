package tech.kayys.gollek.cli;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.core.graph.ExecutionContext;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.backend.metal.MetalBackend;
import java.util.concurrent.Callable;

@Command(name = "gollek", mixinStandardHelpOptions = true, version = "gollek 2.0-newarch",
         description = "Gollek New Architecture CLI")
public class NewGollekCLI implements Callable<Integer> {

    @Command(name = "run", description = "Run inference")
    public Integer run(
            @Option(names = {"-m", "--model"}, required = true) String model,
            @Option(names = {"-p", "--prompt"}, required = true) String prompt) {
        
        System.out.println("🚀 Initializing New Gollek Architecture...");
        
        // 1. Initialize Backend
        ComputeBackend backend = new MetalBackend();
        ExecutionContext ctx = new ExecutionContext(backend, false);
        
        System.out.println("📦 Model: " + model);
        System.out.println("💬 Prompt: " + prompt);
        
        // 2. Load Model (Mock for now)
        System.out.println("⚡ Executing via MetalBackend...");
        
        // TODO: Real execution logic
        System.out.println("\n[Output]: This is a response from the new Gollek architecture running on Metal.");
        
        return 0;
    }

    @Override
    public Integer call() {
        return 0;
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new NewGollekCLI()).execute(args);
        System.exit(exitCode);
    }
}
