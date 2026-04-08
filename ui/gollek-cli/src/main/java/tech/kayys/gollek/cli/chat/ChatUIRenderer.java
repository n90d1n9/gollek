package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;

/**
 * Handles terminal output rendering and colors.
 * Respects the NO_COLOR env var (https://no-color.org) and --no-color flag.
 */
@Dependent
public class ChatUIRenderer {

    // Populated at init time; empty strings when color is disabled.
    public static String RESET  = "\u001B[0m";
    public static String BOLD   = "\u001B[1m";
    public static String YELLOW = "\u001B[33m";
    public static String CYAN   = "\u001B[36m";
    public static String GREEN  = "\u001B[32m";
    public static String RED    = "\u001B[31m";
    public static String DIM    = "\u001B[2m";

    static {
        if (System.getenv("NO_COLOR") != null) disableColor();
    }

    public static void disableColor() {
        RESET = BOLD = YELLOW = CYAN = GREEN = RED = DIM = "";
    }

    private boolean jsonMode = false;
    
    public void setJsonMode(boolean jsonMode) {
        this.jsonMode = jsonMode;
    }

    public void printBanner() {
        if (jsonMode) return;
        System.out.println(BOLD + YELLOW + "  _____       _  _      _    " + RESET);
        System.out.println(BOLD + YELLOW + " / ____|     | || |    | |   " + RESET);
        System.out.println(BOLD + YELLOW + "| |  __  ___ | || | ___| | __" + RESET);
        System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| || |/ _ \\ |/ /" + RESET);
        System.out.println(BOLD + YELLOW + "| |__| | (_) | || |  __/   < " + RESET);
        System.out.println(BOLD + YELLOW + " \\_____|\\___/|_||_|\\___|_|\\_\\" + RESET);
        System.out.println();
    }

    public void printModelInfo(String modelId, String providerId, String outputFile) {
        if (jsonMode) return;
        System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", modelId);
        System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n", providerId != null ? providerId : "auto");
        if (outputFile != null) {
            System.out.printf(BOLD + "Output: " + RESET + YELLOW + "%s" + RESET + "%n", outputFile);
        }
        System.out.println(DIM + "Commands: 'exit' to quit, '/reset' to clear history." + RESET);
        System.out.println(DIM + "-".repeat(50) + RESET);
    }

    public void printAssistantPrefix(boolean quiet, boolean streaming) {
        if (jsonMode) return;
        if (!quiet) {
            if (!streaming) System.out.print("\n");
            System.out.print(BOLD + GREEN + "Assistant: " + RESET);
            System.out.flush();
        }
    }

    public void printStats(int tokens, double duration, double tps, boolean quiet) {
        if (jsonMode) return;
        if (!quiet) {
            System.out.printf(DIM + "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET + "%n",
                    tokens, duration, tps);
        }
    }

    public String getPrompt(boolean quiet) {
        return quiet ? "\n>>> " : "\n" + BOLD + CYAN + ">>> " + RESET;
    }

    public String getSecondaryPrompt(boolean quiet) {
        return quiet ? "... " : DIM + "... " + RESET;
    }

    public void printGoodbye(boolean quiet) {
        if (!quiet) System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
    }

    public void printError(String message, boolean quiet) {
        System.err.println("\n" + RED + "Error: " + RESET + message);
    }

    public void printWarning(String message, boolean quiet) {
        if (!quiet) System.out.println(YELLOW + "Warning: " + RESET + message);
    }
    
    public void printInfo(String message, boolean quiet) {
        if (!quiet) System.out.println(DIM + message + RESET);
    }

    public void printSuccess(String message, boolean quiet) {
        if (!quiet) System.out.println(GREEN + message + RESET);
    }

    public void printHardwareInfo(String hardware, boolean quiet) {
        if (!quiet) {
            System.out.printf(DIM + "[Running on %s]" + RESET + "%n", hardware);
        }
    }
}
