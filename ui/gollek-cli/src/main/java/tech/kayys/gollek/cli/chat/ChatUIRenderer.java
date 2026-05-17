package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import java.util.List;
import java.util.Map;

/**
 * Handles terminal output rendering and colors.
 * Respects the NO_COLOR env var (https://no-color.org) and --no-color flag.
 */
@Dependent
public class ChatUIRenderer {

    // Populated at init time; empty strings when color is disabled.
    public static String RESET = "\u001B[0m";
    public static String BOLD = "\u001B[1m";
    public static String YELLOW = "\u001B[33m";
    public static String CYAN = "\u001B[36m";
    public static String GREEN = "\u001B[32m";
    public static String RED = "\u001B[31m";
    public static String DIM = "\u001B[2m";

    static {
        if (System.getenv("NO_COLOR") != null)
            disableColor();
    }

    public static void disableColor() {
        RESET = BOLD = YELLOW = CYAN = GREEN = RED = DIM = "";
    }

    private boolean jsonMode = false;

    public void setJsonMode(boolean jsonMode) {
        this.jsonMode = jsonMode;
    }

    public void printBanner() {
        if (jsonMode)
            return;
        System.out.println(BOLD + YELLOW + "  _____       _  _      _    " + RESET);
        System.out.println(BOLD + YELLOW + " / ____|     | || |    | |   " + RESET);
        System.out.println(BOLD + YELLOW + "| |  __  ___ | || | ___| | __" + RESET);
        System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| || |/ _ \\ |/ /" + RESET);
        System.out.println(BOLD + YELLOW + "| |__| | (_) | || |  __/   < " + RESET);
        System.out.println(BOLD + YELLOW + " \\_____|\\___/|_||_|\\___|_|\\_\\" + RESET);
        System.out.println();
    }

    public void printModelInfo(String modelId, String providerId, String format, String outputFile,
            boolean isInteractive) {
        if (jsonMode)
            return;
        System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", modelId);
        String provStr = providerId != null ? providerId : "auto-select";
        if (format != null && !format.isBlank()) {
            System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s, format=%s" + RESET + "%n", provStr,
                    format.toLowerCase());
        } else {
            System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n", provStr);
        }
        if (outputFile != null) {
            System.out.printf(BOLD + "Output: " + RESET + YELLOW + "%s" + RESET + "%n", outputFile);
        }
        if (isInteractive) {
            System.out.println(
                    DIM + "Commands: 'exit' to quit, '/reset' to clear history, '/retry' to rerun the last request."
                            + RESET);
        }
        System.out.println(DIM + "-".repeat(50) + RESET);
    }

    public void printAssistantPrefix(boolean quiet, boolean streaming) {
        if (jsonMode)
            return;
        if (!quiet) {
            if (!streaming)
                System.out.print("\n");
            System.out.print(BOLD + GREEN + "Assistant: " + RESET);
            System.out.flush();
        }
    }

    public void printStats(int tokens, double duration, double tps, boolean quiet) {
        printStats(tokens, duration, tps, null, quiet);
    }

    public void printStats(int tokens, double duration, double tps, Double ttftMs, boolean quiet) {
        if (jsonMode)
            return;
        if (!quiet) {
            if (ttftMs != null) {
                System.out.printf(DIM + "\n[Stream updates: %d, Duration: %.2fs, Speed: %.2f t/s, TTFT: %.2f ms]"
                                + RESET + "%n",
                        tokens, duration, tps, ttftMs);
            } else {
                System.out.printf(DIM + "\n[Stream updates: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET + "%n",
                        tokens, duration, tps);
            }
        }
    }

    public void printBenchmarks(Map<String, Object> metadata, boolean quiet) {
        if (jsonMode || quiet || metadata == null || metadata.isEmpty())
            return;

        System.out.println(DIM + "Performance Metrics:" + RESET);

        Double loadMs = metaDouble(metadata, "bench.load_ms");
        Double prefillTps = metaDouble(metadata, "bench.prefill_tps");
        Double genTps = metaDouble(metadata, "bench.generation_tps");
        Double ttftMs = metaDouble(metadata, "bench.ttft_ms");
        Double tpotMs = metaDouble(metadata, "bench.tpot_ms");
        Integer inputTokens = metaInt(metadata, "tokens.input");
        Integer outputTokens = metaInt(metadata, "tokens.output");

        if (loadMs != null)
            System.out.printf(DIM + "  load time      = %9.2f ms" + RESET + "%n", loadMs);
        if (prefillTps != null && inputTokens != null) {
            System.out.printf(DIM + "  prompt eval    = %9.2f t/s (%d tokens)" + RESET + "%n", prefillTps, inputTokens);
        }
        if (genTps != null && outputTokens != null) {
            System.out.printf(DIM + "  generation     = %9.2f t/s (%d tokens)" + RESET + "%n", genTps, outputTokens);
        }
        if (ttftMs != null)
            System.out.printf(DIM + "  latency (ttft) = %9.2f ms" + RESET + "%n", ttftMs);
        if (tpotMs != null)
            System.out.printf(DIM + "  token latency  = %9.2f ms/token" + RESET + "%n", tpotMs);
        System.out.println();
    }

    private static Double metaDouble(Map<String, Object> metadata, String key) {
        Object v = metadata.get(key);
        return v instanceof Number n ? n.doubleValue() : null;
    }

    private static Integer metaInt(Map<String, Object> metadata, String key) {
        Object v = metadata.get(key);
        return v instanceof Number n ? n.intValue() : null;
    }

    public String getPrompt(boolean quiet) {
        return quiet ? "\n>>> " : "\n" + BOLD + CYAN + ">>> " + RESET;
    }

    public String getSecondaryPrompt(boolean quiet) {
        return quiet ? "... " : DIM + "... " + RESET;
    }

    public void printGoodbye(boolean quiet) {
        if (!quiet)
            System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
    }

    public void printError(String message, boolean quiet) {
        System.err.println("\n" + RED + "Error: " + RESET + message);
    }

    public void printWarning(String message, boolean quiet) {
        if (!quiet)
            System.out.println(YELLOW + "Warning: " + RESET + message);
    }

    public void printInfo(String message, boolean quiet) {
        if (!quiet)
            System.out.println(DIM + message + RESET);
    }

    public void printSuccess(String message, boolean quiet) {
        if (!quiet)
            System.out.println(GREEN + message + RESET);
    }

    public void printHardwareInfo(String hardware, boolean quiet) {
        if (!quiet) {
            System.out.printf(DIM + "[Running on %s]" + RESET + "%n", hardware);
        }
    }
}
