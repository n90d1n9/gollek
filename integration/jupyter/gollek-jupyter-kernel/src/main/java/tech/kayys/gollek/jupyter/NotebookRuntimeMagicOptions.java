package tech.kayys.gollek.jupyter;

final class NotebookRuntimeMagicOptions {

    private NotebookRuntimeMagicOptions() {
    }

    record TimeitOptions(int runs, String expression) {}

    static TimeitOptions parseTimeitOptions(String raw) {
        String usage = "Usage: %timeit [-n N] <java-expression-or-cell>";
        String expression = raw == null ? "" : raw.trim();
        if (expression.isBlank()) {
            throw new IllegalArgumentException(usage);
        }

        int runs = 5;
        if (expression.startsWith("-n ")) {
            String remainder = expression.substring(3).trim();
            int split = remainder.indexOf(' ');
            if (split <= 0) {
                throw new IllegalArgumentException(usage);
            }
            String runsToken = remainder.substring(0, split).trim();
            try {
                runs = Integer.parseInt(runsToken);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid run count for %timeit: " + runsToken);
            }
            expression = remainder.substring(split + 1).trim();
        }

        if (runs <= 0) {
            throw new IllegalArgumentException("Run count for %timeit must be > 0");
        }
        if (expression.isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        return new TimeitOptions(runs, expression);
    }
}
