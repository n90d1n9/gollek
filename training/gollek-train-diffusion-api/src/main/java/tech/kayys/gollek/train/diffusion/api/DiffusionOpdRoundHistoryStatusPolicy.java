package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed policy for evaluating round-history stability into a status/alert view.
 */
public record DiffusionOpdRoundHistoryStatusPolicy(
        String name,
        int minRows,
        double unstableLatestStepMultiplier,
        double unstableLossRangeMultiplier,
        double healthyLatestStepMultiplier,
        boolean requireImprovingForHealthy) {

    public static DiffusionOpdRoundHistoryStatusPolicy defaults() {
        return new DiffusionOpdRoundHistoryStatusPolicy(
                "default",
                2,
                1.0d,
                1.0d,
                1.0d,
                true);
    }

    public static DiffusionOpdRoundHistoryStatusPolicy strict() {
        return new DiffusionOpdRoundHistoryStatusPolicy(
                "strict",
                2,
                0.90d,
                0.90d,
                0.50d,
                true);
    }

    public static DiffusionOpdRoundHistoryStatusPolicy lenient() {
        return new DiffusionOpdRoundHistoryStatusPolicy(
                "lenient",
                2,
                1.25d,
                1.25d,
                1.25d,
                true);
    }
}
