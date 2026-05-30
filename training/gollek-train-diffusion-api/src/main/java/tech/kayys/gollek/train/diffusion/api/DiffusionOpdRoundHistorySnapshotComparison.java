package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed baseline-versus-current comparison for one round-history snapshot.
 */
public record DiffusionOpdRoundHistorySnapshotComparison(
        DiffusionOpdRoundHistorySnapshot baseline,
        DiffusionOpdRoundHistorySnapshot current,
        Double meanLossDelta,
        Double latestLossDelta,
        Boolean improved,
        Boolean statusChanged,
        Boolean recommendationChanged) {
}
