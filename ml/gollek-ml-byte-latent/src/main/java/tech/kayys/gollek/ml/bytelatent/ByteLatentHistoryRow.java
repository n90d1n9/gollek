package tech.kayys.gollek.ml.bytelatent;

/**
 * One persisted epoch-level history row for byte-latent training.
 */
public record ByteLatentHistoryRow(
        int epoch,
        int globalStep,
        int batchCount,
        double trainLoss) {
}
