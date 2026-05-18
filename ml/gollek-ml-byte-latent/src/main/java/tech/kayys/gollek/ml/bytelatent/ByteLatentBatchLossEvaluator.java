package tech.kayys.gollek.ml.bytelatent;

/**
 * Hook for custom per-batch loss evaluation in byte-latent trainer sessions.
 */
@FunctionalInterface
public interface ByteLatentBatchLossEvaluator {
    double evaluate(ByteSequenceWindowBatch batch, ByteLatentTrainerConfig config, int epoch, int batchIndex);
}
