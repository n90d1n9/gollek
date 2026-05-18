package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Tiny deterministic reference implementation for byte-latent forward passes.
 */
public final class ReferenceByteLatentModel implements ByteLatentModel {
    private final ByteLatentModelSpec spec;
    private final List<ByteLatentTransformerBlock> blocks;

    public ReferenceByteLatentModel(ByteLatentModelSpec spec) {
        this.spec = Objects.requireNonNull(spec, "spec must not be null");
        this.blocks = java.util.stream.IntStream.range(0, spec.transformerLayerCount())
                .mapToObj(index -> new ByteLatentTransformerBlock(spec, true))
                .toList();
    }

    @Override
    public ByteLatentModelSpec spec() {
        return spec;
    }

    @Override
    public ByteLatentForwardPass forward(ByteSequenceWindowBatch batch) {
        Objects.requireNonNull(batch, "batch must not be null");
        int[][] predicted = new int[batch.batchSize()][batch.sequenceLength()];
        boolean[][] attentionMask = batch.attentionMask();
        int[][] inputIds = batch.inputIds();
        int[][] targetIds = batch.targetIds();
        int vocab = spec.byteVocabularySize();
        double lossSum = 0.0d;
        int tokenCount = 0;
        for (int batchIndex = 0; batchIndex < batch.batchSize(); batchIndex++) {
            for (int position = 0; position < batch.sequenceLength(); position++) {
                int inputId = inputIds[batchIndex][position];
                int transformed = inputId;
                for (ByteLatentTransformerBlock block : blocks) {
                    transformed = block.transformToken(transformed, position);
                }
                int predictedId = Math.floorMod(transformed + spec.attentionHeadCount(), vocab);
                predicted[batchIndex][position] = predictedId;
                if (!attentionMask[batchIndex][position]) {
                    continue;
                }
                int targetId = targetIds[batchIndex][position];
                lossSum += Math.abs(predictedId - targetId) / (double) Math.max(1, vocab - 1);
                tokenCount++;
            }
        }
        double meanLoss = tokenCount == 0 ? 0.0d : lossSum / tokenCount;
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("modelType", "reference-byte-latent");
        metadata.put("familyId", familyId());
        metadata.put("layerCount", blocks.size());
        metadata.put("attentionHeadCount", spec.attentionHeadCount());
        metadata.put("latentDimension", spec.latentDimension());
        metadata.put("tokenCount", tokenCount);
        return new ByteLatentForwardPass(predicted, meanLoss, tokenCount, metadata);
    }
}
