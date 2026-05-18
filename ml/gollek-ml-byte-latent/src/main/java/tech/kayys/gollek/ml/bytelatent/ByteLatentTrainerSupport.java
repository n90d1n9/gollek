package tech.kayys.gollek.ml.bytelatent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Trainer-preparation helpers for byte-latent dataset flows.
 */
public final class ByteLatentTrainerSupport {
    private ByteLatentTrainerSupport() {
    }

    public static ByteLatentTrainingPlan plan(
            TextByteSequenceDataset dataset,
            ByteLatentTrainerConfig config) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(config, "config must not be null");

        List<Integer> indices = new ArrayList<>(dataset.size());
        for (int i = 0; i < dataset.size(); i++) {
            indices.add(i);
        }
        if (config.shuffle()) {
            Collections.shuffle(indices, new Random(config.seed()));
        }

        List<ByteSequenceWindowBatch> batches = new ArrayList<>();
        for (int start = 0; start < indices.size(); start += config.batchSize()) {
            int end = Math.min(start + config.batchSize(), indices.size());
            List<byte[]> rows = new ArrayList<>(end - start);
            for (int i = start; i < end; i++) {
                rows.add(dataset.get(indices.get(i)));
            }
            ByteSequenceBatch batch = new ByteSequenceBatch(rows);
            batches.add(ByteSequenceCollator.causalLanguageModeling(
                    batch,
                    config.windowLength(),
                    config.padTokenId()));
        }

        return new ByteLatentTrainingPlan(config, dataset.size(), batches.size(), batches);
    }
}
