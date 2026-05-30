package tech.kayys.gollek.train.data;

import java.util.Objects;

final class PaddingDiagnostics {
    private PaddingDiagnostics() {
    }

    static PaddingEfficiencyReport paddingEfficiency(Iterable<DataLoader.PaddedBatch> batches) {
        Objects.requireNonNull(batches, "batches must not be null");
        PaddingStats inputs = null;
        PaddingStats labels = null;
        int batchCount = 0;

        for (DataLoader.PaddedBatch batch : batches) {
            Objects.requireNonNull(batch, "padded batch must not be null");
            inputs = merge(inputs, batch.inputPaddingStats());
            labels = merge(labels, batch.labelPaddingStats());
            batchCount++;
        }

        if (batchCount == 0) {
            throw new IllegalArgumentException("batches must contain at least one padded batch");
        }
        return new PaddingEfficiencyReport(inputs, labels);
    }

    private static PaddingStats merge(PaddingStats current, PaddingStats next) {
        return current == null ? next : current.merge(next);
    }
}
