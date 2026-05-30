package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Audits all folds in a multimodal cross-validation run.
 */
public final class MultimodalCrossValidationDiagnostics {
    private MultimodalCrossValidationDiagnostics() {
    }

    public static MultimodalCrossValidationReport inspect(
            List<Dataset.Fold<List<MultimodalContent>>> folds) {
        Objects.requireNonNull(folds, "folds must not be null");
        if (folds.isEmpty()) {
            throw new IllegalArgumentException("folds must not be empty");
        }

        List<MultimodalSplitReport> reports = new ArrayList<>(folds.size());
        for (Dataset.Fold<List<MultimodalContent>> fold : folds) {
            Objects.requireNonNull(fold, "folds must not contain null");
            reports.add(MultimodalSplitDiagnostics.inspect(
                    new Dataset.Split<>(fold.train(), fold.validation())));
        }
        return new MultimodalCrossValidationReport(reports);
    }
}
