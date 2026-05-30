package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.List;
import java.util.Objects;

/**
 * Audits train/validation/test splits for multimodal training workflows.
 */
public final class MultimodalThreeWaySplitDiagnostics {
    private MultimodalThreeWaySplitDiagnostics() {
    }

    public static MultimodalThreeWaySplitReport inspect(Dataset.ThreeWaySplit<List<MultimodalContent>> split) {
        Objects.requireNonNull(split, "split must not be null");
        if (split.train().size() == 0) {
            throw new IllegalArgumentException("train split must not be empty");
        }
        if (split.validation().size() == 0) {
            throw new IllegalArgumentException("validation split must not be empty");
        }
        if (split.test().size() == 0) {
            throw new IllegalArgumentException("test split must not be empty");
        }
        return new MultimodalThreeWaySplitReport(
                split.train().size(),
                split.validation().size(),
                split.test().size(),
                MultimodalSplitDiagnostics.inspect(new Dataset.Split<>(split.train(), split.validation())),
                MultimodalSplitDiagnostics.inspect(new Dataset.Split<>(split.train(), split.test())),
                MultimodalSplitDiagnostics.inspect(new Dataset.Split<>(split.validation(), split.test())));
    }
}
