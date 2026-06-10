package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.List;

/**
 * Selects every dataset row in natural order.
 */
public final class SequentialSampler implements IndexSampler {

    @Override
    public List<Integer> sample(int datasetSize) {
        DataLoaderCounts.requireDatasetSize(datasetSize);
        List<Integer> indices = new ArrayList<>(datasetSize);
        for (int i = 0; i < datasetSize; i++) {
            indices.add(i);
        }
        return indices;
    }

    @Override
    public int sampleCount(int datasetSize) {
        DataLoaderCounts.requireDatasetSize(datasetSize);
        return datasetSize;
    }
}
