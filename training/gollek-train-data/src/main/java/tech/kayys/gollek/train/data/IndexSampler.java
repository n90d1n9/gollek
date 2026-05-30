package tech.kayys.gollek.train.data;

import java.util.List;

/**
 * Selects dataset row indices for one logical pass through a tensor data loader.
 */
public interface IndexSampler {

    List<Integer> sample(int datasetSize);

    int sampleCount(int datasetSize);
}
