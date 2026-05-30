package tech.kayys.gollek.ml.data;

import java.util.List;

/**
 * Selects already-formed dataset row batches for one logical loader pass.
 *
 * <p>Use this when batch size is data-dependent, such as token-budgeted
 * sequence training. Fixed-size sampling should continue to use
 * {@link IndexSampler}.</p>
 */
public interface BatchSampler {

    List<List<Integer>> sampleBatches(int datasetSize);

    int sampleCount(int datasetSize);

    int batchCount(int datasetSize);
}
