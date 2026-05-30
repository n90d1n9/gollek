package tech.kayys.gollek.train.data;

import java.util.List;
import tech.kayys.gollek.ml.autograd.GradTensor;

abstract class TensorDatasetSupport implements DataLoader.TensorDatasetAdapter {
    private final List<GradTensor[]> samples;

    TensorDatasetSupport(GradTensor[]... samples) {
        this.samples = TensorDatasetSamples.fromTuples(samples);
    }

    TensorDatasetSupport(List<GradTensor[]> samples) {
        this.samples = TensorDatasetSamples.fromList(samples);
    }

    TensorDatasetSupport(GradTensor inputs, GradTensor targets) {
        this.samples = TensorDatasetSamples.fromBatchedPair(inputs, targets);
    }

    abstract DataLoader.TensorDataset self();

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public GradTensor[] get(int index) {
        return samples.get(index);
    }

    public DataLoader.TensorDatasetSplit split(double trainFraction, long seed) {
        return TensorDatasetSplits.randomSplit(self(), trainFraction, seed);
    }

    public DataLoader.TensorDatasetThreeWaySplit split(
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorDatasetSplits.randomSplit(self(), trainFraction, validationFraction, seed);
    }

    public List<DataLoader.TensorDatasetFold> kFold(int folds, long seed) {
        return TensorDatasetSplits.kFold(self(), folds, seed);
    }

    public List<DataLoader.TensorDatasetFold> repeatedKFold(int folds, int repeats, long seed) {
        return DataLoader.repeatedKFold(self(), folds, repeats, seed);
    }

    public List<DataLoader.TensorDatasetFold> groupKFold(int[] groups, int folds, long seed) {
        return DataLoader.groupKFold(self(), groups, folds, seed);
    }

    public List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.stratifiedGroupKFold(self(), labels, groups, folds, seed);
    }

    public List<DataLoader.TensorDatasetFold> timeSeriesSplit(int splits) {
        return DataLoader.timeSeriesSplit(self(), splits);
    }

    public List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return DataLoader.timeSeriesSplit(self(), splits, validationSize, gap, maxTrainSize);
    }
}
