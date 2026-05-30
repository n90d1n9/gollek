package tech.kayys.gollek.train.data;

interface TrainValidationTensorDatasetView {
    DataLoader.TensorDataset train();

    DataLoader.TensorDataset validation();

    default DataLoader.TensorBuilder trainLoader() {
        return DataLoader.tensorBuilder(train());
    }

    default DataLoader.TensorBuilder validationLoader() {
        return DataLoader.tensorBuilder(validation());
    }

    default DataLoader.TensorDataLoader trainLoader(int batchSize) {
        return trainLoader().batchSize(batchSize).build();
    }

    default DataLoader.TensorDataLoader trainLoader(int batchSize, boolean shuffle, long seed) {
        return trainLoader().batchSize(batchSize).shuffle(shuffle).seed(seed).build();
    }

    default DataLoader.TensorDataLoader validationLoader(int batchSize) {
        return validationLoader().batchSize(batchSize).build();
    }
}

interface ThreeWayTensorDatasetView extends TrainValidationTensorDatasetView {
    DataLoader.TensorDataset test();

    default DataLoader.TensorDatasetSplit trainValidation() {
        return new DataLoader.TensorDatasetSplit(train(), validation());
    }

    default DataLoader.TensorBuilder testLoader() {
        return DataLoader.tensorBuilder(test());
    }

    default DataLoader.TensorDataLoader testLoader(int batchSize) {
        return testLoader().batchSize(batchSize).build();
    }
}
