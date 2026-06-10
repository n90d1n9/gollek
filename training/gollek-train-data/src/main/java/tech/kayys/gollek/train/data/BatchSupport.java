package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.internal.DataLoaderTensorBatchValidationRules;

final class BatchSupport {

    private BatchSupport() {
    }

    static GradTensor requireTensor(String name, GradTensor tensor) {
        return DataLoaderTensorBatchValidationRules.requireTensor(name, tensor);
    }

    static void requireCompatibleBatchDimensions(GradTensor inputs, GradTensor labels) {
        DataLoaderTensorBatchValidationRules.requireCompatibleBatchDimensions(inputs, labels);
    }
}
