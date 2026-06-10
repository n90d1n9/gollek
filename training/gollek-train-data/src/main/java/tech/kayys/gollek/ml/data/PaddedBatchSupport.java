package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import tech.kayys.gollek.train.data.internal.DataLoaderTensorBatchValidationRules;

final class PaddedBatchSupport {
    private PaddedBatchSupport() {
    }

    static GradTensor requireTensor(String name, GradTensor tensor) {
        return DataLoaderTensorBatchValidationRules.requireTensor(name, tensor);
    }

    static int[] copyLengths(String name, int[] lengths) {
        return DataLoaderTensorBatchValidationRules.copyLengths(name, lengths);
    }

    static void validate(String name, GradTensor values, GradTensor mask, int[] lengths) {
        DataLoaderTensorBatchValidationRules.validatePadded(name, values, mask, lengths);
    }

    static void validateAlignedBatchSize(GradTensor inputs, GradTensor labels) {
        BatchSupport.requireCompatibleBatchDimensions(inputs, labels);
    }

    static PaddingStats stats(String name, GradTensor values, int[] lengths) {
        return PaddingStats.fromLengths(paddedLength(name, values), lengths);
    }

    private static int paddedLength(String name, GradTensor values) {
        return DataLoaderTensorBatchValidationRules.paddedLength(name, values);
    }
}
