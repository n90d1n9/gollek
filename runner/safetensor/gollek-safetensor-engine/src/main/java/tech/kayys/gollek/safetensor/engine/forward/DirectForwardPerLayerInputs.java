/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class DirectForwardPerLayerInputs {
    private DirectForwardPerLayerInputs() {
    }

    static AccelTensor[] build(long[] inputIds, AccelTensor inputsEmbeds,
            ModelConfigTraits traits, ModelConfig config, ResolvedModelWeights resolvedWeights,
            DirectForwardOperators operators) {
        if (!DirectForwardElementwisePolicy.shouldBuildPerLayerInputs(
                traits,
                config.hiddenSizePerLayerInput())) {
            return null;
        }

        AccelTensor packedPleEmbeddings = resolvedWeights.packedPleEmbeddings();
        AccelTensor pleProjection = resolvedWeights.pleProjection();
        AccelTensor pleProjectionNorm = resolvedWeights.pleProjectionNorm();
        if (packedPleEmbeddings == null || pleProjection == null || pleProjectionNorm == null) {
            return null;
        }

        int numLayers = config.numHiddenLayers();
        int pleDim = config.hiddenSizePerLayerInput();
        int seqLen = inputIds.length;

        AccelTensor tokenPleRaw = packedPleEmbeddings.indexSelect(inputIds)
                .reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor tokenPle = AccelOps.mulScalar(tokenPleRaw, (float) Math.sqrt(Math.max(1, pleDim)));
        tokenPleRaw.close();
        AccelTensor projectedPle = operators.linear(inputsEmbeds, pleProjection, null, "ple_projection", config);
        float pleProjectionScale = (float) (1.0 / Math.sqrt(Math.max(1, config.hiddenSize())));
        AccelTensor projectedPleScaled = AccelOps.mulScalar(projectedPle, pleProjectionScale);
        projectedPle.close();

        AccelTensor projectedPle4d = projectedPleScaled.reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor projectedPleNormed = AccelOps.rmsNorm(projectedPle4d, pleProjectionNorm,
                config.rmsNormEps(), resolvedWeights.addOneRmsNorm());
        projectedPle4d.close();

        AccelTensor combinedPle = AccelOps.add(projectedPleNormed, tokenPle);
        projectedPleNormed.close();
        tokenPle.close();

        AccelTensor scaledPle = AccelOps.mulScalar(combinedPle, (float) (1.0 / Math.sqrt(2.0)));
        combinedPle.close();

        AccelTensor[] layers = new AccelTensor[numLayers];
        if (seqLen == 1) {
            MemorySegment packed = scaledPle.dataPtr();
            long layerBytes = (long) pleDim * Float.BYTES;
            for (int layer = 0; layer < numLayers; layer++) {
                layers[layer] = AccelTensor.view(
                        packed.asSlice((long) layer * layerBytes, layerBytes),
                        new long[] { 1L, 1L, pleDim },
                        scaledPle);
            }
            return layers;
        }

        MemorySegment src = scaledPle.dataPtr();
        for (int layer = 0; layer < numLayers; layer++) {
            AccelTensor layerTensor = AccelTensor.zeros(new long[] { 1L, seqLen, pleDim });
            MemorySegment dst = layerTensor.dataPtr();
            for (int token = 0; token < seqLen; token++) {
                long srcIndex = ((long) token * numLayers + layer) * pleDim;
                long dstIndex = (long) token * pleDim;
                MemorySegment.copy(src, srcIndex * Float.BYTES, dst, dstIndex * Float.BYTES,
                        (long) pleDim * Float.BYTES);
            }
            layers[layer] = layerTensor;
        }
        scaledPle.close();
        return layers;
    }

    static boolean needed(ModelConfigTraits traits, ModelConfig config, ResolvedModelWeights resolvedWeights) {
        if (!DirectForwardElementwisePolicy.shouldBuildPerLayerInputs(
                traits,
                config.hiddenSizePerLayerInput())) {
            return false;
        }
        return resolvedWeights.packedPleEmbeddings() != null
                && resolvedWeights.pleProjection() != null
                && resolvedWeights.pleProjectionNorm() != null;
    }

    static void applyResidual(MemorySegment hiddenSeg, long[] hiddenShape, int seqLen,
            ModelConfig config, ModelArchitecture arch, ResolvedLayerWeights layerWeights,
            AccelTensor perLayerInput, ForwardWorkspace ws,
            boolean useMetalElementwise, boolean useNativeElementwiseAdd, boolean addOneRmsNorm,
            DirectForwardRuntimeContext runtime, DirectForwardOperators operators) {
        if (perLayerInput == null) {
            return;
        }

        AccelTensor gateWeight = layerWeights.perLayerInputGateWeight();
        AccelTensor projectionWeight = layerWeights.perLayerProjectionWeight();
        AccelTensor normWeight = layerWeights.postPerLayerInputNormWeight();
        if (gateWeight == null || projectionWeight == null || normWeight == null) {
            return;
        }

        AccelTensor hidden = AccelTensor.view(hiddenSeg, hiddenShape);
        AccelTensor gate = operators.linear(hidden, gateWeight, null, "per_layer_gate", config);
        hidden.close();

        AccelTensor mixed = activationMultiply(gate, perLayerInput, arch, ws, useMetalElementwise,
                runtime);
        gate.close();

        AccelTensor projected = operators.linear(mixed, projectionWeight, null, "per_layer_projection", config);
        mixed.close();

        AccelTensor normed;
        if (useMetalElementwise && ws != null) {
            normed = AccelTensor.view(ws.getNormedFfnSeg(), hiddenShape);
            DirectForwardElementwiseOps.rmsNormRowsMetal(runtime.metalBinding(), normed.dataPtr(), projected.dataPtr(),
                    normWeight.dataPtr(), seqLen, config.hiddenSize(), (float) config.rmsNormEps(), addOneRmsNorm);
        } else {
            normed = AccelOps.rmsNorm(projected, normWeight, config.rmsNormEps(), addOneRmsNorm);
        }
        projected.close();

        DirectForwardElementwiseOps.residualAdd(runtime.log(), runtime.metalBinding(), hiddenSeg, normed, hiddenSeg,
                seqLen, config.hiddenSize(), useNativeElementwiseAdd);
        normed.close();
    }

    static void close(AccelTensor[] perLayerInputs) {
        if (perLayerInputs == null) {
            return;
        }
        for (AccelTensor perLayerInput : perLayerInputs) {
            if (perLayerInput != null) {
                perLayerInput.closeWithParent();
            }
        }
    }

    private static AccelTensor activationMultiply(AccelTensor gate, AccelTensor perLayerInput,
            ModelArchitecture arch, ForwardWorkspace ws, boolean useMetalElementwise,
            DirectForwardRuntimeContext runtime) {
        long[] shape = gate.shape();
        int elements = Math.toIntExact(gate.numel());
        if (useMetalElementwise && ws != null && runtime.metalBinding() != null) {
            AccelTensor mixed = DirectForwardTensorOps.reusableWorkspaceView(ws.getCombinedSeg(), shape);
            if (mixed == null) {
                mixed = AccelTensor.zeros(shape);
            }
            try {
                int rc = arch.activationType() == FFNActivationType.GELU
                        ? runtime.metalBinding().geluFfn(
                                mixed.dataPtr(), gate.dataPtr(), perLayerInput.dataPtr(), elements)
                        : runtime.metalBinding().siluFfn(
                                mixed.dataPtr(), gate.dataPtr(), perLayerInput.dataPtr(), elements);
                if (rc == 0) {
                    return mixed;
                }
            } catch (IllegalStateException | UnsupportedOperationException e) {
                runtime.log().debugf("Falling back from Metal per-layer activation multiply: %s", e.getMessage());
            }
            mixed.close();
        }

        AccelTensor activated = arch.activationType() == FFNActivationType.GELU
                ? AccelOps.gelu(gate)
                : AccelOps.silu(gate);
        try {
            return AccelOps.mul(activated, perLayerInput);
        } finally {
            activated.close();
        }
    }
}
