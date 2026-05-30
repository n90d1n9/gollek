/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logSegmentStats;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logTensorStats;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.attention.AttentionInput;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardTransformerLayer {
    private DirectForwardTransformerLayer() {
    }

    static void forward(DirectForwardRuntimeContext runtime,
                        FlashAttentionKernel attentionKernel,
                        MoeForwardPass moeForwardPass,
                        ModelConfigTraits traits,
                        MemorySegment hiddenIn,
                        MemorySegment hiddenOut,
                        AccelTensor perLayerInput,
                        Map<String, AccelTensor> weights,
                        ModelConfig config,
                        ModelArchitecture arch,
                        KVCacheManager.KVCacheSession kvCache,
                        int layerIdx,
                        int startPos,
                        int seqLen,
                        long[] hiddenShape,
                        KVCacheManager.KVCacheSession.ForwardWorkspace ws,
                        Map<Integer, SharedKvState> sharedKvStates,
                        ResolvedLayerWeights layerWeights,
                        ResolvedModelWeights resolvedWeights,
                        DirectForwardOperators operators) {
        boolean verboseLayers = DirectForwardExecutionOptions.verboseLayersEnabled();
        boolean useMetalElementwise = runtime.canUseMetalElementwise(traits, seqLen);
        boolean useNativeElementwiseAdd = useMetalElementwise
                && runtime.capabilities().nativeElementwiseKernelsAvailable();

        MemorySegment normedAttnSeg = ws.getNormedAttnSeg();
        rmsNormRows(normedAttnSeg, hiddenIn, layerWeights.attentionNormWeight(), hiddenShape,
                seqLen, config, resolvedWeights.addOneRmsNorm(), useMetalElementwise, runtime);

        AttentionInput attnIn = new AttentionInput(
                AccelTensor.view(normedAttnSeg, hiddenShape),
                layerWeights.queryWeight(),
                layerWeights.keyWeight(),
                layerWeights.valueWeight(),
                layerWeights.outputWeight(),
                layerWeights.queryBias(),
                layerWeights.keyBias(),
                layerWeights.valueBias(),
                layerWeights.outputBias(),
                arch, config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                layerWeights.queryNormWeight(),
                layerWeights.keyNormWeight(),
                layerWeights.postAttnNormWeight(),
                sharedKvStates,
                ws.getNormedFfnSeg());

        if (verboseLayers) {
            System.err.printf("[DEBUG] Layer %d Attention start%n", layerIdx);
            System.err.flush();
        }
        long tAttention0 = System.nanoTime();
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        DirectInferenceProfiler.recordAttentionNanos(System.nanoTime() - tAttention0);
        if (verboseLayers) {
            logTensorStats(attnOut, "layer " + layerIdx + " attnOut");
        }

        DirectForwardElementwiseOps.residualAdd(runtime.log(), runtime.metalBinding(), hiddenIn, attnOut, hiddenOut,
                seqLen, config.hiddenSize(), useNativeElementwiseAdd);
        attnOut.close();
        if (verboseLayers) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postAttnResidual");
        }

        AccelTensor preFfnNormW = layerWeights.preFfnNormWeight();
        AccelTensor postFfnNormW = layerWeights.postFfnNormWeight();

        MemorySegment normedFfnSeg = ws.getNormedFfnSeg();
        rmsNormRows(normedFfnSeg, hiddenOut, preFfnNormW, hiddenShape,
                seqLen, config, resolvedWeights.addOneRmsNorm(), useMetalElementwise, runtime);

        long tFfn0 = System.nanoTime();
        AccelTensor mlpOutputBuffer = AccelTensor.view(hiddenIn, hiddenShape);
        AccelTensor mlpIn = AccelTensor.view(normedFfnSeg, hiddenShape);
        AccelTensor mlpOut = config.isMoeLayer(layerIdx)
                ? moeForwardPass.computeAccel(mlpIn, weights, config, arch, layerIdx)
                : operators.swigluFfn(mlpIn, arch, config,
                        layerWeights.ffnGateWeight(), layerWeights.ffnGateBias(),
                        layerWeights.ffnUpWeight(), layerWeights.ffnUpBias(),
                        layerWeights.ffnDownWeight(), layerWeights.ffnDownBias(), ws,
                        mlpOutputBuffer);
        DirectInferenceProfiler.recordFfnNanos(System.nanoTime() - tFfn0);
        if (verboseLayers) {
            logTensorStats(mlpOut, "layer " + layerIdx + " mlpOut");
        }

        AccelTensor mlpNormed;
        if (postFfnNormW != null) {
            if (useMetalElementwise && DirectForwardElementwisePolicy.shouldUseMetalPostFfnNorm(traits)) {
                DirectForwardElementwiseOps.rmsNormRowsMetal(runtime.metalBinding(), normedFfnSeg, mlpOut.dataPtr(),
                        postFfnNormW.dataPtr(), seqLen, config.hiddenSize(), (float) config.rmsNormEps(),
                        resolvedWeights.addOneRmsNorm());
                mlpNormed = AccelTensor.view(normedFfnSeg, hiddenShape);
            } else {
                mlpNormed = AccelOps.rmsNorm(mlpOut, postFfnNormW, config.rmsNormEps(),
                        resolvedWeights.addOneRmsNorm());
            }
            mlpOut.close();
        } else {
            mlpNormed = mlpOut;
        }

        DirectForwardElementwiseOps.residualAdd(runtime.log(), runtime.metalBinding(), hiddenOut, mlpNormed, hiddenOut,
                seqLen, config.hiddenSize(), useNativeElementwiseAdd);
        mlpNormed.close();
        if (verboseLayers) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postFfnResidual");
        }

        DirectForwardPerLayerInputs.applyResidual(
                hiddenOut,
                hiddenShape,
                seqLen,
                config,
                arch,
                layerWeights,
                perLayerInput,
                ws,
                useMetalElementwise,
                useNativeElementwiseAdd,
                resolvedWeights.addOneRmsNorm(),
                runtime,
                operators);
        if (layerWeights.hasLayerScalar()
                && DirectForwardElementwisePolicy.shouldApplyLayerScalar(traits)) {
            DirectForwardElementwiseOps.scaleSegmentInPlace(
                    runtime.log(),
                    runtime.metalBinding(),
                    hiddenOut,
                    seqLen,
                    config.hiddenSize(),
                    layerWeights.layerScalarValue(),
                    runtime.canUseMetalLayerScalarScale(useMetalElementwise, seqLen));
        }
        if (verboseLayers && perLayerInput != null) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postPle");
        }
        if (verboseLayers && layerWeights.hasLayerScalar()) {
            System.err.printf("[DEBUG] layer %d layerScalar=%f%n", layerIdx, layerWeights.layerScalarValue());
        }
    }

    private static void rmsNormRows(MemorySegment out,
                                    MemorySegment in,
                                    AccelTensor weight,
                                    long[] shape,
                                    int seqLen,
                                    ModelConfig config,
                                    boolean addOne,
                                    boolean useMetalElementwise,
                                    DirectForwardRuntimeContext runtime) {
        if (useMetalElementwise) {
            DirectForwardElementwiseOps.rmsNormRowsMetal(runtime.metalBinding(), out, in,
                    weight.dataPtr(), seqLen, config.hiddenSize(), (float) config.rmsNormEps(), addOne);
            return;
        }

        AccelTensor inTensor = AccelTensor.view(in, shape);
        AccelTensor outTensor = AccelOps.rmsNorm(inTensor, weight, config.rmsNormEps(), addOne);
        MemorySegment.copy(outTensor.dataPtr(), 0, out, 0,
                (long) seqLen * config.hiddenSize() * Float.BYTES);
        outTensor.close();
    }

}
