/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.selectLastToken;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class DirectForwardOutputProjection {
    private DirectForwardOutputProjection() {
    }

    static AccelTensor prefillLogits(DirectForwardRuntimeContext runtime,
                                     ModelConfigTraits traits,
                                     MemorySegment hiddenSeg,
                                     long[] hiddenShape,
                                     AccelTensor embeddings,
                                     ResolvedModelWeights resolvedWeights,
                                     ModelConfig config,
                                     KVCacheManager.KVCacheSession.ForwardWorkspace ws,
                                     int seqLen,
                                     boolean verboseTokens) {
        AccelTensor hidden = AccelTensor.view(hiddenSeg, hiddenShape);
        AccelTensor normed = finalNorm(runtime, traits, hidden, hiddenShape,
                resolvedWeights, config, ws, seqLen);
        if (hidden != embeddings && hidden != normed) {
            hidden.close();
        }

        AccelTensor lmHeadW = lmHeadOrThrow(resolvedWeights);
        if (verboseTokens && seqLen > 1) {
            DirectForwardLogits.debugSequencePositionLogits(normed, lmHeadW, seqLen,
                    (input, weight, bias) -> linear(runtime, traits, config,
                            false, input, weight, bias, null, null));
        }

        AccelTensor lastPos = selectLastToken(normed, seqLen);
        AccelTensor logitsOutput = DirectForwardLogits.reusableOutputTensor(ws, lastPos, lmHeadW);
        try {
            return projectLogits(runtime, traits, config,
                    false, lastPos, lmHeadW, logitsOutput);
        } finally {
            lastPos.closeWithParent();
        }
    }

    static AccelTensor decodeLogits(DirectForwardRuntimeContext runtime,
                                    ModelConfigTraits traits,
                                    MemorySegment hiddenSeg,
                                    long[] hiddenShape,
                                    ResolvedModelWeights resolvedWeights,
                                    ModelConfig config,
                                    KVCacheManager.KVCacheSession.ForwardWorkspace ws,
                                    boolean reuseLogitsOutput) {
        AccelTensor finalHidden = AccelTensor.view(hiddenSeg, hiddenShape);
        AccelTensor normed = finalNorm(runtime, traits, finalHidden, hiddenShape,
                resolvedWeights, config, ws, 1);
        if (finalHidden != normed) {
            finalHidden.close();
        }

        AccelTensor lmHeadW = lmHeadOrThrow(resolvedWeights);
        AccelTensor logitsOutput = reuseLogitsOutput
                ? DirectForwardLogits.reusableOutputTensor(ws, normed, lmHeadW)
                : null;
        try {
            return projectLogits(runtime, traits, config,
                    true, normed, lmHeadW, logitsOutput);
        } finally {
            if (normed.dataPtr() != ws.getNormedAttnSeg()) {
                normed.close();
            }
        }
    }

    private static AccelTensor finalNorm(DirectForwardRuntimeContext runtime,
                                         ModelConfigTraits traits,
                                         AccelTensor hidden,
                                         long[] hiddenShape,
                                         ResolvedModelWeights resolvedWeights,
                                         ModelConfig config,
                                         KVCacheManager.KVCacheSession.ForwardWorkspace ws,
                                         int rows) {
        if (runtime.canUseMetalElementwise(traits, rows)) {
            AccelTensor normed = AccelTensor.view(ws.getNormedAttnSeg(), hiddenShape);
            DirectForwardElementwiseOps.rmsNormRowsMetal(
                    runtime.metalBinding(),
                    normed.dataPtr(),
                    hidden.dataPtr(),
                    resolvedWeights.finalNorm().dataPtr(),
                    rows,
                    config.hiddenSize(),
                    (float) config.rmsNormEps(),
                    resolvedWeights.addOneRmsNorm());
            return normed;
        }
        return AccelOps.rmsNorm(hidden, resolvedWeights.finalNorm(), config.rmsNormEps(),
                resolvedWeights.addOneRmsNorm());
    }

    private static AccelTensor projectLogits(DirectForwardRuntimeContext runtime,
                                             ModelConfigTraits traits,
                                             ModelConfig config,
                                             boolean decodeLogitsPhase,
                                             AccelTensor input,
                                             AccelTensor lmHeadW,
                                             AccelTensor outputBuffer) {
        long tLogits0 = System.nanoTime();
        AccelTensor logits = linear(runtime, traits, config,
                decodeLogitsPhase, input, lmHeadW, null, "logits", outputBuffer);
        DirectInferenceProfiler.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
        return logits;
    }

    private static AccelTensor linear(DirectForwardRuntimeContext runtime,
                                      ModelConfigTraits traits,
                                      ModelConfig config,
                                      boolean decodeLogitsPhase,
                                      AccelTensor input,
                                      AccelTensor weight,
                                      AccelTensor bias,
                                      String profileKey,
                                      AccelTensor outputBuffer) {
        return DirectForwardLinearProjection.linear(
                runtime,
                traits,
                config,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
    }

    private static AccelTensor lmHeadOrThrow(ResolvedModelWeights resolvedWeights) {
        AccelTensor lmHeadW = resolvedWeights.lmHead();
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        return lmHeadW;
    }

}
