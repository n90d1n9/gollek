/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logSegmentStats;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logTensorStats;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.attention.AttentionInput;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardAttentionStage {
    private DirectForwardAttentionStage() {
    }

    static void run(DirectForwardLayerStageContext ctx,
                    FlashAttentionKernel attentionKernel,
                    KVCacheManager.KVCacheSession kvCache,
                    int startPos,
                    Map<Integer, SharedKvState> sharedKvStates) {
        DirectForwardAttentionWeights attentionWeights = DirectForwardAttentionWeights.fromLayer(ctx.layerWeights());
        MemorySegment normedAttnSeg = ctx.workspace().getNormedAttnSeg();
        DirectForwardNorms.rmsNormRows(normedAttnSeg, ctx.hiddenIn(),
                attentionWeights.attentionNormWeight(), ctx.hiddenShape(),
                ctx.seqLen(), ctx.config(), ctx.addOneRmsNorm(), ctx.useMetalElementwise(), ctx.runtime());

        AttentionInput attnIn = DirectForwardAttentionInputs.causalLayerInput(
                ctx,
                attentionWeights,
                AccelTensor.view(normedAttnSeg, ctx.hiddenShape()),
                kvCache,
                startPos,
                sharedKvStates);

        if (ctx.verboseLayers()) {
            System.err.printf("[DEBUG] Layer %d Attention start%n", ctx.layerIdx());
            System.err.flush();
        }
        long tAttention0 = System.nanoTime();
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        DirectInferenceProfiler.recordAttentionNanos(System.nanoTime() - tAttention0);
        if (ctx.verboseLayers()) {
            logTensorStats(attnOut, "layer " + ctx.layerIdx() + " attnOut");
        }

        DirectForwardElementwiseOps.residualAdd(ctx.runtime().log(), ctx.runtime().metalBinding(),
                ctx.hiddenIn(), attnOut, ctx.hiddenOut(),
                ctx.seqLen(), ctx.config().hiddenSize(), ctx.useNativeElementwiseAdd());
        attnOut.close();
        if (ctx.verboseLayers()) {
            logSegmentStats(ctx.hiddenOut(), ctx.hiddenShape(), "layer " + ctx.layerIdx() + " postAttnResidual");
        }
    }
}
