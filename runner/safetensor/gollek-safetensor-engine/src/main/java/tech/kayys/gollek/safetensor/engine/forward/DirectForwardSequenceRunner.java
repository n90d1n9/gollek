/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardSequenceRunner {
    private DirectForwardSequenceRunner() {
    }

    static AccelTensor prefillTokenIds(DirectForwardRuntimeContext runtime,
                                       FlashAttentionKernel attentionKernel,
                                       MoeForwardPass moeForwardPass,
                                       ModelConfigTraits traits,
                                       long[] inputIds,
                                       Map<String, AccelTensor> weights,
                                       ModelConfig config,
                                       ModelArchitecture arch,
                                       KVCacheManager.KVCacheSession kvCache,
                                       ResolvedModelWeights resolvedWeights,
                                       DirectForwardOperators operators) {
        AccelTensor embedTable = resolvedWeights.embedTokens();
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        }
        int seqLen = inputIds.length;
        if (seqLen < 1) {
            throw new IllegalArgumentException("Prompt must result in at least one token.");
        }
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());
        long[] hiddenShape = new long[] { 1L, seqLen, config.hiddenSize() };
        float scale = arch.embeddingScaleFactor((int) embedTable.size(-1));
        embedTable.copyRowsToFloatSegment(inputIds, ws.getHiddenASeg(), scale);
        AccelTensor hidden = AccelTensor.view(ws.getHiddenASeg(), hiddenShape);
        AccelTensor[] perLayerInputs = DirectForwardPerLayerInputs.build(
                inputIds, hidden, traits, config, resolvedWeights, operators);
        try {
            return prefillEmbeddings(
                    runtime,
                    attentionKernel,
                    moeForwardPass,
                    traits,
                    hidden,
                    inputIds,
                    perLayerInputs,
                    weights,
                    config,
                    arch,
                    kvCache,
                    resolvedWeights,
                    true,
                    operators);
        } finally {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            hidden.closeWithParent();
        }
    }

    static AccelTensor prefillEmbeddings(DirectForwardRuntimeContext runtime,
                                         FlashAttentionKernel attentionKernel,
                                         MoeForwardPass moeForwardPass,
                                         ModelConfigTraits traits,
                                         AccelTensor embeddings,
                                         long[] inputIds,
                                         AccelTensor[] perLayerInputs,
                                         Map<String, AccelTensor> weights,
                                         ModelConfig config,
                                         ModelArchitecture arch,
                                         KVCacheManager.KVCacheSession kvCache,
                                         ResolvedModelWeights resolvedWeights,
                                         boolean embeddingsAlreadyInWorkspace,
                                         DirectForwardOperators operators) {
        ensureDirectFfnSupported(config);
        int startPos = kvCache.currentPos();
        long seqLen = embeddings.size(1);
        if (seqLen < 1) {
            throw new IllegalArgumentException(
                    "Invalid sequence length: " + seqLen + ". Prompt must result in at least one token.");
        }
        int seqLenInt = Math.toIntExact(seqLen);
        long[] hiddenShape = new long[] { 1L, seqLen, config.hiddenSize() };
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

        if (!embeddingsAlreadyInWorkspace) {
            MemorySegment.copy(embeddings.dataPtr(), 0, ws.getHiddenASeg(), 0,
                    (long) seqLen * config.hiddenSize() * Float.BYTES);
        }

        MemorySegment currentHidden = ws.getHiddenASeg();
        MemorySegment nextHidden = ws.getHiddenBSeg();
        boolean verboseTokens = DirectForwardExecutionOptions.verboseTokensEnabled();
        Map<Integer, SharedKvState> sharedKvStates = sharedKvStatesForPrefill(config, kvCache, startPos);
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            if (verboseTokens) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d start%n", i, config.numHiddenLayers());
                System.err.flush();
            }
            DirectForwardTransformerLayer.forward(
                    runtime,
                    attentionKernel,
                    moeForwardPass,
                    traits,
                    currentHidden,
                    nextHidden,
                    perLayerInputs != null ? perLayerInputs[i] : null,
                    weights,
                    config,
                    arch,
                    kvCache,
                    i,
                    startPos,
                    seqLenInt,
                    hiddenShape,
                    ws,
                    sharedKvStates,
                    resolvedWeights.layer(i),
                    resolvedWeights,
                    operators);

            MemorySegment temp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = temp;
        }

        AccelTensor logits = DirectForwardOutputProjection.prefillLogits(
                runtime,
                traits,
                currentHidden,
                hiddenShape,
                embeddings,
                resolvedWeights,
                config,
                ws,
                seqLenInt,
                verboseTokens);
        kvCache.advance(seqLenInt);
        return logits;
    }

    static AccelTensor decodeToken(DirectForwardRuntimeContext runtime,
                                   FlashAttentionKernel attentionKernel,
                                   MoeForwardPass moeForwardPass,
                                   ModelConfigTraits traits,
                                   long tokenId,
                                   int startPos,
                                   Map<String, AccelTensor> weights,
                                   ModelConfig config,
                                   ModelArchitecture arch,
                                   KVCacheManager.KVCacheSession kvCache,
                                   ResolvedModelWeights resolvedWeights,
                                   boolean reuseLogitsOutput,
                                   DirectForwardOperators operators) {
        ensureDirectFfnSupported(config);
        AccelTensor embedTable = resolvedWeights.embedTokens();
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight.");
        }
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity(config.hiddenSize(), config.hiddenSize(), config.intermediateSize());
        long[] tokenHiddenShape = new long[] { 1L, 1L, config.hiddenSize() };
        float scale = arch.embeddingScaleFactor((int) embedTable.size(-1));
        embedTable.copyRowToFloatSegment(tokenId, ws.getHiddenASeg(), scale);

        AccelTensor hiddenView = null;
        AccelTensor[] perLayerInputs = null;
        try {
            if (DirectForwardPerLayerInputs.needed(traits, config, resolvedWeights)) {
                long[] tokenIds = { tokenId };
                hiddenView = AccelTensor.view(ws.getHiddenASeg(), tokenHiddenShape);
                perLayerInputs = DirectForwardPerLayerInputs.build(
                        tokenIds, hiddenView, traits, config, resolvedWeights, operators);
            }

            MemorySegment currentHidden = ws.getHiddenASeg();
            MemorySegment nextHidden = ws.getHiddenBSeg();
            Map<Integer, SharedKvState> sharedKvStates = sharedKvStatesForDecode(config, kvCache);
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                DirectForwardTransformerLayer.forward(
                        runtime,
                        attentionKernel,
                        moeForwardPass,
                        traits,
                        currentHidden,
                        nextHidden,
                        perLayerInputs != null ? perLayerInputs[i] : null,
                        weights,
                        config,
                        arch,
                        kvCache,
                        i,
                        startPos,
                        1,
                        tokenHiddenShape,
                        ws,
                        sharedKvStates,
                        resolvedWeights.layer(i),
                        resolvedWeights,
                        operators);

                MemorySegment temp = currentHidden;
                currentHidden = nextHidden;
                nextHidden = temp;
            }

            AccelTensor logits = DirectForwardOutputProjection.decodeLogits(
                    runtime,
                    traits,
                    currentHidden,
                    tokenHiddenShape,
                    resolvedWeights,
                    config,
                    ws,
                    reuseLogitsOutput);
            kvCache.advance(1);
            return logits;
        } finally {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            if (hiddenView != null) {
                hiddenView.close();
            }
        }
    }

    private static Map<Integer, SharedKvState> sharedKvStatesForPrefill(
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache,
            int startPos) {
        if (config.resolvedNumKvSharedLayers() <= 0) {
            return null;
        }
        if (startPos == 0) {
            kvCache.clearSharedKvStates();
        }
        return kvCache.sharedKvStates();
    }

    private static Map<Integer, SharedKvState> sharedKvStatesForDecode(
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache) {
        if (config.resolvedNumKvSharedLayers() <= 0) {
            return null;
        }
        return kvCache.sharedKvStates();
    }

    private static void ensureDirectFfnSupported(ModelConfig config) {
        if (config != null && config.requiresGemma4PackedMoeRuntime()) {
            throw new UnsupportedOperationException(
                    "Gemma4 packed MoE FFN is not supported by the direct SafeTensor runtime yet. "
                            + "Use a GGUF/backend plugin with Gemma4 packed experts support, or load a dense Gemma4 text checkpoint.");
        }
    }
}
