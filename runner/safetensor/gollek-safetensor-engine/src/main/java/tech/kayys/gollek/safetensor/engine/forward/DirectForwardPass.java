/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.metal.MetalComputeBackend;
import jakarta.enterprise.inject.Instance;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;

/**
 * Full transformer forward pass using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectForwardPass {
    private static final Logger log = Logger.getLogger(DirectForwardPass.class);

    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;
    @Inject
    Instance<MetalComputeBackend> metalBackendInstance;
    
    private MetalComputeBackend metal;

    @jakarta.annotation.PostConstruct
    void init() {
        if (metalBackendInstance.isResolvable()) {
            this.metal = metalBackendInstance.get();
        }
    }

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        AccelTensor embedded = embeddingLookup(embedTable, inputIds);
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        try {
            if (scale != 1.0f)
                embedded.close();
            return prefill(hidden, inputIds, weights, config, arch, kvCache);
        } finally {
            hidden.closeWithParent();
        }
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        long seqLen = embeddings.size(1);
        if (seqLen < 1) {
            throw new IllegalArgumentException("Invalid sequence length: " + seqLen + ". Prompt must result in at least one token.");
        }
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

        // Initial copy: embeddings -> hiddenASeg
        java.lang.foreign.MemorySegment.copy(embeddings.dataPtr(), 0, ws.getHiddenASeg(), 0, (long) seqLen * config.hiddenSize() * 4);

        java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
        java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            if (verbose) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d start\n", i, config.numHiddenLayers());
                System.err.flush();
            }
            transformerLayer(currentHidden, nextHidden, inputIds, weights, config, arch, kvCache, i, 0, (int) seqLen, ws);

            // Swap buffers
            java.lang.foreign.MemorySegment temp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = temp;
        }

        AccelTensor hidden = AccelTensor.view(currentHidden, embeddings.shape());
        
        AccelTensor normed;
        if (metal != null && metal.deviceName() != null
                && !metal.deviceName().contains("CPU")) {
            normed = AccelTensor.view(ws.getNormedAttnSeg(), hidden.shape());
            metal.rmsNorm(normed.dataPtr(), hidden.dataPtr(),
                    weights.get(arch.finalNormWeight()).dataPtr(), (int) hidden.size(-1),
                    (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                    arch.addOneToRmsNormWeight());
        }
        if (hidden != embeddings)
            hidden.close();

        AccelTensor lastPos = selectLastToken(normed, (int) seqLen);
        if (normed.dataPtr() != ws.getNormedAttnSeg())
            normed.close();

        AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
        if (lmHeadW == null && config.tieWordEmbeddings()) {
            lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
        }
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        AccelTensor logits = AccelOps.linear(lastPos, lmHeadW);
        lastPos.close();

        float finalSoftCap = arch.defaultFinalSoftCap();
        if (finalSoftCap > 0.0f) {
            AccelTensor scaled = AccelOps.mulScalar(logits, 1.0f / finalSoftCap);
            AccelTensor tanhed = AccelOps.tanh(scaled);
            AccelTensor capped = AccelOps.mulScalar(tanhed, finalSoftCap);
            logits.close();
            scaled.close();
            tanhed.close();
            logits = capped;
        }

        float[] result = logits.toFloatArray();
        logits.close();

        // Diagnostics
        double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (float f : result) {
            sum += f;
            if (f < min)
                min = f;
            if (f > max)
                max = f;
        }
        if (verbose && config.numAttentionHeads() > 0) { // Just a dummy condition to keep it quiet usually
            System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d\n", min, max, sum, result.length);

            // Print top 5 IDs for debugging
            List<Integer> topIndices = new ArrayList<>();
            for (int i = 0; i < result.length; i++)
                topIndices.add(i);
            topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
            for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
                int id = topIndices.get(k);
                System.err.printf("  Top %d: ID=%d, val=%f\n", k, id, result[id]);
            }
        }

        kvCache.advance((int) seqLen);
        return result;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor embedded = embeddingLookup(embedTable, new long[] { tokenId });
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        try {
            if (scale != 1.0f)
                embedded.close();
            long[] tokenIds = new long[] { tokenId };

            KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
            ws.ensureCapacity(config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

            // Initial copy: hidden -> hiddenASeg
            java.lang.foreign.MemorySegment.copy(hidden.dataPtr(), 0, ws.getHiddenASeg(), 0, config.hiddenSize() * 4);
            hidden.close();

            java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
            java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

            for (int i = 0; i < config.numHiddenLayers(); i++) {
                transformerLayer(currentHidden, nextHidden, tokenIds, weights, config, arch, kvCache, i, startPos, 1, ws);
                // Swap
                java.lang.foreign.MemorySegment temp = currentHidden;
                currentHidden = nextHidden;
                nextHidden = temp;
            }

            AccelTensor finalHidden = AccelTensor.view(currentHidden, new long[]{1, 1, config.hiddenSize()});
            
            AccelTensor normed;
            if (metal != null && metal.deviceName() != null
                    && !metal.deviceName().contains("CPU")) {
                normed = AccelTensor.view(ws.getNormedAttnSeg(), finalHidden.shape());
                metal.rmsNorm(normed.dataPtr(), finalHidden.dataPtr(),
                        weights.get(arch.finalNormWeight()).dataPtr(), (int) finalHidden.size(-1),
                        (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
            } else {
                normed = AccelOps.rmsNorm(finalHidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                        arch.addOneToRmsNormWeight());
            }
            finalHidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null && config.tieWordEmbeddings()) {
                lmHeadW = weights.get(arch.embedTokensWeight());
            }
            if (lmHeadW == null) {
                throw new IllegalStateException(
                        "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
            }
            AccelTensor logits = AccelOps.linear(normed, lmHeadW);
            if (normed.dataPtr() != ws.getNormedAttnSeg())
                normed.close();

            // Logit Soft-capping (Gemma-2)
            float finalCap = arch.defaultFinalSoftCap();
            if (finalCap > 0) {
                AccelTensor scaledIn = AccelOps.mulScalar(logits, 1.0f / finalCap);
                AccelTensor capped = AccelOps.tanh(scaledIn);
                AccelTensor scaledOut = AccelOps.mulScalar(capped, finalCap);
                logits.close();
                scaledIn.close();
                capped.close();
                logits = scaledOut;
            }

            float[] result = logits.toFloatArray();
            logits.close();
            kvCache.advance(1);
            return result;
        } finally {
            if (!hidden.isClosed())
                hidden.closeWithParent();
        }
    }

    private void transformerLayer(java.lang.foreign.MemorySegment hiddenIn, java.lang.foreign.MemorySegment hiddenOut, 
            long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config, 
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache, int layerIdx, int startPos, int seqLen, 
            KVCacheManager.KVCacheSession.ForwardWorkspace ws) {
        
        long[] hiddenShape = new long[]{1, seqLen, config.hiddenSize()};
        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        
        // Attention norm
        java.lang.foreign.MemorySegment normedAttnSeg = ws.getNormedAttnSeg();
        if (metal != null && metal.deviceName() != null && !metal.deviceName().contains("CPU")) {
            metal.rmsNorm(normedAttnSeg, hiddenIn,
                    weights.get(arch.layerAttentionNormWeight(layerIdx)).dataPtr(), config.hiddenSize(),
                    (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            // Fallback (slow but stable)
            AccelTensor in = AccelTensor.view(hiddenIn, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, weights.get(arch.layerAttentionNormWeight(layerIdx)), config.rmsNormEps(), arch.addOneToRmsNormWeight());
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedAttnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        FlashAttentionKernel.AttentionInput attnIn = new FlashAttentionKernel.AttentionInput(
                AccelTensor.view(normedAttnSeg, hiddenShape),
                weights.get(arch.layerQueryWeight(layerIdx)),
                weights.get(arch.layerKeyWeight(layerIdx)),
                weights.get(arch.layerValueWeight(layerIdx)),
                weights.get(arch.layerOutputWeight(layerIdx)),
                resolveBias(weights, arch.layerQueryBias(layerIdx)),
                resolveBias(weights, arch.layerKeyBias(layerIdx)),
                resolveBias(weights, arch.layerValueBias(layerIdx)),
                weights.get(arch.layerOutputBias(layerIdx)),
                arch, config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                weights.get(arch.layerQueryNormWeight(layerIdx)),
                weights.get(arch.layerKeyNormWeight(layerIdx)),
                weights.get(arch.layerPostAttnNormWeight(layerIdx)));

        if (verbose) {
            System.err.printf("[DEBUG] Layer %d Attention start\n", layerIdx);
            System.err.flush();
        }
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        
        // Residual Add: hiddenIn + attnOut -> hiddenOut (re-using hiddenOut for intermediate)
        if (metal != null && !metal.deviceName().contains("CPU")) {
            metal.add(hiddenIn, attnOut.dataPtr(), hiddenOut, hiddenShape);
        } else {
            AccelTensor inA = AccelTensor.view(hiddenIn, hiddenShape);
            AccelTensor res = AccelOps.add(inA, attnOut);
            java.lang.foreign.MemorySegment.copy(res.dataPtr(), 0, hiddenOut, 0, (long) seqLen * config.hiddenSize() * 4);
            res.close();
        }
        attnOut.close();

        // MLP
        AccelTensor preFfnNormW = weights.get(arch.layerPreFfnNormWeight(layerIdx));
        if (preFfnNormW == null) preFfnNormW = weights.get(arch.layerFfnNormWeight(layerIdx));

        java.lang.foreign.MemorySegment normedFfnSeg = ws.getNormedFfnSeg();
        if (metal != null && !metal.deviceName().contains("CPU")) {
            metal.rmsNorm(normedFfnSeg, hiddenOut, preFfnNormW.dataPtr(), config.hiddenSize(),
                    (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            AccelTensor in = AccelTensor.view(hiddenOut, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, preFfnNormW, config.rmsNormEps(), arch.addOneToRmsNormWeight());
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedFfnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        AccelTensor mlpOut = swigluFfn(AccelTensor.view(normedFfnSeg, hiddenShape),
                weights.get(arch.layerFfnGateWeight(layerIdx)), weights.get(arch.layerFfnGateBias(layerIdx)),
                weights.get(arch.layerFfnUpWeight(layerIdx)), weights.get(arch.layerFfnUpBias(layerIdx)),
                weights.get(arch.layerFfnDownWeight(layerIdx)), weights.get(arch.layerFfnDownBias(layerIdx)), ws);
        
        // Final Residual: hiddenOut (temp) + mlpOut -> hiddenOut
        if (metal != null && !metal.deviceName().contains("CPU")) {
            metal.add(hiddenOut, mlpOut.dataPtr(), hiddenOut, hiddenShape);
        } else {
            AccelTensor inA = AccelTensor.view(hiddenOut, hiddenShape);
            AccelTensor res = AccelOps.add(inA, mlpOut);
            java.lang.foreign.MemorySegment.copy(res.dataPtr(), 0, hiddenOut, 0, (long) seqLen * config.hiddenSize() * 4);
            res.close();
        }
        mlpOut.close();
    }

    private void checkStability(AccelTensor t, String label) {
        float[] arr = t.toFloatArray();
        int nan = 0;
        int inf = 0;
        for (float f : arr) {
            if (Float.isNaN(f))
                nan++;
            if (Float.isInfinite(f))
                inf++;
        }
        if (nan > 0 || inf > 0) {
        }
    }

    AccelTensor ffnNonGated(AccelTensor x, AccelTensor upW, AccelTensor downW) {
        AccelTensor up = AccelOps.linear(x, upW);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = AccelOps.linear(act, downW);
        act.close();
        return out;
    }

    public AccelTensor swigluFfn(AccelTensor x, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB) {
        return swigluFfn(x, gateW, gateB, upW, upB, downW, downB, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, KVCacheManager.KVCacheSession.ForwardWorkspace ws) {
        AccelTensor gate = AccelOps.linear(x, gateW);
        AccelTensor up = AccelOps.linear(x, upW);

        AccelTensor combined;
        if (metal != null && metal.deviceName() != null
                && !metal.deviceName().contains("CPU") && ws != null) {
            combined = AccelTensor.view(ws.getCombinedSeg(), gate.shape());
            metal.siluFfn(combined.dataPtr(), gate.dataPtr(), up.dataPtr(),
                    (int) gate.numel());
        } else {
            combined = AccelOps.swiglu(gate, up);
        }

        gate.close();
        up.close();
        AccelTensor out = AccelOps.linear(combined, downW);
        if (ws == null || combined.dataPtr() != ws.getCombinedSeg())
            combined.close();
        return out;
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        int seqLen = tokenIds.length;
        AccelTensor selected = embedTable.indexSelect(tokenIds);
        long hiddenSize = selected.size(1);
        return selected.reshape(1L, seqLen, hiddenSize);
    }

    private AccelTensor selectLastToken(AccelTensor hidden, int seqLen) {
        if (seqLen < 1) return hidden; // Safety fallback
        long hiddenSize = hidden.size(hidden.shape().length - 1);
        long lastTokenOffset = Math.max(0L, (long) (seqLen - 1) * hiddenSize);
        return AccelTensor.view(hidden.dataPtr().asSlice(lastTokenOffset * 4L), new long[] { 1, hiddenSize });
    }

    // ── Linear ────────────────────────────────────────────────────────

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        AccelTensor mm = AccelOps.linear(input, weight);
        if (bias != null) {
            AccelTensor out = AccelOps.add(mm, bias);
            mm.close();
            return out;
        }
        return mm;
    }

    private static AccelTensor resolveBias(Map<String, AccelTensor> weights, String key) {
        return key != null ? weights.get(key) : null;
    }
}
